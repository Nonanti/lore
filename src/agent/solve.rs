//! Solve: the multi-step tool loop (ReAct) — mode dispatcher, text and
//! native drivers, the shared epilogue, and procedure learning.
//!
//! Split from `agent/mod.rs` at the project's module-size threshold
//! (M25 convention). Driver design:
//! `docs/superpowers/specs/2026-07-24-native-tool-calling-design.md`.

use super::{Agent, MAX_SOLVE_STEPS, SOLVE_PRIOR_LIMIT, SOLVE_PRIOR_MIN_WILSON};
use crate::error::{LoreError, Result};
use crate::id::MemoryId;
use crate::memory::retrieval::wilson_lower_bound;
use crate::memory::{Memory, MemoryKind, Outcome, Query, Scored, Tier};
use crate::model::{ChatMessage, ContentBlock, Prompt, Thread, ToolMode};
use crate::tool::{catalog, parse_tool_call, tool_specs, ToolCall, ToolContext};
use std::sync::atomic::Ordering;

/// Prior-procedure bundle shared by both solve drivers.
struct SolvePriors {
    /// Recalled procedural candidates (dedup + reinforcement targets).
    priors: Vec<Scored<Memory>>,
    /// Proven-procedure hint lines injected into the prompt/system.
    hints: Vec<String>,
    /// Ids of injected procedures (Wilson failure attribution).
    injected: Vec<MemoryId>,
}

impl Agent {
    /// Multi-step tool loop (ReAct): **think → call tool → feed observation back →
    /// think again → ... → final response.** The model either returns a tool call JSON
    /// (executed, observation added to scratchpad) or plain text as the final response
    /// at each step. Tool errors are also fed back as observations — the model can self-correct.
    /// On the last step, tool rights expire and a final response is requested (loop guaranteed to terminate).
    ///
    /// Procedure learning: before the loop, similar past solutions (procedural tier)
    /// are retrieved; proven ones (Wilson ≥ [`SOLVE_PRIOR_MIN_WILSON`]) enter
    /// the prompt as guiding hints. On successful completion, a past procedure
    /// following the same tool sequence is reinforced with `Success` instead of
    /// creating a new record; otherwise a new procedure is learned. In a fallback
    /// run (ended by step limit), injected procedures receive `Failure` — Wilson evidence accumulates bidirectionally.
    pub async fn solve(&self, ctx: &ToolContext, input: &str, max_steps: usize) -> Result<String> {
        let max_steps = max_steps.clamp(1, MAX_SOLVE_STEPS);
        let priors = self.solve_priors(input).await;
        match self.effective_tool_mode() {
            ToolMode::Text => self.solve_text(ctx, input, max_steps, &priors).await,
            // Explicit native: an unsupported provider is a hard error.
            ToolMode::Native => self.solve_native(ctx, input, max_steps, &priors).await,
            ToolMode::Auto => {
                if !self.model.supports_native_tools()
                    || self.native_downgraded.load(Ordering::Relaxed)
                {
                    return self.solve_text(ctx, input, max_steps, &priors).await;
                }
                match self.solve_native(ctx, input, max_steps, &priors).await {
                    // Downgrade is only reachable from step 0 (no side effects
                    // yet — solve_native converts later-step occurrences), so
                    // the text rerun cannot repeat tool executions.
                    Err(LoreError::NativeToolsUnsupported(m)) => {
                        tracing::warn!(
                            reason = %m,
                            "native tool calling unavailable; downgrading agent to text protocol"
                        );
                        self.native_downgraded.store(true, Ordering::Relaxed);
                        self.solve_text(ctx, input, max_steps, &priors).await
                    }
                    r => r,
                }
            }
        }
    }

    /// Prior procedures: both dedup candidates and (if proven) hints.
    /// A recall failure is logged but not fatal — solve proceeds without priors.
    async fn solve_priors(&self, input: &str) -> SolvePriors {
        let priors = match self
            .recall(
                &Query::new(input)
                    .tier(Tier::Procedural)
                    .semantic()
                    .limit(SOLVE_PRIOR_LIMIT),
            )
            .await
        {
            Ok(p) => p,
            Err(e) => {
                tracing::warn!(error = %e, "solve: prior procedures could not be recalled");
                Vec::new()
            }
        };
        let mut hints: Vec<String> = Vec::new();
        let mut injected: Vec<MemoryId> = Vec::new();
        for p in &priors {
            if let MemoryKind::Procedural {
                title,
                steps,
                successes,
                failures,
            } = &p.item.kind
            {
                if wilson_lower_bound(*successes, *failures) >= SOLVE_PRIOR_MIN_WILSON {
                    hints.push(format!(
                        "[prior solution] {title} — steps followed: {}",
                        steps.join(" → ")
                    ));
                    injected.push(p.item.id.clone());
                }
            }
        }
        SolvePriors {
            priors,
            hints,
            injected,
        }
    }

    /// Text-protocol solve driver — the pre-native behavior, unchanged:
    /// instruct the model to emit `{"tool":..,"args":..}` JSON and parse it
    /// out of plain text. Serves `ToolMode::Text` and every provider without
    /// native support (`auto` downgrades land here).
    async fn solve_text(
        &self,
        ctx: &ToolContext,
        input: &str,
        max_steps: usize,
        sp: &SolvePriors,
    ) -> Result<String> {
        let catalog = catalog(&ctx.registry);
        // `hints` and `scratchpad` (observations) are kept separate: the
        // final record/note logic only considers actual observations.
        let mut scratchpad: Vec<String> = Vec::new();
        let mut calls: Vec<ToolCall> = Vec::new();
        // Was a tool error seen along the procedure path? (For Failure attribution —
        // the model's inability to stop is not the procedure's fault)
        let mut had_tool_error = false;
        for step in 0..max_steps {
            let last = step + 1 == max_steps;
            let instruction = if last {
                "No more tool calls. Give the FINAL answer based on observations as plain text."
                    .to_string()
            } else {
                format!(
                    "If using a tool, return ONLY this JSON: \
                     {{\"tool\":\"<name>\",\"args\":\"<argument — in tool args format>\"}}\n\
                     Available tools:\n{catalog}\n\
                     If the answer is ready, do not call a tool; give the final response as plain text."
                )
            };
            let prompt = Prompt {
                system: format!("{}\n\n{instruction}", self.persona.identity_prompt()),
                context: sp.hints.iter().chain(scratchpad.iter()).cloned().collect(),
                user: input.to_string(),
                ..Default::default()
            };
            let completion = self.model.complete(&prompt).await?;

            // Tool call? (no tool rights on the last step — text is accepted as final.)
            if !last {
                if let Some(call) = parse_tool_call(&completion.text) {
                    let (obs, ok) = match ctx.registry.get(&call.tool) {
                        Some(tool) => match tool.run(&call.args).await {
                            Ok(o) => (o, true),
                            // An error is also an observation: the model can correct in the next step.
                            Err(e) => (format!("ERROR: {e}"), false),
                        },
                        None => (format!("ERROR: no such tool '{}'", call.tool), false),
                    };
                    // Only SUCCESSFUL calls enter the learned procedure —
                    // failed attempts remain in observations, do not pollute the procedure.
                    if ok {
                        calls.push(call.clone());
                    } else {
                        had_tool_error = true;
                    }
                    scratchpad.push(format!(
                        "[observation] {}({}) → {}",
                        call.tool, call.args, obs
                    ));
                    continue;
                }
            }

            // Final response. If the model ignores the instruction on the last step
            // and returns a tool JSON again, do not leak raw JSON to the user
            // (reachable only from the `last` branch — on prior steps, JSON goes
            // through `continue` into the tool loop).
            let fell_back = parse_tool_call(&completion.text).is_some();
            return self
                .finish_solve(
                    input,
                    completion.text,
                    fell_back,
                    &scratchpad,
                    &calls,
                    had_tool_error,
                    sp,
                )
                .await;
        }
        // Unreachable: last step always returns; safety belt just in case.
        Err(crate::error::LoreError::Model(
            "solve step limit exceeded".into(),
        ))
    }

    /// Native solve driver: the thread protocol every provider tool API is
    /// trained on — assistant `tool_use` blocks answered by user
    /// `tool_result` blocks, tools travelling as structured specs instead of
    /// prompt text. One step = one model roundtrip; a step may execute
    /// several parallel tool calls.
    async fn solve_native(
        &self,
        ctx: &ToolContext,
        input: &str,
        max_steps: usize,
        sp: &SolvePriors,
    ) -> Result<String> {
        let specs = tool_specs(&ctx.registry);
        // Hints fold into system — the flat path does the same (providers
        // append Prompt.context lines to their system slot).
        let mut system = self.persona.identity_prompt();
        if !sp.hints.is_empty() {
            system.push_str("\n\nWhat you recall:\n");
            for h in &sp.hints {
                system.push_str("- ");
                system.push_str(h);
                system.push('\n');
            }
        }
        let mut thread = Thread::new(system);
        thread.push(ChatMessage::user_text(input));

        let mut scratchpad: Vec<String> = Vec::new();
        let mut calls: Vec<ToolCall> = Vec::new();
        let mut had_tool_error = false;
        for step in 0..max_steps {
            let last = step + 1 == max_steps;
            if last {
                // Tools stay in the request (Anthropic rejects tool-blocked
                // threads without a `tools` param) — the nudge plus the
                // unexecuted-ToolUse guard below enforce termination instead.
                // With max_steps == 1 the nudge lands before any tool ran;
                // that mirrors text mode exactly (its only step carries the
                // no-tools final-answer instruction) — deliberate parity,
                // not a bug.
                // NOTE: this produces CONSECUTIVE user messages (tool_result
                // → nudge, or task → nudge at max_steps=1). Live-verified
                // accepted by the Anthropic Messages API (2026-07-24, both
                // shapes); if a future provider requires strict role
                // alternation, merge the nudge into the previous user
                // message instead of "fixing" the loop.
                thread.push(ChatMessage::user_text(
                    "No more tool calls — give the final answer based on the results above.",
                ));
            }
            let reply = match self.model.complete_thread(&thread, &specs).await {
                Ok(r) => r,
                // Step 0 unsupported is clean (nothing executed) — `auto` may
                // downgrade and rerun. Later steps have run tools; a rerun
                // would repeat side effects, so surface a plain model error.
                Err(LoreError::NativeToolsUnsupported(m)) if step > 0 => {
                    return Err(LoreError::Model(format!(
                        "native tools became unavailable mid-run: {m}"
                    )));
                }
                Err(e) => return Err(e),
            };

            // Owned copies release the borrow so reply.blocks can move below.
            let uses: Vec<(String, String, serde_json::Value)> = reply
                .tool_uses()
                .iter()
                .map(|u| (u.id.to_string(), u.name.to_string(), u.input.clone()))
                .collect();
            if !last && !uses.is_empty() {
                let mut results: Vec<ContentBlock> = Vec::new();
                for (id, name, input_v) in &uses {
                    let (obs, ok, args) = match ctx.registry.get(name) {
                        Some(tool) => {
                            let args = tool.args_from_input(input_v);
                            match tool.run(&args).await {
                                Ok(o) => (o, true, args),
                                // An error is also an observation: the model
                                // can correct in the next step.
                                Err(e) => (format!("ERROR: {e}"), false, args),
                            }
                        }
                        None => (
                            format!("ERROR: no such tool '{name}'"),
                            false,
                            serde_json::to_string(input_v).unwrap_or_default(),
                        ),
                    };
                    // Only SUCCESSFUL calls enter the learned procedure —
                    // failed attempts remain in observations.
                    if ok {
                        calls.push(ToolCall {
                            tool: name.clone(),
                            args: args.clone(),
                        });
                    } else {
                        had_tool_error = true;
                    }
                    scratchpad.push(format!("[observation] {name}({args}) → {obs}"));
                    results.push(ContentBlock::ToolResult {
                        tool_use_id: id.clone(),
                        content: obs,
                        is_error: !ok,
                    });
                }
                thread.push(ChatMessage::assistant_blocks(reply.blocks));
                thread.push(ChatMessage::tool_results(results));
                continue;
            }

            // Final. An unexecuted ToolUse on the last step means the model
            // ignored the nudge — the fell-back guard answers from the last
            // observation (raw blocks never leak to the user).
            let fell_back = !uses.is_empty();
            return self
                .finish_solve(
                    input,
                    reply.text(),
                    fell_back,
                    &scratchpad,
                    &calls,
                    had_tool_error,
                    sp,
                )
                .await;
        }
        // Unreachable: last step always returns; safety belt just in case.
        Err(crate::error::LoreError::Model(
            "solve step limit exceeded".into(),
        ))
    }

    /// Shared solve epilogue — identical for both drivers: fell-back text
    /// substitution, Wilson failure attribution, exchange/note memory, and
    /// procedure learning.
    #[allow(clippy::too_many_arguments)]
    async fn finish_solve(
        &self,
        input: &str,
        final_text: String,
        fell_back: bool,
        scratchpad: &[String],
        calls: &[ToolCall],
        had_tool_error: bool,
        sp: &SolvePriors,
    ) -> Result<String> {
        let text = if fell_back {
            match scratchpad.last() {
                Some(obs) => format!("step limit reached; last info: {obs}"),
                None => "step limit reached; no final response generated.".to_string(),
            }
        } else {
            if final_text.trim().is_empty() {
                // Same propagation as always (both drivers, both protocols) —
                // but an empty final is worth a diagnostic trail.
                tracing::warn!(input, "solve produced an empty final answer");
            }
            final_text
        };
        if fell_back && !sp.injected.is_empty() && had_tool_error {
            // Failure is processed ONLY if a tool error was seen along the procedure path.
            // Hitting the step limit alone is not evidence against the procedure —
            // the model may simply "not know when to stop" (no unfair penalty).
            if let Err(e) = self
                .memory
                .reinforce_many(&sp.injected, Outcome::Failure)
                .await
            {
                tracing::warn!(error = %e, "procedure failure could not be processed");
            }
        }
        if scratchpad.is_empty() {
            self.remember_exchange(input, &text).await?;
        } else {
            // Automatic trace: unused, decay reclaims it; accessed, it is preserved.
            self.note(
                format!(
                    "completed task '{input}' with {} tool steps",
                    scratchpad.len()
                ),
                format!("{}\nResult: {text}", scratchpad.join("\n")),
            )
            .await?;
            if !fell_back && !calls.is_empty() {
                self.learn_procedure(input, calls, &sp.priors).await;
            }
        }
        Ok(text)
    }

    /// Learns a successful tool sequence as a procedure.
    ///
    /// If a past procedure follows the same tool sequence (ordered tool names),
    /// no duplicate is created — the existing record is reinforced with `Success` (Wilson
    /// evidence accumulates, decay protection strengthens). Otherwise, a new
    /// procedural record is opened with steps in `tool: args` format and its first success is processed.
    /// A learning error never corrupts the solve result (logged and skipped).
    async fn learn_procedure(&self, input: &str, calls: &[ToolCall], priors: &[Scored<Memory>]) {
        let seq: Vec<&str> = calls.iter().map(|c| c.tool.as_str()).collect();
        for p in priors {
            if let MemoryKind::Procedural { steps, .. } = &p.item.kind {
                let prior_seq: Vec<&str> = steps
                    .iter()
                    .map(|s| s.split(':').next().unwrap_or(s.as_str()).trim())
                    .collect();
                if prior_seq == seq {
                    if let Err(e) = self
                        .memory
                        .reinforce_many(std::slice::from_ref(&p.item.id), Outcome::Success)
                        .await
                    {
                        tracing::warn!(error = %e, "procedure could not be reinforced");
                    }
                    return;
                }
            }
        }
        let steps: Vec<String> = calls
            .iter()
            .map(|c| format!("{}: {}", c.tool, c.args))
            .collect();
        let mem = Memory::procedural(self.scope(), format!("task '{input}'"), steps);
        match self.remember(mem).await {
            Ok(id) => {
                if let Err(e) = self.memory.reinforce(&id, Outcome::Success).await {
                    tracing::warn!(error = %e, "procedure first success could not be processed");
                }
            }
            Err(e) => tracing::warn!(error = %e, "procedure could not be saved"),
        }
    }
}
