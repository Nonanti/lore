//! Agent test suite — split from `agent/mod.rs` at the module-size
//! threshold (same pattern as `agent/work/tests.rs`).

use super::*;
use crate::memory::{InMemoryStore, SemanticCat};
use crate::model::{ContentBlock, MockModel, Thread};

fn agent_with(name: &str, store: Arc<dyn MemoryStore>) -> Agent {
    let persona = Persona::new(name, "researcher").with_trait("curious");
    Agent::new(persona, store, Arc::new(MockModel::new()))
}

#[tokio::test]
async fn respond_uses_recalled_memory_and_records_episode() {
    let store: Arc<dyn MemoryStore> = Arc::new(InMemoryStore::new());
    let agent = agent_with("Aria", store.clone());

    agent
        .experience("Learned Rust", "Studied ownership and borrow checker")
        .await
        .unwrap();

    let before = agent.recall(&Query::new("rust")).await.unwrap().len();
    assert_eq!(before, 1);

    let reply = agent.respond("what do you know about rust").await.unwrap();
    // Recalled context should be reflected in the response.
    assert!(reply.contains("recalling"));

    // A new episodic memory was added after respond.
    let after = agent.recall(&Query::new("rust")).await.unwrap().len();
    assert!(after > before);
}

#[tokio::test]
async fn converse_carries_working_memory_across_turns() {
    let store: Arc<dyn MemoryStore> = Arc::new(InMemoryStore::new());
    let agent = agent_with("Aria", store);
    let mut convo = Conversation::new();

    // First turn: no history.
    let first = agent.converse(&mut convo, "hi").await.unwrap();
    assert!(!first.contains("chat history"), "no history on first turn");
    assert_eq!(convo.len(), 2, "exchange recorded to window");

    // Second turn: previous 2 messages (user+assistant) are included in the prompt.
    let second = agent.converse(&mut convo, "what did I say?").await.unwrap();
    assert!(
        second.contains("chat history: 2 messages"),
        "model saw history: {second}"
    );
    assert_eq!(convo.len(), 4);

    // respond, however, remains without history (behavior unchanged).
    let plain = agent.respond("hello again").await.unwrap();
    assert!(!plain.contains("chat history"));
}

/// Test model that returns scripted replies in sequence (for ReAct scenarios).
struct SeqModel(std::sync::Mutex<std::collections::VecDeque<String>>);
impl SeqModel {
    fn new(replies: &[&str]) -> Self {
        Self(std::sync::Mutex::new(
            replies.iter().map(|s| s.to_string()).collect(),
        ))
    }
}
#[async_trait::async_trait]
impl Model for SeqModel {
    async fn complete(&self, _p: &Prompt) -> Result<crate::model::Completion> {
        let text = self
            .0
            .lock()
            .unwrap()
            .pop_front()
            .unwrap_or_else(|| "no reply left".into());
        Ok(crate::model::Completion::new(text))
    }
}

fn calc_ctx() -> ToolContext {
    use crate::tool::{CalcTool, KeywordRouter};
    let mut reg = ToolRegistry::new();
    reg.register(Arc::new(CalcTool::new()));
    ToolContext {
        registry: reg,
        router: Arc::new(KeywordRouter::new()),
    }
}

// ── Native tool-calling driver ──────────────────────────────────

use crate::model::{ChatRole, StopReason, ThreadReply, ToolSpec};

/// Scripted native-tools model: queued thread outcomes + queued text
/// replies (text-protocol fallback), capturing every Thread and counting
/// `complete_thread` calls.
enum ThreadStep {
    Reply(ThreadReply),
    Unsupported(String),
}
struct ThreadSeqModel {
    steps: std::sync::Mutex<std::collections::VecDeque<ThreadStep>>,
    texts: std::sync::Mutex<std::collections::VecDeque<String>>,
    threads: Arc<std::sync::Mutex<Vec<Thread>>>,
    thread_calls: Arc<std::sync::atomic::AtomicUsize>,
}
impl ThreadSeqModel {
    fn new(steps: Vec<ThreadStep>, texts: &[&str]) -> Self {
        Self {
            steps: std::sync::Mutex::new(steps.into()),
            texts: std::sync::Mutex::new(texts.iter().map(|s| s.to_string()).collect()),
            threads: Arc::new(std::sync::Mutex::new(Vec::new())),
            thread_calls: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        }
    }
    fn reply(blocks: Vec<ContentBlock>, stop: StopReason) -> ThreadStep {
        ThreadStep::Reply(ThreadReply {
            blocks,
            stop,
            reasoning_fallback: false,
        })
    }
    fn use_block(id: &str, name: &str, args: &str) -> ContentBlock {
        ContentBlock::ToolUse {
            id: id.into(),
            name: name.into(),
            input: serde_json::json!({ "args": args }),
        }
    }
    fn text_block(t: &str) -> ContentBlock {
        ContentBlock::Text { text: t.into() }
    }
}
#[async_trait::async_trait]
impl Model for ThreadSeqModel {
    async fn complete(&self, _p: &Prompt) -> Result<crate::model::Completion> {
        let text = self
            .texts
            .lock()
            .unwrap()
            .pop_front()
            .unwrap_or_else(|| "no text reply left".into());
        Ok(crate::model::Completion::new(text))
    }
    async fn complete_thread(&self, thread: &Thread, _tools: &[ToolSpec]) -> Result<ThreadReply> {
        self.thread_calls
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        self.threads.lock().unwrap().push(thread.clone());
        match self.steps.lock().unwrap().pop_front() {
            Some(ThreadStep::Reply(r)) => Ok(r),
            Some(ThreadStep::Unsupported(m)) => Err(LoreError::NativeToolsUnsupported(m)),
            None => Ok(ThreadReply {
                blocks: vec![ThreadSeqModel::text_block("no reply left")],
                stop: StopReason::EndTurn,
                reasoning_fallback: false,
            }),
        }
    }
    fn supports_native_tools(&self) -> bool {
        true
    }
}

#[tokio::test]
async fn solve_native_chains_tool_results_back() {
    let store: Arc<dyn MemoryStore> = Arc::new(InMemoryStore::new());
    let model = ThreadSeqModel::new(
        vec![
            ThreadSeqModel::reply(
                vec![ThreadSeqModel::use_block("t1", "calc", "3 + 4")],
                StopReason::ToolUse,
            ),
            ThreadSeqModel::reply(
                vec![ThreadSeqModel::text_block("The result is 7.")],
                StopReason::EndTurn,
            ),
        ],
        &[],
    );
    let threads = model.threads.clone();
    let agent = Agent::new(Persona::new("Aria", "solver"), store, Arc::new(model));

    let out = agent.solve(&calc_ctx(), "3+4?", 5).await.unwrap();
    assert_eq!(out, "The result is 7.");

    // Second roundtrip carried the full native protocol: user task →
    // assistant tool_use → user tool_result (correlated, no error flag).
    // (Guard scoped: recall() awaits after this block.)
    {
        let seen = threads.lock().unwrap();
        assert_eq!(seen.len(), 2);
        let t2 = &seen[1];
        assert!(t2.system.contains("Aria"), "identity in system");
        assert_eq!(t2.messages.len(), 3);
        assert_eq!(t2.messages[0].role, ChatRole::User);
        assert_eq!(t2.messages[1].role, ChatRole::Assistant);
        assert!(matches!(
            &t2.messages[1].blocks[0],
            ContentBlock::ToolUse { id, name, .. } if id == "t1" && name == "calc"
        ));
        assert_eq!(t2.messages[2].role, ChatRole::User);
        match &t2.messages[2].blocks[0] {
            ContentBlock::ToolResult {
                tool_use_id,
                content,
                is_error,
            } => {
                assert_eq!(tool_use_id, "t1");
                assert_eq!(content, "7");
                assert!(!is_error);
            }
            other => panic!("expected ToolResult, got {other:?}"),
        }
    }

    // Tool trace remembered like the text path.
    let mems = agent.recall(&Query::new("task")).await.unwrap();
    assert!(mems
        .iter()
        .any(|m| m.item.summary().contains("1 tool steps")));
}

#[tokio::test]
async fn solve_native_executes_parallel_calls_in_order() {
    let store: Arc<dyn MemoryStore> = Arc::new(InMemoryStore::new());
    let model = ThreadSeqModel::new(
        vec![
            ThreadSeqModel::reply(
                vec![
                    ThreadSeqModel::use_block("a", "calc", "1 + 1"),
                    ThreadSeqModel::use_block("b", "calc", "2 + 2"),
                ],
                StopReason::ToolUse,
            ),
            ThreadSeqModel::reply(
                vec![ThreadSeqModel::text_block("2 and 4.")],
                StopReason::EndTurn,
            ),
        ],
        &[],
    );
    let threads = model.threads.clone();
    let agent = Agent::new(Persona::new("Aria", "solver"), store, Arc::new(model));

    let out = agent.solve(&calc_ctx(), "1+1 and 2+2?", 5).await.unwrap();
    assert_eq!(out, "2 and 4.");

    // One results message carrying BOTH tool results, input order.
    let seen = threads.lock().unwrap();
    let results = &seen[1].messages[2].blocks;
    assert_eq!(results.len(), 2);
    match (&results[0], &results[1]) {
        (
            ContentBlock::ToolResult {
                tool_use_id: id0,
                content: c0,
                ..
            },
            ContentBlock::ToolResult {
                tool_use_id: id1,
                content: c1,
                ..
            },
        ) => {
            assert_eq!((id0.as_str(), c0.as_str()), ("a", "2"));
            assert_eq!((id1.as_str(), c1.as_str()), ("b", "4"));
        }
        other => panic!("expected two ToolResults, got {other:?}"),
    }
}

#[tokio::test]
async fn solve_native_flags_tool_errors_and_recovers() {
    let store: Arc<dyn MemoryStore> = Arc::new(InMemoryStore::new());
    let model = ThreadSeqModel::new(
        vec![
            // Unknown tool, then bad args, then a good call, then final.
            ThreadSeqModel::reply(
                vec![ThreadSeqModel::use_block("x", "google", "q")],
                StopReason::ToolUse,
            ),
            ThreadSeqModel::reply(
                vec![ThreadSeqModel::use_block("y", "calc", "5 5")],
                StopReason::ToolUse,
            ),
            ThreadSeqModel::reply(
                vec![ThreadSeqModel::use_block("z", "calc", "5 + 5")],
                StopReason::ToolUse,
            ),
            ThreadSeqModel::reply(vec![ThreadSeqModel::text_block("10.")], StopReason::EndTurn),
        ],
        &[],
    );
    let threads = model.threads.clone();
    let agent = Agent::new(Persona::new("Aria", "solver"), store, Arc::new(model));

    let out = agent.solve(&calc_ctx(), "5+5?", 6).await.unwrap();
    assert_eq!(out, "10.");

    let seen = threads.lock().unwrap();
    // Unknown tool → is_error with a helpful message.
    match &seen[1].messages[2].blocks[0] {
        ContentBlock::ToolResult {
            content, is_error, ..
        } => {
            assert!(is_error);
            assert!(content.contains("no such tool 'google'"), "{content}");
        }
        other => panic!("expected ToolResult, got {other:?}"),
    }
    // Bad args → is_error from the tool itself.
    match &seen[2].messages[4].blocks[0] {
        ContentBlock::ToolResult {
            content, is_error, ..
        } => {
            assert!(is_error);
            assert!(content.starts_with("ERROR:"), "{content}");
        }
        other => panic!("expected ToolResult, got {other:?}"),
    }
}

#[tokio::test]
async fn solve_native_last_step_nudges_and_never_leaks_blocks() {
    let store: Arc<dyn MemoryStore> = Arc::new(InMemoryStore::new());
    let model = ThreadSeqModel::new(
        vec![
            ThreadSeqModel::reply(
                vec![ThreadSeqModel::use_block("t1", "calc", "1 + 1")],
                StopReason::ToolUse,
            ),
            // Last step: model ignores the nudge and calls a tool again.
            ThreadSeqModel::reply(
                vec![ThreadSeqModel::use_block("t2", "calc", "9 + 9")],
                StopReason::ToolUse,
            ),
        ],
        &[],
    );
    let threads = model.threads.clone();
    let agent = Agent::new(Persona::new("Aria", "solver"), store, Arc::new(model));

    let out = agent.solve(&calc_ctx(), "sum?", 2).await.unwrap();
    // Unexecuted ToolUse → fell-back answer from the last observation.
    assert!(out.contains("step limit reached"), "{out}");
    assert!(out.contains("calc(1 + 1) → 2"), "{out}");

    // The nudge was appended before the last roundtrip.
    let seen = threads.lock().unwrap();
    let last_thread = seen.last().unwrap();
    let nudge = last_thread
        .messages
        .iter()
        .rev()
        .find(|m| m.role == ChatRole::User)
        .unwrap();
    assert!(matches!(
        &nudge.blocks[0],
        ContentBlock::Text { text } if text.contains("No more tool calls")
    ));
}

#[tokio::test]
async fn solve_auto_downgrades_once_and_latches() {
    let store: Arc<dyn MemoryStore> = Arc::new(InMemoryStore::new());
    let model = ThreadSeqModel::new(
        vec![ThreadStep::Unsupported("registry: no tools".into())],
        &["first text answer", "second text answer"],
    );
    let calls_n = model.thread_calls.clone();
    let agent = Agent::new(Persona::new("Aria", "solver"), store, Arc::new(model));

    // First solve: native probe fails cleanly → text fallback answers.
    let out = agent.solve(&calc_ctx(), "q1", 3).await.unwrap();
    assert_eq!(out, "first text answer");
    assert_eq!(calls_n.load(std::sync::atomic::Ordering::SeqCst), 1);

    // Second solve: the latch skips the doomed probe entirely.
    let out2 = agent.solve(&calc_ctx(), "q2", 3).await.unwrap();
    assert_eq!(out2, "second text answer");
    assert_eq!(calls_n.load(std::sync::atomic::Ordering::SeqCst), 1);
}

#[tokio::test]
async fn solve_native_midrun_unsupported_is_plain_model_error() {
    // Side-effect safety invariant: NativeToolsUnsupported AFTER tools
    // ran (step > 0) must surface as a plain Model error — never as the
    // typed downgrade error — so auto mode cannot rerun executed tools.
    let store: Arc<dyn MemoryStore> = Arc::new(InMemoryStore::new());
    let model = ThreadSeqModel::new(
        vec![
            ThreadSeqModel::reply(
                vec![ThreadSeqModel::use_block("t1", "calc", "1 + 1")],
                StopReason::ToolUse,
            ),
            ThreadStep::Unsupported("backend flipped mid-run".into()),
        ],
        &["text fallback must NOT be reached"],
    );
    let calls_n = model.thread_calls.clone();
    let agent = Agent::new(Persona::new("Aria", "solver"), store, Arc::new(model));

    let err = agent.solve(&calc_ctx(), "sum?", 4).await.unwrap_err();
    assert!(matches!(err, LoreError::Model(_)), "got: {err}");
    assert!(err.to_string().contains("mid-run"), "got: {err}");
    // The downgrade latch must NOT be set: the next solve still probes
    // native (two prior thread calls + one new).
    let _ = agent.solve(&calc_ctx(), "again?", 2).await;
    assert_eq!(calls_n.load(std::sync::atomic::Ordering::SeqCst), 3);
}

#[tokio::test]
async fn solve_native_mode_hard_errors_when_unsupported() {
    let store: Arc<dyn MemoryStore> = Arc::new(InMemoryStore::new());
    let model = ThreadSeqModel::new(
        vec![ThreadStep::Unsupported("nope".into())],
        &["should not be used"],
    );
    let agent = Agent::new(Persona::new("Aria", "solver"), store, Arc::new(model))
        .with_tool_mode(ToolMode::Native);

    let err = agent.solve(&calc_ctx(), "q", 3).await.unwrap_err();
    assert!(
        matches!(err, LoreError::NativeToolsUnsupported(_)),
        "got: {err}"
    );
}

#[tokio::test]
async fn solve_text_mode_never_touches_complete_thread() {
    let store: Arc<dyn MemoryStore> = Arc::new(InMemoryStore::new());
    let model = ThreadSeqModel::new(
        vec![ThreadSeqModel::reply(
            vec![ThreadSeqModel::text_block("native would answer")],
            StopReason::EndTurn,
        )],
        &["text answer"],
    );
    let calls_n = model.thread_calls.clone();
    let agent = Agent::new(Persona::new("Aria", "solver"), store, Arc::new(model))
        .with_tool_mode(ToolMode::Text);

    let out = agent.solve(&calc_ctx(), "q", 3).await.unwrap();
    assert_eq!(out, "text answer");
    assert_eq!(calls_n.load(std::sync::atomic::Ordering::SeqCst), 0);
}

#[tokio::test]
async fn solve_chains_tools_and_feeds_observations_back() {
    // Scenario: (3+4)*6 — model chains two tool steps, then gives a final response.
    let store: Arc<dyn MemoryStore> = Arc::new(InMemoryStore::new());
    let model = Arc::new(SeqModel::new(&[
        r#"{"tool":"calc","args":"3 + 4"}"#,
        r#"{"tool":"calc","args":"7 * 6"}"#,
        "The result is 42.",
    ]));
    let agent = Agent::new(Persona::new("Aria", "solver"), store, model);

    let out = agent.solve(&calc_ctx(), "(3+4)*6?", 5).await.unwrap();
    assert_eq!(out, "The result is 42.");

    // Two tool steps were remembered as a procedural trace.
    let mems = agent.recall(&Query::new("task")).await.unwrap();
    assert!(!mems.is_empty());
    assert!(mems
        .iter()
        .find(|m| m.item.summary().contains("2 tool steps"))
        .expect("should find tool steps note")
        .item
        .summary()
        .contains("2 tool steps"));
}

#[tokio::test]
async fn solve_recovers_from_tool_error_and_bad_tool() {
    // Model tries a non-existent tool first, then bad arguments; corrects via observations.
    let store: Arc<dyn MemoryStore> = Arc::new(InMemoryStore::new());
    let model = Arc::new(SeqModel::new(&[
        r#"{"tool":"google","args":"x"}"#,
        r#"{"tool":"calc","args":"5 5"}"#,
        r#"{"tool":"calc","args":"5 + 5"}"#,
        "The answer is 10.",
    ]));
    let agent = Agent::new(Persona::new("Aria", "solver"), store, model);

    let out = agent.solve(&calc_ctx(), "5+5", 6).await.unwrap();
    assert_eq!(out, "The answer is 10.");
}

/// Scripted model that records Prompt.context on each call (injection test).
struct CaptureCtxModel {
    replies: std::sync::Mutex<std::collections::VecDeque<String>>,
    seen: Arc<std::sync::Mutex<Vec<Vec<String>>>>,
}
impl CaptureCtxModel {
    fn new(replies: &[&str], seen: Arc<std::sync::Mutex<Vec<Vec<String>>>>) -> Self {
        Self {
            replies: std::sync::Mutex::new(replies.iter().map(|s| s.to_string()).collect()),
            seen,
        }
    }
}
#[async_trait::async_trait]
impl Model for CaptureCtxModel {
    async fn complete(&self, p: &Prompt) -> Result<crate::model::Completion> {
        self.seen.lock().unwrap().push(p.context.clone());
        let text = self
            .replies
            .lock()
            .unwrap()
            .pop_front()
            .unwrap_or_else(|| "no reply left".into());
        Ok(crate::model::Completion::new(text))
    }
}

#[tokio::test]
async fn solve_success_learns_procedure_with_wilson() {
    // Successful tool chain becomes a Procedural record (H1: Wilson is fed).
    let store: Arc<dyn MemoryStore> = Arc::new(InMemoryStore::new());
    let model = Arc::new(SeqModel::new(&[
        r#"{"tool":"calc","args":"3 + 4"}"#,
        r#"{"tool":"calc","args":"7 * 6"}"#,
        "The result is 42.",
    ]));
    let agent = Agent::new(Persona::new("Aria", "solver"), store, model);

    agent.solve(&calc_ctx(), "(3+4)*6?", 5).await.unwrap();

    let procs = agent
        .recall(&Query::new("?").tier(crate::memory::Tier::Procedural))
        .await
        .unwrap();
    assert_eq!(procs.len(), 1, "one procedure should be learned");
    let m = &procs[0].item;
    assert!(
        m.summary().contains("1\u{2713}/0\u{2717}"),
        "first success processed: {}",
        m.summary()
    );
    let crate::memory::MemoryKind::Procedural { steps, .. } = &m.kind else {
        panic!("expected procedural");
    };
    assert_eq!(
        steps,
        &vec!["calc: 3 + 4".to_string(), "calc: 7 * 6".to_string()]
    );
}

#[tokio::test]
async fn repeated_solve_reinforces_instead_of_duplicating() {
    // Similar task using the same tool sequence: existing procedure is reinforced
    // with Success instead of creating a new record (Wilson evidence accumulates, no dup bloat).
    let store: Arc<dyn MemoryStore> = Arc::new(InMemoryStore::new());
    let model = Arc::new(SeqModel::new(&[
        // First task.
        r#"{"tool":"calc","args":"3 + 4"}"#,
        r#"{"tool":"calc","args":"7 * 6"}"#,
        "The result is 42.",
        // Similar task — same tool sequence, different arguments.
        r#"{"tool":"calc","args":"5 + 2"}"#,
        r#"{"tool":"calc","args":"7 * 6"}"#,
        "The result is again 42.",
    ]));
    let agent = Agent::new(Persona::new("Aria", "solver"), store, model);

    agent.solve(&calc_ctx(), "(3+4)*6?", 5).await.unwrap();
    agent.solve(&calc_ctx(), "(5+2)*6?", 5).await.unwrap();

    let procs = agent
        .recall(&Query::new("?").tier(crate::memory::Tier::Procedural))
        .await
        .unwrap();
    assert_eq!(procs.len(), 1, "no duplicate procedure should be created");
    assert!(
        procs[0].item.summary().contains("2\u{2713}/0\u{2717}"),
        "repeated success should reinforce existing procedure: {}",
        procs[0].item.summary()
    );
}

#[tokio::test]
async fn proven_procedure_steps_enter_solve_prompt() {
    // Proven procedure (Wilson ≥ threshold) enters the next solve's prompt
    // as a guiding hint — also feeds Wilson behavior.
    let store: Arc<dyn MemoryStore> = Arc::new(InMemoryStore::new());
    let model = Arc::new(SeqModel::new(&[
        r#"{"tool":"calc","args":"3 + 4"}"#,
        "The answer is 7.",
    ]));
    let agent = Agent::new(Persona::new("Aria", "solver"), store.clone(), model);
    agent.solve(&calc_ctx(), "3+4?", 5).await.unwrap();

    // Second solve: capture prompts (SAME identity — required for scope isolation).
    let seen: Arc<std::sync::Mutex<Vec<Vec<String>>>> = Arc::new(std::sync::Mutex::new(Vec::new()));
    let cap = Arc::new(CaptureCtxModel::new(
        &["The answer is again 7."],
        seen.clone(),
    ));
    let mut agent2 = Agent::new(Persona::new("Aria", "solver"), store, cap);
    agent2.id = agent.id.clone();
    agent2.solve(&calc_ctx(), "3+4?", 5).await.unwrap();

    let seen = seen.lock().unwrap();
    let first = seen.first().expect("at least one prompt");
    assert!(
        first.iter().any(|c| c.contains("calc: 3 + 4")),
        "prior procedure steps should enter prompt: {first:?}"
    );
}

#[tokio::test]
async fn fallback_without_tool_errors_does_not_penalize_procedure() {
    // If the model keeps calling tools and hits the limit (tools all SUCCESSFUL),
    // this is not the procedure's fault — Failure is NOT applied (no noisy signal).
    let store: Arc<dyn MemoryStore> = Arc::new(InMemoryStore::new());
    let model = Arc::new(SeqModel::new(&[
        r#"{"tool":"calc","args":"3 + 4"}"#,
        "The answer is 7.",
    ]));
    let agent = Agent::new(Persona::new("Aria", "solver"), store.clone(), model);
    agent.solve(&calc_ctx(), "3+4?", 5).await.unwrap();

    // Second solve: model never gives a final response, but tools are all SUCCESSFUL.
    let stuck = Arc::new(SeqModel::new(&[
        r#"{"tool":"calc","args":"1 + 1"}"#,
        r#"{"tool":"calc","args":"1 + 1"}"#,
        r#"{"tool":"calc","args":"1 + 1"}"#,
        r#"{"tool":"calc","args":"1 + 1"}"#,
        r#"{"tool":"calc","args":"1 + 1"}"#,
    ]));
    let mut agent2 = Agent::new(Persona::new("Aria", "solver"), store, stuck);
    agent2.id = agent.id.clone();
    let out = agent2.solve(&calc_ctx(), "3+4?", 5).await.unwrap();
    assert!(
        out.contains("step limit reached"),
        "should end with fallback: {out}"
    );

    let procs = agent2
        .recall(&Query::new("?").tier(crate::memory::Tier::Procedural))
        .await
        .unwrap();
    assert!(
        procs[0].item.summary().contains("1\u{2713}/0\u{2717}"),
        "procedure not penalized without tool errors: {}",
        procs[0].item.summary()
    );
}

#[tokio::test]
async fn fallback_with_tool_errors_marks_injected_procedure_failure() {
    // Fallback + tool error along the procedure path: evidence against the procedure —
    // Failure is applied (Wilson penalizes, decay protection weakens).
    let store: Arc<dyn MemoryStore> = Arc::new(InMemoryStore::new());
    let model = Arc::new(SeqModel::new(&[
        r#"{"tool":"calc","args":"3 + 4"}"#,
        "The answer is 7.",
    ]));
    let agent = Agent::new(Persona::new("Aria", "solver"), store.clone(), model);
    agent.solve(&calc_ctx(), "3+4?", 5).await.unwrap();

    // Second solve: non-existent tool (ERROR observation) + unstoppable model.
    let stuck = Arc::new(SeqModel::new(&[
        r#"{"tool":"nonexistent","args":"x"}"#,
        r#"{"tool":"calc","args":"1 + 1"}"#,
        r#"{"tool":"calc","args":"1 + 1"}"#,
        r#"{"tool":"calc","args":"1 + 1"}"#,
        r#"{"tool":"calc","args":"1 + 1"}"#,
    ]));
    let mut agent2 = Agent::new(Persona::new("Aria", "solver"), store, stuck);
    agent2.id = agent.id.clone();
    let out = agent2.solve(&calc_ctx(), "3+4?", 5).await.unwrap();
    assert!(
        out.contains("step limit reached"),
        "should end with fallback: {out}"
    );

    let procs = agent2
        .recall(&Query::new("?").tier(crate::memory::Tier::Procedural))
        .await
        .unwrap();
    assert!(
        procs[0].item.summary().contains("1\u{2713}/1\u{2717}"),
        "failed path should apply Failure to procedure: {}",
        procs[0].item.summary()
    );
}

#[tokio::test]
async fn exchange_records_are_auto_importance() {
    // Exchange records must be born with automatic importance so decay can reclaim them;
    // explicit experience() records keep their default (higher) importance.
    let store: Arc<dyn MemoryStore> = Arc::new(InMemoryStore::new());
    let agent = agent_with("Aria", store.clone());
    agent.respond("hi").await.unwrap();
    agent
        .experience("important", "explicit record")
        .await
        .unwrap();

    let all = agent.recall(&Query::new("").limit(10)).await.unwrap();
    let auto = all
        .iter()
        .find(|s| s.item.summary().contains("responded"))
        .expect("exchange record must exist");
    assert_eq!(auto.item.importance, Memory::AUTO_IMPORTANCE);
    let explicit = all
        .iter()
        .find(|s| s.item.summary().contains("important"))
        .expect("explicit record must exist");
    assert!(explicit.item.importance > Memory::AUTO_IMPORTANCE);
}

#[tokio::test]
async fn solve_prompt_includes_tool_args_format() {
    /// Test model that captures the prompt system and returns a direct final response.
    struct CaptureModel(std::sync::Mutex<String>);
    #[async_trait::async_trait]
    impl Model for CaptureModel {
        async fn complete(&self, p: &Prompt) -> Result<crate::model::Completion> {
            *self.0.lock().unwrap() = p.system.clone();
            Ok(crate::model::Completion::new("done"))
        }
    }

    let store: Arc<dyn MemoryStore> = Arc::new(InMemoryStore::new());
    let model = Arc::new(CaptureModel(std::sync::Mutex::new(String::new())));
    let agent = Agent::new(Persona::new("Aria", "solver"), store, model.clone());
    let _ = agent.solve(&calc_ctx(), "23+17", 3).await.unwrap();
    let sys = model.0.lock().unwrap().clone();
    assert!(
        sys.contains("args format:"),
        "solve tells model the args format: {sys}"
    );
}

#[tokio::test]
async fn solve_last_step_never_leaks_raw_tool_json() {
    // If the model ignores the instruction on the last step and returns a tool JSON again,
    // raw JSON must not leak to the user as the "final response" — respond with observations.
    let store: Arc<dyn MemoryStore> = Arc::new(InMemoryStore::new());
    let model = Arc::new(SeqModel::new(&[
        r#"{"tool":"calc","args":"3 + 4"}"#,
        r#"{"tool":"calc","args":"7 * 6"}"#, // last step: still JSON
    ]));
    let agent = Agent::new(Persona::new("Aria", "solver"), store, model);
    let out = agent.solve(&calc_ctx(), "(3+4) then?", 2).await.unwrap();
    assert!(
        !out.contains(r#"{"tool""#),
        "raw tool JSON must not leak: {out}"
    );
    assert!(
        out.contains('7'),
        "available observation carried to response: {out}"
    );
}

#[tokio::test]
async fn solve_last_step_forces_final_answer() {
    // Step limit 1: even if the model wants to call a tool, it WON'T run; raw JSON
    // is also not leaked — an explanatory final text is returned.
    let store: Arc<dyn MemoryStore> = Arc::new(InMemoryStore::new());
    let model = Arc::new(SeqModel::new(&[r#"{"tool":"calc","args":"1+1"}"#]));
    let agent = Agent::new(Persona::new("Aria", "solver"), store, model);

    let out = agent.solve(&calc_ctx(), "1+1", 1).await.unwrap();
    assert!(
        !out.contains(r#"{"tool""#),
        "raw tool JSON does not leak: {out}"
    );
    assert!(out.contains("limit"), "explanatory message returned: {out}");
}

#[tokio::test]
async fn respond_stream_yields_chunks_and_remembers_at_end() {
    let store: Arc<dyn MemoryStore> = Arc::new(InMemoryStore::new());
    let agent = agent_with("Aria", store);

    let mut stream = agent.respond_stream("a strange topic").await.unwrap();
    let mut full = String::new();
    while let Some(chunk) = stream.next().await {
        full.push_str(&chunk.unwrap());
    }
    drop(stream);
    assert!(
        full.contains("a strange topic"),
        "stream carried full response"
    );

    // When the stream ended, the exchange was recorded as episodic.
    let mems = agent.recall(&Query::new("strange")).await.unwrap();
    assert!(!mems.is_empty(), "post-stream memory cycle closed");
}

#[tokio::test]
async fn reasoning_fallback_reply_is_truncated_in_memory() {
    // L7: when content is empty, reasoning_content is used — the user sees the full text
    // but raw CoT MUST NOT be written to memory (preventing context pollution
    // + prompt bloat on subsequent recalls). It is stored trimmed.
    struct ReasoningModel;
    #[async_trait::async_trait]
    impl Model for ReasoningModel {
        async fn complete(&self, _p: &Prompt) -> Result<crate::model::Completion> {
            Ok(crate::model::Completion::reasoning_fallback(
                "x".repeat(2000),
            ))
        }
    }
    let store: Arc<dyn MemoryStore> = Arc::new(InMemoryStore::new());
    let agent = Agent::new(
        Persona::new("Aria", "role"),
        store,
        Arc::new(ReasoningModel),
    );
    let reply = agent.respond("question").await.unwrap();
    assert_eq!(reply.len(), 2000, "user sees full text");

    let mems = agent.recall(&Query::new("question")).await.unwrap();
    assert_eq!(mems.len(), 1);
    let crate::memory::MemoryKind::Episodic { body, .. } = &mems[0].item.kind else {
        panic!("expected episodic");
    };
    assert!(
        body.chars().count() <= 600,
        "CoT should be truncated before storing: {}",
        body.chars().count()
    );
}

#[tokio::test]
async fn save_to_writes_atomically_without_tmp_leftover() {
    // M3: persona write must be atomic via tmp+rename — crash/SIGKILL
    // mid-write leaves no corrupt JSON, agent does not silently disappear on restart.
    let dir = std::env::temp_dir().join(format!("lore-atomic-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("agent.json");

    let store: Arc<dyn MemoryStore> = Arc::new(InMemoryStore::new());
    let agent = agent_with("Aria", store.clone());
    agent.save_to(&path).unwrap();

    // No temporary file left, target file is valid and loadable.
    let entries: Vec<_> = std::fs::read_dir(&dir).unwrap().collect();
    assert_eq!(entries.len(), 1, "no tmp file leftover: {entries:?}");
    let loaded = Agent::load_from(&path, store, Arc::new(MockModel::new())).unwrap();
    assert_eq!(loaded.persona.name, "Aria");

    // Overwriting also works with the same guarantee.
    agent.save_to(&path).unwrap();
    let entries: Vec<_> = std::fs::read_dir(&dir).unwrap().collect();
    assert_eq!(entries.len(), 1, "no tmp leftover on second write either");
    std::fs::remove_dir_all(&dir).ok();
}

#[tokio::test]
async fn respond_recalls_morphological_variants() {
    // H2: respond's memory access was keyword-only — a "learning" record
    // was filtered out by a "math" query. Flagship morphological capture
    // must also be evident in the agent's own reasoning loop.
    let store: Arc<dyn MemoryStore> = Arc::new(
        InMemoryStore::new().with_embedder(Arc::new(crate::memory::HashingEmbedder::new())),
    );
    let agent = agent_with("Aria", store);
    agent
        .experience("learning", "user is studying math")
        .await
        .unwrap();

    let reply = agent.respond("math").await.unwrap();
    assert!(
        reply.contains("recalling 1 memories") && reply.contains("learning"),
        "morphological variant should be recalled and enter prompt: {reply}"
    );
}

#[tokio::test]
async fn recall_marks_returned_memories_as_accessed() {
    // Textual query: returned records are counted as "accessed" (decay signal is fed).
    let store: Arc<dyn MemoryStore> = Arc::new(InMemoryStore::new());
    let agent = agent_with("Aria", store.clone());
    agent
        .experience("stainless topic", "access should be marked")
        .await
        .unwrap();

    let hits = agent.recall(&Query::new("stainless")).await.unwrap();
    assert_eq!(hits.len(), 1);
    let mem = store.get(&hits[0].item.id).await.unwrap().unwrap();
    assert_eq!(mem.access_count, 1, "textual recall should mark access");

    // Second recall increments the counter.
    let _ = agent.recall(&Query::new("stainless")).await.unwrap();
    let mem = store.get(&hits[0].item.id).await.unwrap().unwrap();
    assert_eq!(mem.access_count, 2);
}

#[tokio::test]
async fn browse_recall_does_not_touch_memories() {
    // Browse (textless) bulk scans — graph construction, board reading —
    // do not count as access; otherwise every full scan would completely kill decay.
    let store: Arc<dyn MemoryStore> = Arc::new(InMemoryStore::new());
    let agent = agent_with("Aria", store.clone());
    agent.experience("topic", "content").await.unwrap();

    let hits = agent.recall(&Query::new("")).await.unwrap();
    assert_eq!(hits.len(), 1);
    let mem = store.get(&hits[0].item.id).await.unwrap().unwrap();
    assert_eq!(mem.access_count, 0, "browse recall should not touch");
}

#[tokio::test]
async fn freshly_recalled_low_value_memory_survives_decay() {
    // H1 regression: old + low-importance (automatic) record, if accessed via recall,
    // consolidation MUST NOT forget it. Pre-reinforcement behavior:
    // recall was not counting access → record was caught by the 90-day rule and forgotten.
    let store: Arc<dyn MemoryStore> = Arc::new(InMemoryStore::new());
    let agent = agent_with("Aria", store.clone());

    let mut m = Memory::episodic(
        crate::memory::Scope::World, // remember() pulls into own scope
        "old but recalled topic",
        "still works",
    )
    .with_importance(Memory::AUTO_IMPORTANCE);
    let old = chrono::Utc::now() - chrono::Duration::days(120);
    m.created_at = old;
    m.last_access = old;
    agent.remember(m).await.unwrap();

    // Record is genuinely found and used via textual query.
    let hits = agent.recall(&Query::new("recalled")).await.unwrap();
    assert_eq!(hits.len(), 1);

    // Consolidation runs: accessed record should survive.
    let report = store.consolidate().await.unwrap();
    assert_eq!(
        report.forgotten, 0,
        "accessed record should not be forgotten"
    );
    let still = agent.recall(&Query::new("recalled")).await.unwrap();
    assert_eq!(still.len(), 1, "record still accessible");
}

#[tokio::test]
async fn two_agents_share_store_but_have_separate_memories() {
    let store: Arc<dyn MemoryStore> = Arc::new(InMemoryStore::new());
    let aria = agent_with("Aria", store.clone());
    let kai = agent_with("Kai", store.clone());

    aria.remember(Memory::semantic(
        Scope::World, // scope is set to own scope inside respond
        "Aria's personal note alpha",
        SemanticCat::Fact,
    ))
    .await
    .unwrap();

    // Kai should not see Aria's personal record.
    assert_eq!(kai.recall(&Query::new("alpha")).await.unwrap().len(), 0);
    // Aria should see her own record.
    assert_eq!(aria.recall(&Query::new("alpha")).await.unwrap().len(), 1);
}

#[tokio::test]
async fn empty_memory_agent_acknowledges() {
    let store: Arc<dyn MemoryStore> = Arc::new(InMemoryStore::new());
    let agent = agent_with("Aria", store);
    let reply = agent.respond("hi").await.unwrap();
    assert!(reply.contains("memory empty"));
}

#[tokio::test]
async fn act_uses_tool_when_routed() {
    use crate::tool::{CalcTool, KeywordRouter, ToolRegistry, ToolRouter};
    let store: Arc<dyn MemoryStore> = Arc::new(InMemoryStore::new());
    let mut reg = ToolRegistry::new();
    reg.register(Arc::new(CalcTool::new()));
    let router: Arc<dyn ToolRouter> = Arc::new(KeywordRouter::new().on("calculate", "calc"));
    let agent = Agent::new(
        Persona::new("Aria", "role"),
        store,
        Arc::new(MockModel::new()),
    )
    .with_tools(reg, router);

    let out = agent.act("calculate 12 * 3").await.unwrap();
    assert_eq!(out, "36");
    // Did it remember the tool usage?
    let mem = agent.recall(&Query::new("calc")).await.unwrap();
    assert!(!mem.is_empty());
}

#[tokio::test]
async fn act_falls_back_to_respond_without_tool() {
    let store: Arc<dyn MemoryStore> = Arc::new(InMemoryStore::new());
    let agent = agent_with("Aria", store);
    let out = agent.act("hi").await.unwrap();
    assert!(out.contains("memory empty"));
}

#[tokio::test]
async fn identity_survives_restart() {
    use crate::id::AgentId;
    use crate::memory::SqliteStore;

    let dir = std::env::temp_dir();
    let stamp = AgentId::new();
    let persona_path = dir.join(format!("lore-agent-{stamp}.json"));
    let db_path = dir.join(format!("lore-agent-{stamp}.db"));
    let persona_path = persona_path.to_str().unwrap().to_string();
    let db_path = db_path.to_str().unwrap().to_string();
    let model: Arc<dyn Model> = Arc::new(MockModel::new());

    // First life: save identity, experience a memory.
    let saved_id = {
        let store: Arc<dyn MemoryStore> = Arc::new(SqliteStore::open(&db_path).unwrap());
        let agent = Agent::new(
            Persona::new("Aria", "researcher").with_trait("curious"),
            store.clone(),
            model.clone(),
        );
        agent.save_to(&persona_path).unwrap();
        agent
            .experience("important event", "should be recalled after restart")
            .await
            .unwrap();
        agent.id.clone()
    };

    // Rebirth: persona file + same DB → both character and memories restored.
    {
        let store: Arc<dyn MemoryStore> = Arc::new(SqliteStore::open(&db_path).unwrap());
        let agent = Agent::load_from(&persona_path, store, model.clone()).unwrap();
        assert_eq!(agent.id, saved_id, "same AgentId");
        assert_eq!(agent.persona.name, "Aria");
        assert!(agent.persona.traits.contains(&"curious".to_string()));
        let mem = agent.recall(&Query::new("important")).await.unwrap();
        assert_eq!(mem.len(), 1, "same scope → memories restored");
    }

    let _ = std::fs::remove_file(&persona_path);
    let _ = std::fs::remove_file(&db_path);
}

#[tokio::test]
async fn recall_hyde_runs_and_searches() {
    use crate::memory::HashingEmbedder;
    let store: Arc<dyn MemoryStore> =
        Arc::new(InMemoryStore::new().with_embedder(Arc::new(HashingEmbedder::new())));
    let agent = agent_with("Aria", store);
    agent
        .remember(Memory::semantic(
            Scope::World,
            "thoughts on math",
            SemanticCat::Preference,
        ))
        .await
        .unwrap();

    // HyDE: MockModel generates a hypothesis (includes the input), embed_text is computed from it.
    let res = agent.recall_hyde("math").await.unwrap();
    assert!(!res.is_empty(), "HyDE should return at least one record");
}
