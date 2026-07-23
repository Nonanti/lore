//! Tools: how agents interact with the world.
//!
//! `Tool` abstracts a capability (calculation, search, time...). `ToolRegistry`
//! holds them by name. `ToolRouter` decides which tool to call for a given
//! input — starts with the native `KeywordRouter` (deterministic); later an
//! LLM router can plug into the same trait. `Agent::act` drives this loop.

pub mod builtin;
pub mod fs_write;
pub mod shell;

pub use builtin::{CalcTool, FileReadTool, TimeTool, WebFetchTool};
pub use fs_write::{FileEditTool, FileWriteTool};
pub use shell::ShellTool;

use crate::error::Result;
use crate::model::{Model, Prompt};
use async_trait::async_trait;
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

/// Invocable capability.
#[async_trait]
pub trait Tool: Send + Sync {
    /// Unique tool name.
    fn name(&self) -> &str;
    /// Brief description of what it does (may be presented to the model).
    fn description(&self) -> &str;
    /// Expected format of arguments — presented to the model in the catalog.
    /// Returns empty string if no format line should appear in catalog.
    /// Without this hint, models invent their own format (e.g. qwen3's
    /// comma-separated args) — see §5.1.
    fn args_hint(&self) -> &str {
        ""
    }
    /// Runs the tool with the given arguments.
    async fn run(&self, args: &str) -> Result<String>;
}

/// Tool catalog presented to the model: sorted by name (deterministic),
/// description + args format. `LlmRouter` and `Agent::solve` share the same
/// catalog — single source of truth, format contract cannot drift in two
/// places.
pub fn catalog(tools: &ToolRegistry) -> String {
    let mut names = tools.names();
    names.sort();
    names
        .iter()
        .filter_map(|n| tools.get(n).map(|t| (n, t)))
        .map(|(n, t)| {
            let hint = t.args_hint();
            if hint.is_empty() {
                format!("- {n}: {}", t.description())
            } else {
                format!("- {n}: {} — args format: {hint}", t.description())
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// A tool call (which tool, which arguments).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolCall {
    /// Tool name.
    pub tool: String,
    /// Arguments (raw text).
    pub args: String,
}

/// Registry holding tools by name.
#[derive(Default, Clone)]
pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn Tool>>,
}

impl ToolRegistry {
    /// Empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a tool (by its own name).
    pub fn register(&mut self, tool: Arc<dyn Tool>) -> &mut Self {
        self.tools.insert(tool.name().to_string(), tool);
        self
    }

    /// Accesses a tool by name.
    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools.get(name).cloned()
    }

    /// Registered tool names.
    pub fn names(&self) -> Vec<String> {
        self.tools.keys().cloned().collect()
    }

    /// Number of tools.
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }
}

/// Strategy for deciding which tool to call for a given input.
#[async_trait]
pub trait ToolRouter: Send + Sync {
    /// Proposes a tool call matching the input (`None` if none).
    async fn route(&self, input: &str, tools: &ToolRegistry) -> Option<ToolCall>;
}

/// Native, deterministic router: keyword → tool mapping.
/// `BTreeMap`: iteration order is fixed — when multiple keywords match,
/// which tool is selected is deterministic.
#[derive(Default, Clone)]
pub struct KeywordRouter {
    map: BTreeMap<String, String>, // keyword (lower) -> tool name
}

impl KeywordRouter {
    /// Empty router.
    pub fn new() -> Self {
        Self::default()
    }

    /// Binds a trigger keyword to a tool (builder).
    pub fn on(mut self, keyword: impl Into<String>, tool: impl Into<String>) -> Self {
        self.map.insert(keyword.into().to_lowercase(), tool.into());
        self
    }
}

#[async_trait]
impl ToolRouter for KeywordRouter {
    async fn route(&self, input: &str, tools: &ToolRegistry) -> Option<ToolCall> {
        let low = input.to_lowercase();
        for (kw, tool) in &self.map {
            if low.contains(kw.as_str()) && tools.get(tool).is_some() {
                return Some(ToolCall {
                    tool: tool.clone(),
                    args: input.to_string(),
                });
            }
        }
        None
    }
}

/// LLM-based router: presents the tool catalog to the model, model selects
/// a tool via JSON. Sits behind the same `ToolRouter` trait as the native
/// `KeywordRouter`.
pub struct LlmRouter {
    model: Arc<dyn Model>,
}

impl LlmRouter {
    /// New router with the given model.
    pub fn new(model: Arc<dyn Model>) -> Self {
        Self { model }
    }
}

#[async_trait]
impl ToolRouter for LlmRouter {
    async fn route(&self, input: &str, tools: &ToolRegistry) -> Option<ToolCall> {
        if tools.is_empty() {
            return None;
        }
        let catalog = catalog(tools);
        let prompt = Prompt {
            system: format!(
                "You are a tool router. Available tools:\n{catalog}\n\
                 Select the best tool for the input and return ONLY this JSON: \
                 {{\"tool\":\"<name or null>\",\"args\":\"<argument — in tool args format>\"}}"
            ),
            user: input.to_string(),
            ..Default::default()
        };
        let completion = self.model.complete(&prompt).await.ok()?;
        let call = parse_tool_call(&completion.text)?;
        // Model may have hallucinated; verify the tool actually exists.
        if tools.get(&call.tool).is_some() {
            Some(call)
        } else {
            None
        }
    }
}

/// Extracts the first JSON tool-call object from free-form text (between
/// prose/code fences) and converts it to a `{"tool":..,"args":..}` call.
/// Returns `None` if tool is null/empty.
///
/// Trailing content after the first complete object is ignored — models
/// sometimes emit several calls in one reply (`{…}\n{…}`); only the first
/// is taken, the loop re-prompts for the rest. Each `{` is tried as a
/// potential start, so prose containing stray braces before the real JSON
/// does not break parsing.
pub fn parse_tool_call(text: &str) -> Option<ToolCall> {
    for (start, _) in text.match_indices('{') {
        let mut stream =
            serde_json::Deserializer::from_str(&text[start..]).into_iter::<serde_json::Value>();
        let Some(Ok(v)) = stream.next() else {
            continue;
        };
        let Some(tool) = v.get("tool").and_then(|t| t.as_str()) else {
            continue;
        };
        let tool = tool.trim().to_string();
        if tool.is_empty() || tool == "null" {
            return None;
        }
        let args = v
            .get("args")
            .and_then(|a| a.as_str())
            .unwrap_or("")
            .to_string();
        return Some(ToolCall { tool, args });
    }
    None
}

/// Tool context bound to an agent: registry + router.
pub struct ToolContext {
    /// Tool registry.
    pub registry: ToolRegistry,
    /// Routing strategy.
    pub router: Arc<dyn ToolRouter>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn keyword_router_matches_and_misses() {
        let mut reg = ToolRegistry::new();
        reg.register(Arc::new(CalcTool::new()));
        let router = KeywordRouter::new().on("calculate", "calc");

        let hit = router.route("calculate 2 + 2", &reg).await;
        assert_eq!(hit.unwrap().tool, "calc");

        let miss = router.route("hello", &reg).await;
        assert!(miss.is_none());
    }

    #[tokio::test]
    async fn registry_tracks_tools() {
        let mut reg = ToolRegistry::new();
        assert!(reg.is_empty());
        reg.register(Arc::new(CalcTool::new()));
        assert_eq!(reg.len(), 1);
        assert!(reg.get("calc").is_some());
    }

    #[test]
    fn catalog_includes_args_hint() {
        let mut reg = ToolRegistry::new();
        reg.register(Arc::new(CalcTool::new()));
        let cat = catalog(&reg);
        assert!(cat.contains("- calc:"), "tool line present: {cat}");
        assert!(cat.contains("args format:"), "format hint present: {cat}");
        assert!(cat.contains("23 + 17"), "example usage present: {cat}");
    }

    #[tokio::test]
    async fn llm_router_prompt_carries_args_format() {
        /// Test model that captures the prompt's system and selects no tool.
        struct CaptureModel(std::sync::Mutex<String>);
        #[async_trait]
        impl crate::model::Model for CaptureModel {
            async fn complete(&self, p: &Prompt) -> Result<crate::model::Completion> {
                *self.0.lock().unwrap() = p.system.clone();
                Ok(crate::model::Completion::new(r#"{"tool":null}"#))
            }
        }

        let mut reg = ToolRegistry::new();
        reg.register(Arc::new(CalcTool::new()));
        let model = Arc::new(CaptureModel(std::sync::Mutex::new(String::new())));
        let router = LlmRouter::new(model.clone());
        let _ = router.route("add 23 and 17", &reg).await;
        let sys = model.0.lock().unwrap().clone();
        assert!(
            sys.contains("args format:"),
            "router tells model the args format: {sys}"
        );
    }

    #[test]
    fn parse_tool_call_variants() {
        // Plain JSON.
        let c = parse_tool_call(r#"{"tool":"calc","args":"12*3"}"#).unwrap();
        assert_eq!(c.tool, "calc");
        assert_eq!(c.args, "12*3");

        // Extraction from between prose and code fences.
        let c2 = parse_tool_call(
            "Here is my choice:\n```json\n{\"tool\":\"calc\",\"args\":\"2+2\"}\n```",
        )
        .unwrap();
        assert_eq!(c2.tool, "calc");

        // null / empty / no JSON → None.
        assert!(parse_tool_call(r#"{"tool":null}"#).is_none());
        assert!(parse_tool_call(r#"{"tool":""}"#).is_none());
        assert!(parse_tool_call("no json here").is_none());
    }

    #[test]
    fn parse_tool_call_takes_first_of_many() {
        // Models sometimes emit several calls in one reply — only the
        // first is taken (the solve loop re-prompts for the rest).
        let two = "{\"tool\":\"write\",\"args\":\"a\"}\n{\"tool\":\"shell\",\"args\":\"b\"}";
        let c = parse_tool_call(two).unwrap();
        assert_eq!(c.tool, "write");
        assert_eq!(c.args, "a");

        // Stray braces in prose before the real JSON do not break parsing.
        let noisy = "use {tool} syntax: {\"tool\":\"calc\",\"args\":\"1+1\"}";
        let c2 = parse_tool_call(noisy).unwrap();
        assert_eq!(c2.tool, "calc");
    }

    /// Test model that returns a fixed JSON tool call.
    struct StubModel(String);
    #[async_trait]
    impl crate::model::Model for StubModel {
        async fn complete(&self, _p: &Prompt) -> Result<crate::model::Completion> {
            Ok(crate::model::Completion::new(self.0.clone()))
        }
    }

    #[tokio::test]
    async fn llm_router_selects_and_validates() {
        let mut reg = ToolRegistry::new();
        reg.register(Arc::new(CalcTool::new()));

        // Model selects a valid tool.
        let ok = LlmRouter::new(Arc::new(StubModel(
            r#"{"tool":"calc","args":"7*6"}"#.to_string(),
        )));
        let call = ok.route("what is the product", &reg).await.unwrap();
        assert_eq!(call.tool, "calc");
        assert_eq!(call.args, "7*6");

        // If the model hallucinates a non-existent tool, it is rejected.
        let bad = LlmRouter::new(Arc::new(StubModel(
            r#"{"tool":"nonexistent","args":"x"}"#.to_string(),
        )));
        assert!(bad.route("hello", &reg).await.is_none());
    }
}
