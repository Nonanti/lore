//! Phase 1 "hands" demo: policy-gated shell/write/edit tools.
//!
//! Part 1 exercises the tools directly (deterministic, offline):
//! auto-allowed shell, sandboxed write/edit, an approval prompt, a
//! deny-list refusal, and a sandbox-escape rejection.
//!
//! Part 2 (optional) drives a real agent through the same tools when an
//! OpenAI-compatible endpoint is configured:
//!
//! ```bash
//! cargo run --example hands_demo                     # part 1 only
//! LORE_LLM_BASE=http://localhost:11434/v1 LORE_LLM_MODEL=qwen3:8b \
//!   cargo run --example hands_demo                   # part 1 + agent
//! ```

use lore::{
    Action, Agent, CliApprover, FileEditTool, FileWriteTool, Gate, InMemoryStore, KeywordRouter,
    MemoryStore, OpenAiModel, Persona, Policy, ShellTool, Tool, ToolContext, ToolRegistry,
};
use std::path::PathBuf;
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Workspace: everything the tools touch stays under this root.
    let workspace = PathBuf::from("lore-data/demo-workspace");
    std::fs::create_dir_all(&workspace)?;
    let workspace = std::fs::canonicalize(&workspace)?;
    println!("workspace: {}\n", workspace.display());

    // Personal-use defaults + interactive approval for anything unlisted.
    let policy = Policy::default_for(workspace.clone());
    let gate = Arc::new(Gate::new(policy, Arc::new(CliApprover)));

    let shell = ShellTool::new(gate.clone(), workspace.clone());
    let write = FileWriteTool::new(gate.clone(), workspace.clone());
    let edit = FileEditTool::new(gate.clone(), workspace.clone());

    // ── Part 1: the gate in action ─────────────────────────────────────
    println!("1) auto-allowed shell (echo is on the auto_allow list):");
    println!("{}\n", shell.run("echo hello from Lore").await?);

    println!("2) sandboxed write (inside workspace -> allowed):");
    println!(
        "{}\n",
        write
            .run(r#"{"path":"notes.txt","content":"ilk satir\nikinci satir\n"}"#)
            .await?
    );

    println!("3) sandboxed edit (single exact match required):");
    println!(
        "{}\n",
        edit.run(r#"{"path":"notes.txt","old":"ikinci satir","new":"duzenlenmis satir"}"#)
            .await?
    );

    println!("4) unlisted command -> approval prompt (answer y/N):");
    match shell.run("touch approved.txt").await {
        Ok(out) => println!("approved and ran: {out}\n"),
        Err(e) => println!("refused: {e}\n"),
    }

    println!("5) deny-listed command (sudo) -> refused without asking:");
    match shell.run("sudo whoami").await {
        Ok(out) => println!("unexpected: {out}\n"),
        Err(e) => println!("refused: {e}\n"),
    }

    println!("6) sandbox escape -> rejected:");
    match write
        .run(r#"{"path":"../escape.txt","content":"nope"}"#)
        .await
    {
        Ok(out) => println!("unexpected: {out}\n"),
        Err(e) => println!("rejected: {e}\n"),
    }

    // ── Part 2: a real agent using the same hands ──────────────────────
    let (Ok(base), Ok(model)) = (
        std::env::var("LORE_LLM_BASE"),
        std::env::var("LORE_LLM_MODEL"),
    ) else {
        println!("(set LORE_LLM_BASE + LORE_LLM_MODEL to run the agent part)");
        return Ok(());
    };

    println!("── agent part (model: {model}) ──");
    let store: Arc<dyn MemoryStore> = Arc::new(InMemoryStore::new());
    let kaya = Agent::new(
        Persona::new("Kaya", "developer").with_traits(["pragmatic", "careful"]),
        store,
        Arc::new(OpenAiModel::new(base, model)),
    );

    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(ShellTool::new(gate.clone(), workspace.clone())));
    registry.register(Arc::new(FileWriteTool::new(
        gate.clone(),
        workspace.clone(),
    )));
    registry.register(Arc::new(FileEditTool::new(gate, workspace)));
    let ctx = ToolContext {
        registry,
        router: Arc::new(KeywordRouter::new()),
    };

    let task = "Create a file named gorev.txt containing the single line \
                'merhaba patron', then run `cat gorev.txt` to verify, and \
                report the result.";
    println!("task: {task}\n");
    let answer = kaya.solve(&ctx, task, 6).await?;
    println!("Kaya: {answer}");

    // Check the gate held: was Action::Write actually evaluated? Show proof.
    let proof = std::fs::read_to_string("lore-data/demo-workspace/gorev.txt");
    println!("\non-disk proof: {proof:?}");
    Ok(())
}

// Silence the unused-import lint when Part 2 is skipped at runtime: Action is
// referenced here so the example always compiles with the full public API.
#[allow(dead_code)]
fn _uses_public_api(a: Action) -> Action {
    a
}
