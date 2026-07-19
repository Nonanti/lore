//! End-to-end integration: REAL binary + real file DB + real HTTP.
//!
//! Verifies layers that unit tests cannot see: CLI arg flow, process lifetime,
//! post-restart persistence, authenticated HTTP service. Binary path comes from
//! Cargo's `CARGO_BIN_EXE_lore` env — no extra dependency.

use std::process::{Child, Command, Stdio};
use std::time::Duration;

const BIN: &str = env!("CARGO_BIN_EXE_lore");

/// Isolated test data directory (deleted on drop).
struct TmpData(std::path::PathBuf);
impl TmpData {
    fn new() -> Self {
        let p = std::env::temp_dir().join(format!("lore-e2e-{}", std::process::id()));
        let p = p.join(format!("{:x}", rand_suffix()));
        std::fs::create_dir_all(&p).unwrap();
        Self(p)
    }
    fn path(&self) -> &str {
        self.0.to_str().unwrap()
    }
}
impl Drop for TmpData {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn rand_suffix() -> u128 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos()
}

/// Runs a CLI command, returns stdout (panics with stderr on failure).
fn cli(data: &str, args: &[&str]) -> String {
    let out = Command::new(BIN)
        .env("LORE_DATA", data)
        .env("LORE_LOG", "error") // suppress log noise in test output
        .args(args)
        .output()
        .expect("binary should run");
    assert!(
        out.status.success(),
        "command failed {args:?} (data={data}): {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).into_owned()
}

#[test]
fn cli_full_memory_lifecycle_with_restart() {
    // new-agent → remember → recall → (process dies) → new process recall →
    // export/import round-trip. Each command is a SEPARATE process: persistence
    // truly relies on disk, not same-process cache.
    let data = TmpData::new();
    let d = data.path();

    let out = cli(d, &["new-agent", "--name", "Aria", "--role", "tester"]);
    let id = out
        .split_whitespace()
        .find(|t| t.len() == 26 && t.chars().all(|c| c.is_ascii_alphanumeric()))
        .expect("expected ULID")
        .to_string();
    cli(
        d,
        &[
            "remember",
            &id,
            "--title",
            "Learned Rust",
            "--body",
            "ownership and borrow checker",
        ],
    );

    // Separate process: keyword recall.
    let hits = cli(d, &["recall", &id, "rust"]);
    assert!(hits.contains("Learned Rust"), "keyword recall: {hits}");

    // Separate process: short query semantic (token-level fallback, not FTS).
    let sem = cli(d, &["recall", &id, "learning", "--semantic"]);
    assert!(sem.contains("Learned Rust"), "semantic recall: {sem}");

    // Export → import into new data directory → re-export comparison.
    // (Import carries MEMORIES, NOT agent identities — deliberate product decision;
    // verification is done via dump comparison instead of agent-based recall.)
    let dump = data.0.join("dump.json");
    cli(d, &["export", "--out", dump.to_str().unwrap()]);
    let data2 = TmpData::new();
    cli(data2.path(), &["import", dump.to_str().unwrap()]);
    let dump2 = data.0.join("dump2.json");
    cli(data2.path(), &["export", "--out", dump2.to_str().unwrap()]);
    let a: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&dump).unwrap()).unwrap();
    let b: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&dump2).unwrap()).unwrap();
    assert_eq!(a, b, "imported dump preserved exactly on re-export");
    assert!(
        a.as_array().is_some_and(|v| !v.is_empty()),
        "dump is not empty"
    );
}

/// Find a free port (bind + drop — race window is acceptable in tests).
fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

/// Guard that kills the child process and waits on drop.
struct KillOnDrop(Child);
impl Drop for KillOnDrop {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

async fn wait_ready(base: &str) {
    for _ in 0..100 {
        if let Ok(r) = reqwest::get(format!("{base}/ready")).await {
            if r.status() == 200 {
                return;
            }
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    panic!("service {base} not ready");
}

#[tokio::test]
async fn serve_e2e_auth_persistence_and_kill_restart() {
    // Real `lore serve` process: auth required, create agent, ask, SIGKILL,
    // restart → agent AND memory come back from disk.
    let data = TmpData::new();
    let port = free_port();
    let base = format!("http://127.0.0.1:{port}");
    let spawn = |data: &str| {
        KillOnDrop(
            Command::new(BIN)
                .env("LORE_DATA", data)
                .env("LORE_API_KEY", "secret-key")
                .env("LORE_LOG", "error")
                .args(["serve", "--addr", &format!("127.0.0.1:{port}")])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
                .expect("serve should start"),
        )
    };

    let child = spawn(data.path());
    wait_ready(&base).await;
    let http = reqwest::Client::new();

    // Auth required: 401 without key.
    let unauth = http
        .post(format!("{base}/agents"))
        .json(&serde_json::json!({"name":"Aria","role":"t"}))
        .send()
        .await
        .unwrap();
    assert_eq!(unauth.status(), 401, "request without key rejected");

    // Create + ask with key.
    let created: serde_json::Value = http
        .post(format!("{base}/agents"))
        .header("x-api-key", "secret-key")
        .json(&serde_json::json!({"name":"Aria","role":"t"}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let id = created["id"].as_str().unwrap().to_string();
    let ask: serde_json::Value = http
        .post(format!("{base}/agents/{id}/ask"))
        .header("x-api-key", "secret-key")
        .json(&serde_json::json!({"message":"permanent note: blue door"}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(!ask["reply"].as_str().unwrap().is_empty());

    // Hard death + rebirth (same port, same data).
    drop(child); // SIGKILL
    let _child2 = spawn(data.path());
    wait_ready(&base).await;

    // Agent list came back from disk.
    let agents: serde_json::Value = http
        .get(format!("{base}/agents"))
        .header("x-api-key", "secret-key")
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(
        agents
            .as_array()
            .unwrap()
            .iter()
            .any(|a| a["id"] == id.as_str()),
        "agent persists after restart: {agents}"
    );
    // Memory (SQLite) is accessible after restart — conversation trace read from WAL.
    let recall: serde_json::Value = http
        .get(format!("{base}/agents/{id}/recall?q=blue"))
        .header("x-api-key", "secret-key")
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(
        !recall.as_array().unwrap().is_empty(),
        "memory survives restart: {recall}"
    );
}
