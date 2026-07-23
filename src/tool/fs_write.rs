//! File write and edit tools: sandboxed, policy-gated file operations.
//!
//! [`FileWriteTool`] creates/overwrites files atomically (tmp + rename).
//! [`FileEditTool`] replaces an exact substring — zero or multiple matches
//! produce descriptive errors. Both enforce relative-path sandboxing
//! (no absolute paths, no `..`, no symlink escapes) and pass through the
//! [`Gate`] with [`Action::Write`].

use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;

use crate::error::{LoreError, Result};
use crate::policy::approval::Gate;
use crate::policy::Action;
use crate::tool::Tool;

// ── Shared sandbox ──────────────────────────────────────────────────────

/// Validates that a relative path stays inside the workspace root.
///
/// Rejects:
/// - Absolute paths
/// - `..` components (traversal)
/// - Symlink escapes (canonicalization check)
///
/// Returns the canonicalized full path on success.
fn sandbox_path(rel: &str, root: &Path) -> Result<PathBuf> {
    if rel.is_empty() {
        return Err(LoreError::InvalidInput("file path required".into()));
    }
    let p = Path::new(rel);
    if p.is_absolute() {
        return Err(LoreError::InvalidInput(
            "only relative paths within the workspace are allowed".into(),
        ));
    }
    if rel.split(['/', '\\']).any(|s| s == "..") {
        return Err(LoreError::InvalidInput(
            "path traversal (..) is not allowed".into(),
        ));
    }
    let full = root.join(p);
    // For writes, the file may not exist yet — canonicalize root and
    // verify the joined path doesn't escape via symlinks in existing
    // ancestors.
    let root_canon = std::fs::canonicalize(root).map_err(|e| LoreError::Storage(e.to_string()))?;
    // Walk ancestors of `full` to find nearest existing, canonicalize it,
    // append the rest, and verify containment.
    let canon = canonicalize_nearest_existing(&full);
    let canon = canon
        .map_err(|e| LoreError::Storage(format!("cannot resolve path {}: {e}", full.display())))?;
    if !canon.starts_with(&root_canon) {
        return Err(LoreError::InvalidInput(
            "path escapes workspace root (symlink or traversal)".into(),
        ));
    }
    Ok(canon)
}

/// Canonicalize the nearest existing ancestor, then append the
/// non-existing suffix. Returns `Err` if no ancestor can be resolved.
fn canonicalize_nearest_existing(path: &Path) -> std::result::Result<PathBuf, String> {
    for i in 0..path.components().count() {
        let ancestor: PathBuf = path
            .components()
            .take(path.components().count() - i)
            .collect();
        if let Ok(canon) = std::fs::canonicalize(&ancestor) {
            let suffix: PathBuf = path
                .components()
                .skip(path.components().count() - i)
                .collect();
            if suffix.components().count() == 0 {
                return Ok(canon);
            }
            return Ok(canon.join(suffix));
        }
    }
    Err("no existing ancestor found".into())
}

// ── JSON args ───────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct WriteArgs {
    path: String,
    content: String,
}

#[derive(Deserialize)]
struct EditArgs {
    path: String,
    old: String,
    new: String,
}

// ── FileWriteTool ───────────────────────────────────────────────────────

/// Creates or overwrites a file atomically (tmp file + rename).
/// Creates parent directories as needed.
///
/// Args: JSON `{"path":"rel/path","content":"..."}`
#[derive(Clone, Debug)]
pub struct FileWriteTool {
    gate: Arc<Gate>,
    root: PathBuf,
}

impl FileWriteTool {
    /// New write tool with the given gate and workspace root.
    pub fn new(gate: Arc<Gate>, root: PathBuf) -> Self {
        let _ = std::fs::create_dir_all(&root);
        Self { gate, root }
    }
}

#[async_trait]
impl Tool for FileWriteTool {
    fn name(&self) -> &str {
        "write"
    }

    fn description(&self) -> &str {
        "Creates or overwrites a file in the workspace (atomic write)"
    }

    fn args_hint(&self) -> &str {
        r#"JSON {"path":"relative/path","content":"file content"}"#
    }

    async fn run(&self, args: &str) -> Result<String> {
        let wa: WriteArgs = serde_json::from_str(args).map_err(|e| {
            LoreError::InvalidInput(format!(
                "bad JSON args for write tool: {e}\nexpected: {{\"path\":\"...\",\"content\":\"...\"}}"
            ))
        })?;

        let full = sandbox_path(&wa.path, &self.root)?;

        // Policy gate check.
        self.gate
            .check(&Action::Write { path: full.clone() })
            .await?;

        // Create parent dirs.
        if let Some(parent) = full.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| LoreError::Storage(format!("cannot create parent dirs: {e}")))?;
        }

        // Atomic write: tmp file + rename.
        let tmp_name = format!(".lore-write-tmp-{}", ulid::Ulid::new());
        let tmp_path = full.parent().unwrap_or(Path::new(".")).join(&tmp_name);
        std::fs::write(&tmp_path, &wa.content)
            .map_err(|e| LoreError::Storage(format!("write failed: {e}")))?;
        std::fs::rename(&tmp_path, &full)
            .map_err(|e| LoreError::Storage(format!("atomic rename failed: {e}")))?;

        Ok(format!("wrote {} bytes to {}", wa.content.len(), wa.path))
    }
}

// ── FileEditTool ────────────────────────────────────────────────────────

/// Edits a file by replacing an exact substring.
///
/// `old` must match **exactly once**; zero or >1 matches produce a
/// descriptive error with the match count.
///
/// Args: JSON `{"path":"rel/path","old":"exact text","new":"replacement"}`
#[derive(Clone, Debug)]
pub struct FileEditTool {
    gate: Arc<Gate>,
    root: PathBuf,
}

impl FileEditTool {
    /// New edit tool with the given gate and workspace root.
    pub fn new(gate: Arc<Gate>, root: PathBuf) -> Self {
        let _ = std::fs::create_dir_all(&root);
        Self { gate, root }
    }
}

#[async_trait]
impl Tool for FileEditTool {
    fn name(&self) -> &str {
        "edit"
    }

    fn description(&self) -> &str {
        "Replaces an exact substring in a file (single match required)"
    }

    fn args_hint(&self) -> &str {
        r#"JSON {"path":"relative/path","old":"exact text","new":"replacement"}"#
    }

    async fn run(&self, args: &str) -> Result<String> {
        let ea: EditArgs = serde_json::from_str(args).map_err(|e| {
            LoreError::InvalidInput(format!(
                "bad JSON args for edit tool: {e}\nexpected: {{\"path\":\"...\",\"old\":\"...\",\"new\":\"...\"}}"
            ))
        })?;

        let full = sandbox_path(&ea.path, &self.root)?;

        // Policy gate check.
        self.gate
            .check(&Action::Write { path: full.clone() })
            .await?;

        // Read existing file.
        let content = std::fs::read_to_string(&full)
            .map_err(|e| LoreError::NotFound(format!("cannot read {}: {e}", ea.path)))?;

        // Count matches.
        let count = content.matches(&ea.old).count();
        if count == 0 {
            return Err(LoreError::InvalidInput(format!(
                "old text not found in {} (0 matches)",
                ea.path
            )));
        }
        if count > 1 {
            return Err(LoreError::InvalidInput(format!(
                "old text matches {} times in {} — must match exactly once",
                count, ea.path
            )));
        }

        // Replace exactly one occurrence.
        let new_content = content.replacen(&ea.old, &ea.new, 1);

        // Atomic write (same as FileWriteTool).
        let tmp_name = format!(".lore-edit-tmp-{}", ulid::Ulid::new());
        let tmp_path = full.parent().unwrap_or(Path::new(".")).join(&tmp_name);
        std::fs::write(&tmp_path, &new_content)
            .map_err(|e| LoreError::Storage(format!("write failed: {e}")))?;
        std::fs::rename(&tmp_path, &full)
            .map_err(|e| LoreError::Storage(format!("atomic rename failed: {e}")))?;

        Ok(format!(
            "edited {}: replaced {} bytes with {} bytes",
            ea.path,
            ea.old.len(),
            ea.new.len()
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::approval::{AllowAll, DenyAll};
    use crate::policy::{DefaultExec, Policy};
    use std::path::PathBuf;
    use std::sync::Arc;

    fn allow_gate(root: PathBuf) -> Arc<Gate> {
        let p = Policy {
            roots: vec![root.clone()],
            auto_allow: vec!["echo".into()],
            deny: vec!["sudo".into()],
            default_exec: DefaultExec::Allow,
            ask_on_write: false,
        };
        Arc::new(Gate::new(p, Arc::new(AllowAll)))
    }

    fn deny_gate(root: PathBuf) -> Arc<Gate> {
        let p = Policy {
            roots: vec![root.clone()],
            auto_allow: vec!["echo".into()],
            deny: vec!["sudo".into()],
            default_exec: DefaultExec::Ask,
            ask_on_write: true,
        };
        Arc::new(Gate::new(p, Arc::new(DenyAll)))
    }

    // ── FileWriteTool ───────────────────────────────────────────────────

    #[tokio::test]
    async fn write_and_read_back() {
        let dir = std::env::temp_dir().join("lore-write-test");
        std::fs::create_dir_all(&dir).unwrap();
        let gate = allow_gate(dir.clone());
        let tool = FileWriteTool::new(gate, dir.clone());
        let out = tool
            .run(r#"{"path":"hello.txt","content":"hello world"}"#)
            .await
            .unwrap();
        assert!(out.contains("hello.txt"), "output: {out}");
        let read = std::fs::read_to_string(dir.join("hello.txt")).unwrap();
        assert_eq!(read, "hello world");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn write_overwrite_existing() {
        let dir = std::env::temp_dir().join("lore-overwrite-test");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("existing.txt"), "old content").unwrap();
        let gate = allow_gate(dir.clone());
        let tool = FileWriteTool::new(gate, dir.clone());
        let out = tool
            .run(r#"{"path":"existing.txt","content":"new content"}"#)
            .await
            .unwrap();
        assert!(out.contains("existing.txt"));
        let read = std::fs::read_to_string(dir.join("existing.txt")).unwrap();
        assert_eq!(read, "new content");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn write_creates_parent_dirs() {
        let dir = std::env::temp_dir().join("lore-parent-test");
        std::fs::create_dir_all(&dir).unwrap();
        let gate = allow_gate(dir.clone());
        let tool = FileWriteTool::new(gate, dir.clone());
        let out = tool
            .run(r#"{"path":"deep/nested/dir/file.txt","content":"nested"}"#)
            .await
            .unwrap();
        assert!(out.contains("file.txt"));
        let read = std::fs::read_to_string(dir.join("deep/nested/dir/file.txt")).unwrap();
        assert_eq!(read, "nested");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn write_escape_rejected() {
        let dir = std::env::temp_dir().join("lore-escape-test");
        std::fs::create_dir_all(&dir).unwrap();
        let gate = allow_gate(dir.clone());
        let tool = FileWriteTool::new(gate, dir.clone());
        // Absolute path rejected.
        assert!(
            tool.run(r#"{"path":"/etc/passwd","content":"evil"}"#)
                .await
                .is_err(),
            "absolute path rejected"
        );
        // Traversal rejected.
        assert!(
            tool.run(r#"{"path":"../etc/passwd","content":"evil"}"#)
                .await
                .is_err(),
            "traversal rejected"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn write_bad_json_args() {
        let dir = std::env::temp_dir().join("lore-badjson-test");
        std::fs::create_dir_all(&dir).unwrap();
        let gate = allow_gate(dir.clone());
        let tool = FileWriteTool::new(gate, dir.clone());
        let result = tool.run("not json at all").await;
        assert!(result.is_err());
        let err = format!("{}", result.unwrap_err());
        assert!(err.contains("bad JSON args"), "error: {err}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn write_denied_by_gate() {
        let dir = std::env::temp_dir().join("lore-write-deny-test");
        std::fs::create_dir_all(&dir).unwrap();
        let gate = deny_gate(dir.clone());
        let tool = FileWriteTool::new(gate, dir.clone());
        let result = tool.run(r#"{"path":"file.txt","content":"test"}"#).await;
        assert!(result.is_err(), "gate denial must be Err");
        let err = format!("{}", result.unwrap_err());
        assert!(err.contains("policy denied"), "error: {err}");
        std::fs::remove_dir_all(&dir).ok();
    }

    // ── FileEditTool ────────────────────────────────────────────────────

    #[tokio::test]
    async fn edit_single_match() {
        let dir = std::env::temp_dir().join("lore-edit-single-test");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("doc.txt"), "hello world").unwrap();
        let gate = allow_gate(dir.clone());
        let tool = FileEditTool::new(gate, dir.clone());
        let out = tool
            .run(r#"{"path":"doc.txt","old":"hello","new":"goodbye"}"#)
            .await
            .unwrap();
        assert!(out.contains("doc.txt"));
        let read = std::fs::read_to_string(dir.join("doc.txt")).unwrap();
        assert_eq!(read, "goodbye world");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn edit_zero_match() {
        let dir = std::env::temp_dir().join("lore-edit-zero-test");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("doc.txt"), "hello world").unwrap();
        let gate = allow_gate(dir.clone());
        let tool = FileEditTool::new(gate, dir.clone());
        let result = tool
            .run(r#"{"path":"doc.txt","old":"missing","new":"replacement"}"#)
            .await;
        assert!(result.is_err());
        let err = format!("{}", result.unwrap_err());
        assert!(err.contains("0 matches"), "error: {err}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn edit_multi_match() {
        let dir = std::env::temp_dir().join("lore-edit-multi-test");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("doc.txt"), "ha ha ha").unwrap();
        let gate = allow_gate(dir.clone());
        let tool = FileEditTool::new(gate, dir.clone());
        let result = tool
            .run(r#"{"path":"doc.txt","old":"ha","new":"ho"}"#)
            .await;
        assert!(result.is_err());
        let err = format!("{}", result.unwrap_err());
        assert!(err.contains("3 times"), "error: {err}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn edit_escape_rejected() {
        let dir = std::env::temp_dir().join("lore-edit-escape-test");
        std::fs::create_dir_all(&dir).unwrap();
        let gate = allow_gate(dir.clone());
        let tool = FileEditTool::new(gate, dir.clone());
        assert!(
            tool.run(r#"{"path":"../etc/passwd","old":"a","new":"b"}"#)
                .await
                .is_err(),
            "traversal rejected"
        );
        assert!(
            tool.run(r#"{"path":"/etc/passwd","old":"a","new":"b"}"#)
                .await
                .is_err(),
            "absolute rejected"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn edit_bad_json_args() {
        let dir = std::env::temp_dir().join("lore-edit-badjson-test");
        std::fs::create_dir_all(&dir).unwrap();
        let gate = allow_gate(dir.clone());
        let tool = FileEditTool::new(gate, dir.clone());
        let result = tool.run("garbage input").await;
        assert!(result.is_err());
        let err = format!("{}", result.unwrap_err());
        assert!(err.contains("bad JSON args"), "error: {err}");
        std::fs::remove_dir_all(&dir).ok();
    }

    // ── Additional edge-case tests ──────────────────────────────────────

    #[tokio::test]
    async fn write_empty_content() {
        let dir = std::env::temp_dir().join("lore-write-empty-test");
        std::fs::create_dir_all(&dir).unwrap();
        let gate = allow_gate(dir.clone());
        let tool = FileWriteTool::new(gate, dir.clone());
        let out = tool
            .run(r#"{"path":"empty.txt","content":""}"#)
            .await
            .unwrap();
        assert!(out.contains("0 bytes"), "empty content: {out}");
        let read = std::fs::read_to_string(dir.join("empty.txt")).unwrap();
        assert_eq!(read, "", "file should be empty");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn write_symlink_escape_rejected() {
        let dir = std::env::temp_dir().join("lore-symlink-escape-test");
        std::fs::create_dir_all(&dir).unwrap();
        // Create a symlink inside dir pointing outside the root.
        let outside = std::env::temp_dir().join("lore-symlink-outside-target");
        std::fs::create_dir_all(&outside).unwrap();
        std::os::unix::fs::symlink(&outside, dir.join("evil_link")).unwrap();
        let gate = allow_gate(dir.clone());
        let tool = FileWriteTool::new(gate, dir.clone());
        // Writing through the symlink should be rejected (canonicalization
        // resolves it to a path outside the root).
        let result = tool
            .run(r#"{"path":"evil_link/file.txt","content":"escape"}"#)
            .await;
        assert!(result.is_err(), "symlink escape must be rejected");
        let err = format!("{}", result.unwrap_err());
        assert!(
            err.contains("escapes workspace") || err.contains("only relative"),
            "error: {err}"
        );
        std::fs::remove_dir_all(&dir).ok();
        std::fs::remove_dir_all(&outside).ok();
    }

    #[tokio::test]
    async fn edit_nonexistent_file() {
        let dir = std::env::temp_dir().join("lore-edit-nonexistent-test");
        std::fs::create_dir_all(&dir).unwrap();
        let gate = allow_gate(dir.clone());
        let tool = FileEditTool::new(gate, dir.clone());
        let result = tool
            .run(r#"{"path":"no_such_file.txt","old":"a","new":"b"}"#)
            .await;
        assert!(result.is_err(), "editing nonexistent file must be Err");
        let err = format!("{}", result.unwrap_err());
        assert!(
            err.contains("cannot read") || err.contains("not found"),
            "error: {err}"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn edit_empty_old_string() {
        let dir = std::env::temp_dir().join("lore-edit-empty-old-test");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("doc.txt"), "hello world").unwrap();
        let gate = allow_gate(dir.clone());
        let tool = FileEditTool::new(gate, dir.clone());
        // Empty old string matches N+1 times in Rust → multi-match error.
        let result = tool.run(r#"{"path":"doc.txt","old":"","new":"X"}"#).await;
        assert!(result.is_err(), "empty old must fail");
        let err = format!("{}", result.unwrap_err());
        assert!(err.contains("times"), "multi-match error: {err}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn write_empty_path_rejected() {
        let dir = std::env::temp_dir().join("lore-write-empty-path-test");
        std::fs::create_dir_all(&dir).unwrap();
        let gate = allow_gate(dir.clone());
        let tool = FileWriteTool::new(gate, dir.clone());
        let result = tool.run(r#"{"path":"","content":"test"}"#).await;
        assert!(result.is_err(), "empty path must be rejected");
        let err = format!("{}", result.unwrap_err());
        assert!(
            err.contains("file path required") || err.contains("required"),
            "error: {err}"
        );
        std::fs::remove_dir_all(&dir).ok();
    }
}
