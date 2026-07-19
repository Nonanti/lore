//! Built-in tools.

use super::Tool;
use crate::error::{LoreError, Result};
use async_trait::async_trait;

/// Simple calculator: resolves a "number operator number" pattern from
/// arguments.
/// Supported: `+ - * / x`. E.g.: "calculate 12 * 3" → "36".
#[derive(Clone, Debug, Default)]
pub struct CalcTool;

impl CalcTool {
    /// New calculator.
    pub fn new() -> Self {
        Self
    }
}

/// Normalizes argument variants from model outputs:
/// delimiters (comma, quotes, parentheses...) are replaced with spaces,
/// operators adjacent to digits are separated ("23+17" → "23 + 17"). Signed
/// numbers ("-5") and scientific notation ("1e-3", "1e+300") are preserved:
/// `+`/`-` count as operators only when the preceding character is a digit or
/// period.
fn normalize_args(args: &str) -> String {
    // 1) Replace delimiters with spaces — "23,17,\"+\"" and JSON-like objects
    // are flattened.
    let cleaned: Vec<char> = args
        .chars()
        .map(|c| match c {
            ',' | ';' | ':' | '"' | '\'' | '{' | '}' | '[' | ']' | '(' | ')' | '=' => ' ',
            other => other,
        })
        .collect();
    // 2) Add spaces around operators adjacent to digits.
    let mut out = String::with_capacity(cleaned.len() + 8);
    for (i, &c) in cleaned.iter().enumerate() {
        let prev_num = i > 0 && (cleaned[i - 1].is_ascii_digit() || cleaned[i - 1] == '.');
        let next_num =
            i + 1 < cleaned.len() && (cleaned[i + 1].is_ascii_digit() || cleaned[i + 1] == '.');
        let split = match c {
            '*' | '/' => true,     // never a sign
            '+' | '-' => prev_num, // at start / after space → sign, stays attached
            // "12x3" → multiplication; don't touch x inside words or hex
            // prefix ("0x1F") — splitting hex would produce silently wrong
            // results.
            'x' => {
                let hex_prefix = i > 0
                    && cleaned[i - 1] == '0'
                    && (i < 2 || !(cleaned[i - 2].is_ascii_digit() || cleaned[i - 2] == '.'));
                prev_num && next_num && !hex_prefix
            }
            _ => false,
        };
        if split {
            out.push(' ');
            out.push(c);
            out.push(' ');
        } else {
            out.push(c);
        }
    }
    out
}

#[async_trait]
impl Tool for CalcTool {
    fn name(&self) -> &str {
        "calc"
    }

    fn description(&self) -> &str {
        "Performs arithmetic with two numbers and an operator (+ - * /)"
    }

    fn args_hint(&self) -> &str {
        r#""<number> <operator> <number>" — e.g. "23 + 17""#
    }

    async fn run(&self, args: &str) -> Result<String> {
        let args = normalize_args(args);
        let mut nums: Vec<f64> = Vec::new();
        let mut op: Option<String> = None;
        for t in args.split_whitespace() {
            if let Ok(n) = t.parse::<f64>() {
                nums.push(n);
            } else if ["+", "-", "*", "/", "x"].contains(&t) {
                op = Some(t.to_string());
            }
        }
        if nums.len() < 2 {
            return Err(LoreError::InvalidInput(
                "two numbers required for calculation".into(),
            ));
        }
        let (a, b) = (nums[0], nums[1]);
        // Operator required: silently summing ambiguous input like "5 5" would
        // produce wrong results — explicit error is safer.
        let Some(op) = op else {
            return Err(LoreError::InvalidInput(
                "operator required (+ - * / x)".into(),
            ));
        };
        let result = match op.as_str() {
            "+" => a + b,
            "-" => a - b,
            "*" | "x" => a * b,
            "/" => {
                if b == 0.0 {
                    return Err(LoreError::InvalidInput("division by zero".into()));
                }
                a / b
            }
            other => {
                return Err(LoreError::InvalidInput(format!(
                    "unknown operator: {other}"
                )))
            }
        };
        // If integer, write without decimal (if it fits in i64 range — float
        // representation on overflow).
        if result.fract() == 0.0 && result.abs() <= i64::MAX as f64 {
            Ok(format!("{}", result as i64))
        } else {
            Ok(format!("{result}"))
        }
    }
}

/// Download cap for the `web` tool (bytes) — to avoid bloating the LLM
/// context.
const WEB_FETCH_CAP: usize = 64 * 1024;

/// Read cap for the `file` tool (bytes).
const FILE_READ_CAP: usize = 64 * 1024;

/// Time tool: current UTC timestamp (LLM's calendar/context need).
#[derive(Clone, Debug, Default)]
pub struct TimeTool;

impl TimeTool {
    /// New time tool.
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Tool for TimeTool {
    fn name(&self) -> &str {
        "time"
    }

    fn description(&self) -> &str {
        "Returns current UTC date and time"
    }

    fn args_hint(&self) -> &str {
        "no arguments needed"
    }

    async fn run(&self, _args: &str) -> Result<String> {
        Ok(chrono::Utc::now()
            .format("%Y-%m-%d %H:%M:%S UTC")
            .to_string())
    }
}

/// Web tool: GETs a URL, returns the text with a size cap.
/// HTTP(S) only; body is truncated at [`WEB_FETCH_CAP`]. Uses a shared client
/// pool.
///
/// SSRF protection: private/loopback/link-local addresses are blocked by
/// default (preventing an LLM-driven tool from accessing the internal
/// network/metadata endpoint).
/// Use `with_private_allowed(true)` to fetch your own services.
/// Note: DNS rebinding is out of scope (domain resolution belongs to reqwest);
/// IP literals and known local names are blocked.
#[derive(Clone, Debug)]
pub struct WebFetchTool {
    client: reqwest::Client,
    allow_private: bool,
}

impl Default for WebFetchTool {
    fn default() -> Self {
        Self::new()
    }
}

impl WebFetchTool {
    /// New web tool (10s timeout, connection-pooled client;
    /// private addresses blocked).
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(10))
                .build()
                .expect("web tool client must be buildable"),
            allow_private: false,
        }
    }

    /// Allows private/loopback addresses (builder; for your own network).
    pub fn with_private_allowed(mut self, allow: bool) -> Self {
        self.allow_private = allow;
        self
    }
}

/// Is the host private/loopback/link-local/reserved? (SSRF blocklist)
fn is_private_host(host: &str) -> bool {
    let h = host.trim_matches(['[', ']']).to_ascii_lowercase();
    if h == "localhost" || h.ends_with(".localhost") {
        return true;
    }
    match h.parse::<std::net::IpAddr>() {
        Ok(std::net::IpAddr::V4(v4)) => {
            v4.is_loopback()
                || v4.is_private()
                || v4.is_link_local()
                || v4.is_unspecified()
                || v4.octets()[0] == 100 && (v4.octets()[1] & 0xC0) == 64 // 100.64.0.0/10 CGNAT
        }
        Ok(std::net::IpAddr::V6(v6)) => {
            v6.is_loopback()
                || v6.is_unspecified()
                || (v6.segments()[0] & 0xfe00) == 0xfc00 // fc00::/7 unique local
                || (v6.segments()[0] & 0xffc0) == 0xfe80 // fe80::/10 link-local
                || v6.to_ipv4_mapped().is_some_and(|m| {
                    m.is_loopback() || m.is_private() || m.is_link_local()
                })
        }
        Err(_) => false, // domain name: resolution is in reqwest (see SSRF note)
    }
}

/// Extracts the host component from a URL (userinfo/port/path are discarded).
fn url_host(url: &str) -> &str {
    let after_scheme = url.split("://").nth(1).unwrap_or(url);
    let after_auth = after_scheme.rsplit('@').next().unwrap_or(after_scheme);
    if let Some(rest) = after_auth.strip_prefix('[') {
        let end = rest.find(']').unwrap_or(rest.len());
        &rest[..end]
    } else {
        after_auth.split([':', '/']).next().unwrap_or("")
    }
}

#[async_trait]
impl Tool for WebFetchTool {
    fn name(&self) -> &str {
        "web"
    }

    fn description(&self) -> &str {
        "Downloads a URL's content (text, size-limited)"
    }

    fn args_hint(&self) -> &str {
        r#""<url>" — e.g. "https://example.com/page""#
    }

    async fn run(&self, args: &str) -> Result<String> {
        let url = args.trim();
        if !(url.starts_with("http://") || url.starts_with("https://")) {
            return Err(LoreError::InvalidInput(
                "only http(s) URLs can be fetched".into(),
            ));
        }
        if !self.allow_private && is_private_host(url_host(url)) {
            return Err(LoreError::InvalidInput(
                "private/local address blocked (SSRF protection)".into(),
            ));
        }
        let resp = self.client.get(url).send().await?;
        let status = resp.status();
        if status == reqwest::StatusCode::NOT_FOUND {
            return Err(LoreError::NotFound(format!("web: 404 ({url})")));
        }
        if !status.is_success() {
            return Err(LoreError::Model(format!("web: {status}")));
        }
        let bytes = resp.bytes().await?;
        let mut text =
            String::from_utf8_lossy(&bytes[..bytes.len().min(WEB_FETCH_CAP)]).into_owned();
        if bytes.len() > WEB_FETCH_CAP {
            text.push_str("\n[... content truncated]");
        }
        Ok(text)
    }
}

/// File tool: reads files UNDER the allowed root directory.
/// Sandbox: `..` traversal and out-of-root absolute paths are rejected — the
/// agent's file access is limited to the single intentionally-granted
/// directory.
#[derive(Clone, Debug)]
pub struct FileReadTool {
    root: std::path::PathBuf,
}

impl FileReadTool {
    /// File tool scoped to the given root (root is created if missing — in
    /// production the tool should not be permanently broken if the directory
    /// is absent).
    pub fn new(root: impl Into<std::path::PathBuf>) -> Self {
        let root = root.into();
        let _ = std::fs::create_dir_all(&root);
        Self { root }
    }
}

#[async_trait]
impl Tool for FileReadTool {
    fn name(&self) -> &str {
        "file"
    }

    fn description(&self) -> &str {
        "Reads a file in the allowed directory (size-limited)"
    }

    fn args_hint(&self) -> &str {
        r#""<relative-path>" — e.g. "notes/todo.txt""#
    }

    async fn run(&self, args: &str) -> Result<String> {
        let rel = args.trim();
        if rel.is_empty() {
            return Err(LoreError::InvalidInput("file path required".into()));
        }
        let p = std::path::Path::new(rel);
        // Sandbox: reject absolute paths and traversal; join + canonicalize to
        // verify it stays under root (symlink escapes are also caught).
        if p.is_absolute() || rel.split(['/', '\\']).any(|s| s == "..") {
            return Err(LoreError::InvalidInput(
                "only relative paths within the allowed directory".into(),
            ));
        }
        let full = self.root.join(p);
        let canon =
            std::fs::canonicalize(&full).map_err(|e| LoreError::NotFound(format!("{rel}: {e}")))?;
        let root_canon =
            std::fs::canonicalize(&self.root).map_err(|e| LoreError::Storage(e.to_string()))?;
        if !canon.starts_with(&root_canon) {
            return Err(LoreError::InvalidInput(
                "path escapes root directory".into(),
            ));
        }
        let bytes = std::fs::read(&canon).map_err(|e| LoreError::Storage(e.to_string()))?;
        let mut text =
            String::from_utf8_lossy(&bytes[..bytes.len().min(FILE_READ_CAP)]).into_owned();
        if bytes.len() > FILE_READ_CAP {
            text.push_str("\n[... content truncated]");
        }
        Ok(text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn calc_basic_ops() {
        let c = CalcTool::new();
        assert_eq!(c.run("2 + 3").await.unwrap(), "5");
        assert_eq!(c.run("calculate 12 * 3").await.unwrap(), "36");
        assert_eq!(c.run("10 - 4").await.unwrap(), "6");
        assert_eq!(c.run("9 / 2").await.unwrap(), "4.5");
    }

    #[tokio::test]
    async fn calc_tolerates_model_arg_formats() {
        // Real LLMs (e.g. qwen3) may send arguments in different formats.
        // The parser must tolerate these — TEST_REPORT §5.1 finding.
        let c = CalcTool::new();
        // Comma-separated + quoted operator (qwen3's actual output).
        assert_eq!(c.run(r#"23,17,"+""#).await.unwrap(), "40");
        // Adjacent compact format.
        assert_eq!(c.run("23+17").await.unwrap(), "40");
        assert_eq!(c.run("12*3").await.unwrap(), "36");
        // JSON-like object argument.
        assert_eq!(c.run(r#"{"a":23,"b":17,"op":"+"}"#).await.unwrap(), "40");
        // Negative numbers are preserved (sign is not confused with operator).
        assert_eq!(c.run("-5 + 3").await.unwrap(), "-2");
        assert_eq!(c.run("5 + -3").await.unwrap(), "2");
        assert_eq!(c.run("5+-3").await.unwrap(), "2");
        // Decimal + compact.
        assert_eq!(c.run("3.5+2").await.unwrap(), "5.5");
        // Scientific notation: e-signs are not mistaken for operators.
        assert_eq!(c.run("1e3+2e4").await.unwrap(), "21000");
        assert_eq!(c.run("1e-3+1e+2").await.unwrap(), "100.001");
        assert_eq!(c.run("1e-3 * 1e3").await.unwrap(), "1");
        // Adjacent multiplication 'x' works but hex prefix (0x...) is not
        // split into multiplication — "0x1F + 1" should give a clear error
        // instead of a silently wrong result (1).
        assert_eq!(c.run("10x3").await.unwrap(), "30");
        assert!(
            c.run("0x1F + 1").await.is_err(),
            "hex argument should error"
        );
    }

    mod props {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            /// Argument normalization must never panic with ANY input
            /// (including unicode, control characters, edge cases).
            #[test]
            fn normalize_never_panics(s in "\\PC*") {
                let _ = normalize_args(&s);
            }

            /// Well-formed "a op b" always computes correctly — with spaces,
            /// adjacent, and comma-separated variants.
            #[test]
            fn well_formed_binary_ops_compute(
                a in -1e6f64..1e6,
                b in 1e-3f64..1e6, // b > 0: division defined, no sign ambiguity
                op in prop::sample::select(vec!['+', '-', '*', '/']),
            ) {
                let rt = tokio::runtime::Builder::new_current_thread().build().unwrap();
                let c = CalcTool::new();
                let expected = match op {
                    '+' => a + b,
                    '-' => a - b,
                    '*' => a * b,
                    _ => a / b,
                };
                for args in [
                    format!("{a} {op} {b}"),
                    format!("{a}{op}{b}"),
                    format!("{a},{b},\"{op}\""),
                ] {
                    let out = rt.block_on(c.run(&args)).unwrap();
                    let got: f64 = out.parse().unwrap();
                    let tol = expected.abs().max(1.0) * 1e-9;
                    prop_assert!(
                        (got - expected).abs() <= tol,
                        "args={args} expected={expected} got={got}"
                    );
                }
            }
        }
    }

    #[tokio::test]
    async fn calc_errors() {
        let c = CalcTool::new();
        assert!(c.run("5 / 0").await.is_err());
        assert!(c.run("single number 5").await.is_err());
        // Two numbers without operator: explicit error instead of silent sum.
        assert!(c.run("5 5").await.is_err());
        // Overflowing integer result: not saturated to i64::MAX, float value
        // is written.
        let big = c.run("1e300 * 1e2").await.unwrap();
        assert_ne!(big, i64::MAX.to_string(), "no saturation");
        assert!(big.len() > 20, "actual magnitude preserved: {big}");
    }
}

#[cfg(test)]
mod native_tools_tests {
    use super::*;

    // ── TimeTool ─────────────────────────────────────────────────────────
    #[tokio::test]
    async fn time_tool_returns_current_utc() {
        let out = TimeTool::new().run("").await.unwrap();
        assert!(out.contains("UTC"), "timestamp: {out}");
        assert!(out.chars().any(|c| c.is_ascii_digit()));
    }

    // ── WebFetchTool ─────────────────────────────────────────────────────
    #[tokio::test]
    async fn web_fetch_gets_text_and_caps_size() {
        use axum::{routing::get, Router};
        let big = "x".repeat(200_000);
        let app = Router::new()
            .route("/ok", get(|| async { "hello web world" }))
            .route(
                "/big",
                get(move || {
                    let b = big.clone();
                    async move { b }
                }),
            )
            .route("/err", get(|| async { axum::http::StatusCode::NOT_FOUND }));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        let t = WebFetchTool::new().with_private_allowed(true); // test server on loopback
        let out = t.run(&format!("http://{addr}/ok")).await.unwrap();
        assert!(out.contains("hello web world"));

        let capped = t.run(&format!("http://{addr}/big")).await.unwrap();
        assert!(
            capped.len() <= WEB_FETCH_CAP + 64,
            "size cap: {}",
            capped.len()
        );

        assert!(
            t.run(&format!("http://{addr}/err")).await.is_err(),
            "404 should error"
        );
        assert!(
            t.run("ftp://something").await.is_err(),
            "non-http(s) rejected"
        );
    }

    // ── FileReadTool ─────────────────────────────────────────────────────
    #[tokio::test]
    async fn file_read_sandboxed_to_root() {
        let root = std::env::temp_dir().join(format!("lore-files-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("not.txt"), "secret note: orange desk").unwrap();

        let t = FileReadTool::new(&root);
        let out = t.run("not.txt").await.unwrap();
        assert!(out.contains("orange desk"));

        // Sandbox violations are rejected.
        assert!(
            t.run("../etc/passwd").await.is_err(),
            "path traversal rejected"
        );
        assert!(
            t.run("/etc/passwd").await.is_err(),
            "out-of-root absolute path rejected"
        );
        assert!(
            t.run("missing.txt").await.is_err(),
            "nonexistent file errors"
        );
        std::fs::remove_dir_all(&root).ok();
    }
}

#[cfg(test)]
mod file_root_tests {
    use super::*;

    #[tokio::test]
    async fn file_tool_creates_missing_root_lazily() {
        // Production finding: root directory was never created — the tool
        // would error on every call out of the box. new() should create the
        // root.
        let root = std::env::temp_dir()
            .join(format!("lore-lazy-{}", std::process::id()))
            .join("nested/deep/directory");
        assert!(!root.exists());
        let t = FileReadTool::new(&root);
        std::fs::write(root.join("a.txt"), "hello").unwrap();
        let out = t.run("a.txt").await.unwrap();
        assert!(out.contains("hello"));
        std::fs::remove_dir_all(root.ancestors().nth(3).unwrap()).ok();
    }
}
