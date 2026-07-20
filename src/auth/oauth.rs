//! OAuth flows for consumer subscriptions. Self-contained: Lore performs its
//! own PKCE login and token refresh.
//!
//! **Fragility note:** subscription OAuth uses provider constants (client id,
//! endpoints, scopes) that are not officially published for third-party use.
//! They can change or be revoked. The metered API-key path is the stable one.

use super::Pkce;
use crate::error::{LoreError, Result};
use serde::Deserialize;

/// Anthropic (Claude Code) public OAuth client id — Claude Pro/Max login.
pub const ANTHROPIC_CLIENT_ID: &str = "9d1c250a-e61b-44d9-88ed-5944d1962f5e";
const ANTHROPIC_AUTHORIZE: &str = "https://claude.ai/oauth/authorize";
const ANTHROPIC_TOKEN: &str = "https://console.anthropic.com/v1/oauth/token";
/// Redirect used by the manual (paste-the-code) flow.
pub const ANTHROPIC_MANUAL_REDIRECT: &str = "https://console.anthropic.com/oauth/code/callback";
const ANTHROPIC_SCOPE: &str = "org:create_api_key user:profile user:inference";

/// Which login UX to run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AuthFlow {
    /// Spin a localhost listener and capture the redirect automatically.
    Loopback,
    /// Print a URL; the user pastes the resulting code back (SSH/headless safe).
    Manual,
}

/// Result of a token exchange or refresh.
#[derive(Clone, Debug)]
pub struct OAuthOutcome {
    /// Bearer access token.
    pub access: String,
    /// Refresh token (may be rotated by the provider).
    pub refresh: String,
    /// Absolute expiry, unix milliseconds.
    pub expires_ms: i64,
}

/// Minimal percent-encoding for query values (encodes everything outside the
/// RFC 3986 unreserved set). Enough for scopes/redirect URIs/challenges.
fn urlencode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

/// Builds the Anthropic authorize URL. Pure/testable. `state` is echoed back and
/// verified; the PKCE `challenge` binds the redirect to this login.
pub fn anthropic_authorize_url(pkce: &Pkce, redirect_uri: &str, state: &str) -> String {
    format!(
        "{ANTHROPIC_AUTHORIZE}?code=true&client_id={cid}&response_type=code\
         &redirect_uri={redir}&scope={scope}&code_challenge={chal}\
         &code_challenge_method=S256&state={state}",
        cid = ANTHROPIC_CLIENT_ID,
        redir = urlencode(redirect_uri),
        scope = urlencode(ANTHROPIC_SCOPE),
        chal = urlencode(&pkce.challenge),
        state = urlencode(state),
    )
}

/// Splits a manually pasted code. The console callback returns `code#state`.
pub fn split_manual_code(pasted: &str) -> (String, Option<String>) {
    match pasted.trim().split_once('#') {
        Some((c, s)) => (c.to_string(), Some(s.to_string())),
        None => (pasted.trim().to_string(), None),
    }
}

#[derive(Deserialize)]
struct TokenResponse {
    access_token: String,
    #[serde(default)]
    refresh_token: Option<String>,
    #[serde(default)]
    expires_in: Option<i64>,
}

fn client() -> reqwest::Client {
    reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(30))
        .timeout(std::time::Duration::from_secs(60))
        .build()
        .unwrap_or_default()
}

fn expires_ms_from(expires_in: Option<i64>) -> i64 {
    // Default to one hour if the provider omits it; refresh handles the rest.
    let secs = expires_in.unwrap_or(3600).max(0);
    chrono::Utc::now().timestamp_millis() + secs * 1000
}

async fn post_token(body: serde_json::Value) -> Result<OAuthOutcome> {
    let resp = client()
        .post(ANTHROPIC_TOKEN)
        .json(&body)
        .send()
        .await
        .map_err(LoreError::Http)?;
    let status = resp.status();
    let text = resp.text().await.map_err(LoreError::Http)?;
    if !status.is_success() {
        return Err(LoreError::Model(format!(
            "anthropic oauth token endpoint returned {status}"
        )));
    }
    let tr: TokenResponse = serde_json::from_str(&text)?;
    let refresh = tr
        .refresh_token
        .ok_or_else(|| LoreError::Model("anthropic oauth response missing refresh_token".into()))?;
    Ok(OAuthOutcome {
        access: tr.access_token,
        refresh,
        expires_ms: expires_ms_from(tr.expires_in),
    })
}

/// Exchanges an authorization code for tokens (PKCE).
pub async fn exchange_anthropic_code(
    code: &str,
    state: &str,
    verifier: &str,
    redirect_uri: &str,
) -> Result<OAuthOutcome> {
    post_token(serde_json::json!({
        "grant_type": "authorization_code",
        "code": code,
        "state": state,
        "client_id": ANTHROPIC_CLIENT_ID,
        "redirect_uri": redirect_uri,
        "code_verifier": verifier,
    }))
    .await
}

/// Refreshes an access token using the refresh token.
pub async fn refresh_anthropic(refresh_token: &str) -> Result<OAuthOutcome> {
    let mut out = post_token(serde_json::json!({
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": ANTHROPIC_CLIENT_ID,
    }))
    .await?;
    // Some responses omit a rotated refresh token; keep the existing one.
    if out.refresh.is_empty() {
        out.refresh = refresh_token.to_string();
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::pkce;

    #[test]
    fn authorize_url_has_required_params() {
        let p = pkce();
        let url = anthropic_authorize_url(&p, ANTHROPIC_MANUAL_REDIRECT, "st4te");
        assert!(url.starts_with("https://claude.ai/oauth/authorize?"));
        assert!(url.contains(&format!("client_id={ANTHROPIC_CLIENT_ID}")));
        assert!(url.contains("code_challenge_method=S256"));
        assert!(url.contains(&format!("code_challenge={}", p.challenge)));
        assert!(url.contains("state=st4te"));
        assert!(url.contains("response_type=code"));
        // scope space is encoded.
        assert!(url.contains("scope=org%3Acreate_api_key%20user%3Aprofile%20user%3Ainference"));
        // redirect encoded.
        assert!(url.contains("redirect_uri=https%3A%2F%2Fconsole.anthropic.com"));
    }

    #[test]
    fn manual_code_split() {
        assert_eq!(
            split_manual_code("abc#xyz"),
            ("abc".to_string(), Some("xyz".to_string()))
        );
        assert_eq!(
            split_manual_code("  abc#xyz  "),
            ("abc".to_string(), Some("xyz".to_string()))
        );
        assert_eq!(split_manual_code("plain"), ("plain".to_string(), None));
    }

    #[test]
    fn urlencode_unreserved_passthrough() {
        assert_eq!(urlencode("aZ0-._~"), "aZ0-._~");
        assert_eq!(urlencode("a b/c:d"), "a%20b%2Fc%3Ad");
    }

    #[test]
    fn expires_ms_is_in_the_future() {
        let now = chrono::Utc::now().timestamp_millis();
        assert!(expires_ms_from(Some(3600)) >= now + 3_500_000);
        assert!(expires_ms_from(None) >= now + 3_500_000);
        // Negative clamps to now, not the past.
        assert!(expires_ms_from(Some(-10)) >= now);
    }
}
