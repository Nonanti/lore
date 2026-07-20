//! Lore's own credential subsystem — fully self-contained.
//!
//! Lore does **not** read any other tool's credentials. It runs its own OAuth
//! login ([`oauth`]), stores tokens under `LORE_DATA/auth/<provider>.json`
//! (`0600`, atomic write), and refreshes them itself. Providers are consumable
//! with a metered **API key** or a consumer **subscription** (OAuth).

mod oauth;

pub use oauth::{
    anthropic_authorize_url, exchange_anthropic_code, exchange_openai_code, openai_authorize_url,
    refresh_anthropic, refresh_openai, split_manual_code, AuthFlow, OAuthOutcome,
    ANTHROPIC_CLIENT_ID, ANTHROPIC_MANUAL_REDIRECT, OPENAI_CLIENT_ID,
};

use crate::error::{LoreError, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

/// Skew before real expiry at which an OAuth token is proactively refreshed.
const REFRESH_SKEW_SECS: i64 = 120;

/// Yields a currently-valid bearer/access token (or API key) on demand,
/// refreshing behind the scenes when an OAuth token is near expiry.
#[async_trait]
pub trait AccessTokenProvider: Send + Sync {
    /// Returns a token ready to put on the wire.
    async fn access_token(&self) -> Result<String>;
}

/// A fixed token that never refreshes (API keys, or a caller-managed token).
pub struct StaticToken(pub String);

#[async_trait]
impl AccessTokenProvider for StaticToken {
    async fn access_token(&self) -> Result<String> {
        Ok(self.0.clone())
    }
}

/// An OAuth token that refreshes itself and persists the rotation to the store.
/// The refresh closure is provider-specific (injected), keeping this type free
/// of any single provider's endpoints.
pub struct RefreshingToken {
    store: TokenStore,
    provider: String,
    current: tokio::sync::Mutex<Credential>,
    refresh: RefreshFn,
}

/// Provider-specific refresh: given a refresh token, returns fresh OAuth tokens.
pub type RefreshFn =
    Box<dyn Fn(String) -> futures::future::BoxFuture<'static, Result<OAuthOutcome>> + Send + Sync>;

impl RefreshingToken {
    /// Builds a refreshing token source seeded with the stored credential.
    pub fn new(
        store: TokenStore,
        provider: impl Into<String>,
        seed: Credential,
        refresh: RefreshFn,
    ) -> Self {
        Self {
            store,
            provider: provider.into(),
            current: tokio::sync::Mutex::new(seed),
            refresh,
        }
    }
}

#[async_trait]
impl AccessTokenProvider for RefreshingToken {
    async fn access_token(&self) -> Result<String> {
        let mut cur = self.current.lock().await;
        // Cross-process mitigation: another Lore process (e.g. CLI vs serve
        // sharing LORE_DATA) may have refreshed already. Re-read disk before
        // minting a new token so we do not invalidate a peer's rotated refresh
        // token. This narrows — but does not fully close — the multi-process
        // refresh race (there is no cross-process file lock).
        if cur.is_expired(REFRESH_SKEW_SECS) {
            if let Ok(Some(disk)) = self.store.load(&self.provider) {
                if !disk.is_expired(REFRESH_SKEW_SECS) {
                    *cur = disk;
                }
            }
        }
        if cur.is_expired(REFRESH_SKEW_SECS) {
            if let Credential::OAuth {
                refresh,
                account_id,
                ..
            } = &*cur
            {
                let refresh_tok = refresh.clone();
                let prev_account = account_id.clone();
                let out = (self.refresh)(refresh_tok).await?;
                let fresh = Credential::OAuth {
                    access: out.access,
                    refresh: out.refresh,
                    expires_ms: out.expires_ms,
                    // Prefer a freshly-issued account id; keep the previous one.
                    account_id: out.account_id.or(prev_account),
                };
                self.store.save(&self.provider, &fresh)?;
                *cur = fresh;
                tracing::info!(provider = %self.provider, "refreshed oauth access token");
            }
        }
        match &*cur {
            Credential::OAuth { access, .. } => Ok(access.clone()),
            Credential::ApiKey { key } => Ok(key.clone()),
        }
    }
}

/// A stored credential for one provider. Serialized with a `type` tag so the
/// on-disk shape is self-describing and forward compatible.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Credential {
    /// Metered API key (official, stable path).
    #[serde(rename = "apikey")]
    ApiKey {
        /// The raw provider key.
        key: String,
    },
    /// OAuth tokens from a consumer subscription (ChatGPT / Claude Pro-Max).
    #[serde(rename = "oauth")]
    OAuth {
        /// Bearer access token.
        access: String,
        /// Refresh token (used to mint a fresh access token).
        refresh: String,
        /// Absolute expiry, unix milliseconds.
        #[serde(rename = "expires")]
        expires_ms: i64,
        /// Provider-specific account id (e.g. ChatGPT `chatgpt-account-id`).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        account_id: Option<String>,
    },
}

impl Credential {
    /// Whether an OAuth access token is expired (or expires within `skew_secs`).
    /// API keys never expire.
    pub fn is_expired(&self, skew_secs: i64) -> bool {
        match self {
            Credential::ApiKey { .. } => false,
            Credential::OAuth { expires_ms, .. } => {
                let now = chrono::Utc::now().timestamp_millis();
                *expires_ms <= now + skew_secs * 1000
            }
        }
    }

    /// True for the subscription (OAuth) variant.
    pub fn is_oauth(&self) -> bool {
        matches!(self, Credential::OAuth { .. })
    }
}

/// Provider name guard: only `[a-z0-9-]`, 1..=32 chars. Prevents path traversal
/// via the `<provider>.json` file name.
fn valid_provider(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 32
        && name
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-')
}

/// On-disk credential store: `LORE_DATA/auth/<provider>.json`.
pub struct TokenStore {
    dir: PathBuf,
}

impl TokenStore {
    /// Store rooted at `<data_dir>/auth`.
    pub fn new(data_dir: impl AsRef<Path>) -> Self {
        Self {
            dir: data_dir.as_ref().join("auth"),
        }
    }

    fn path(&self, provider: &str) -> Result<PathBuf> {
        if !valid_provider(provider) {
            return Err(LoreError::InvalidInput(format!(
                "invalid provider name: {provider}"
            )));
        }
        Ok(self.dir.join(format!("{provider}.json")))
    }

    /// Loads a provider's credential, or `None` if not logged in.
    pub fn load(&self, provider: &str) -> Result<Option<Credential>> {
        let path = self.path(provider)?;
        match std::fs::read_to_string(&path) {
            Ok(s) => Ok(Some(serde_json::from_str(&s)?)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(LoreError::Storage(e.to_string())),
        }
    }

    /// Persists a credential atomically (tmp + rename) with `0600` permissions.
    pub fn save(&self, provider: &str, cred: &Credential) -> Result<()> {
        let path = self.path(provider)?;
        std::fs::create_dir_all(&self.dir).map_err(|e| LoreError::Storage(e.to_string()))?;
        // Owner-only directory (0700): peers cannot even enumerate provider names.
        set_owner_only(&self.dir, 0o700)?;
        let json = serde_json::to_string_pretty(cred)?;
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, json).map_err(|e| LoreError::Storage(e.to_string()))?;
        set_owner_only(&tmp, 0o600)?;
        std::fs::rename(&tmp, &path).map_err(|e| LoreError::Storage(e.to_string()))?;
        Ok(())
    }

    /// Removes a provider's credential (logout). Missing file is not an error.
    pub fn delete(&self, provider: &str) -> Result<()> {
        let path = self.path(provider)?;
        match std::fs::remove_file(&path) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(LoreError::Storage(e.to_string())),
        }
    }

    /// Lists providers that currently have a stored credential.
    pub fn list(&self) -> Result<Vec<String>> {
        let mut out = Vec::new();
        let entries = match std::fs::read_dir(&self.dir) {
            Ok(e) => e,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(out),
            Err(e) => return Err(LoreError::Storage(e.to_string())),
        };
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if let Some(p) = name.strip_suffix(".json") {
                out.push(p.to_string());
            }
        }
        out.sort();
        Ok(out)
    }
}

/// Restricts a path to owner-only access (`mode`, e.g. `0o600`/`0o700`) on Unix.
/// On non-Unix it cannot enforce this and logs a warning so operators do not
/// trust a "private" claim the filesystem can't back.
fn set_owner_only(path: &Path, mode: u32) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode))
            .map_err(|e| LoreError::Storage(e.to_string()))?;
    }
    #[cfg(not(unix))]
    {
        let _ = mode;
        tracing::warn!(
            path = %path.display(),
            "non-unix platform: cannot restrict credential file permissions; \
             the token store may be world-readable"
        );
    }
    Ok(())
}

/// Decodes URL-safe base64 (padding optional). Used to read JWT id-token
/// claims (e.g. the ChatGPT account id). Returns `None` on an invalid byte.
pub(crate) fn b64url_decode(s: &str) -> Option<Vec<u8>> {
    fn val(c: u8) -> Option<u32> {
        match c {
            b'A'..=b'Z' => Some((c - b'A') as u32),
            b'a'..=b'z' => Some((c - b'a' + 26) as u32),
            b'0'..=b'9' => Some((c - b'0' + 52) as u32),
            b'-' => Some(62),
            b'_' => Some(63),
            _ => None,
        }
    }
    let mut out = Vec::with_capacity(s.len() * 3 / 4);
    let mut buf = 0u32;
    let mut bits = 0u32;
    for &c in s.as_bytes() {
        if c == b'=' {
            break;
        }
        buf = (buf << 6) | val(c)?;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((buf >> bits) as u8);
        }
    }
    Some(out)
}

/// URL-safe base64 without padding (RFC 4648 §5). Used for PKCE.
pub(crate) fn b64url_nopad(input: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    let mut out = String::with_capacity(input.len().div_ceil(3) * 4);
    for chunk in input.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = *chunk.get(1).unwrap_or(&0) as u32;
        let b2 = *chunk.get(2).unwrap_or(&0) as u32;
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(T[((n >> 18) & 63) as usize] as char);
        out.push(T[((n >> 12) & 63) as usize] as char);
        if chunk.len() > 1 {
            out.push(T[((n >> 6) & 63) as usize] as char);
        }
        if chunk.len() > 2 {
            out.push(T[(n & 63) as usize] as char);
        }
    }
    out
}

/// A generated PKCE pair.
#[derive(Clone, Debug)]
pub struct Pkce {
    /// High-entropy verifier (kept client-side).
    pub verifier: String,
    /// `base64url(sha256(verifier))` — sent as `code_challenge`.
    pub challenge: String,
}

/// Generates a PKCE S256 pair. 256 bits of entropy from two ULIDs.
pub fn pkce() -> Pkce {
    let mut raw = Vec::with_capacity(32);
    raw.extend_from_slice(&ulid::Ulid::new().to_bytes());
    raw.extend_from_slice(&ulid::Ulid::new().to_bytes());
    let verifier = b64url_nopad(&raw);
    let challenge = b64url_nopad(&sha256(verifier.as_bytes()));
    Pkce {
        verifier,
        challenge,
    }
}

fn sha256(data: &[u8]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(data);
    h.finalize().into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn b64url_matches_known_vectors() {
        // RFC 4648 test vectors, url-safe, no padding.
        assert_eq!(b64url_nopad(b""), "");
        assert_eq!(b64url_nopad(b"f"), "Zg");
        assert_eq!(b64url_nopad(b"fo"), "Zm8");
        assert_eq!(b64url_nopad(b"foo"), "Zm9v");
        assert_eq!(b64url_nopad(b"foob"), "Zm9vYg");
        assert_eq!(b64url_nopad(b"fooba"), "Zm9vYmE");
        assert_eq!(b64url_nopad(b"foobar"), "Zm9vYmFy");
    }

    #[test]
    fn b64url_decode_roundtrips() {
        for v in [&b""[..], b"f", b"fo", b"foo", b"foobar", b"hello world!"] {
            let enc = b64url_nopad(v);
            assert_eq!(
                b64url_decode(&enc).as_deref(),
                Some(v),
                "roundtrip for {v:?}"
            );
        }
        assert!(b64url_decode("not base64!!").is_none());
    }

    #[test]
    fn pkce_challenge_is_sha256_of_verifier() {
        // RFC 7636 Appendix B canonical vector.
        let verifier = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk";
        let challenge = b64url_nopad(&sha256(verifier.as_bytes()));
        assert_eq!(challenge, "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM");
    }

    #[test]
    fn pkce_pair_roundtrips() {
        let p = pkce();
        // verifier is 43 chars (32 bytes base64url no pad), challenge too.
        assert_eq!(p.verifier.len(), 43);
        assert_eq!(p.challenge.len(), 43);
        assert_eq!(p.challenge, b64url_nopad(&sha256(p.verifier.as_bytes())));
    }

    #[test]
    fn provider_name_guard() {
        assert!(valid_provider("anthropic"));
        assert!(valid_provider("openai-codex"));
        assert!(!valid_provider(""));
        assert!(!valid_provider("../etc"));
        assert!(!valid_provider("Anthropic"));
        assert!(!valid_provider("a/b"));
    }

    #[test]
    fn store_roundtrip_and_logout() {
        let dir = std::env::temp_dir().join(format!("lore-auth-{}", ulid::Ulid::new()));
        let store = TokenStore::new(&dir);
        assert_eq!(store.list().unwrap(), Vec::<String>::new());

        let cred = Credential::OAuth {
            access: "at".into(),
            refresh: "rt".into(),
            expires_ms: 42,
            account_id: Some("acc".into()),
        };
        store.save("anthropic", &cred).unwrap();
        assert_eq!(store.load("anthropic").unwrap(), Some(cred));
        assert_eq!(store.list().unwrap(), vec!["anthropic".to_string()]);

        // No tmp leftover.
        let leftovers: Vec<_> = std::fs::read_dir(dir.join("auth"))
            .unwrap()
            .flatten()
            .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("tmp"))
            .collect();
        assert!(leftovers.is_empty(), "tmp leftover: {leftovers:?}");

        store.delete("anthropic").unwrap();
        assert_eq!(store.load("anthropic").unwrap(), None);
        store.delete("anthropic").unwrap(); // idempotent
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn saved_file_is_0600() {
        use std::os::unix::fs::PermissionsExt;
        let dir = std::env::temp_dir().join(format!("lore-auth-perm-{}", ulid::Ulid::new()));
        let store = TokenStore::new(&dir);
        store
            .save("anthropic", &Credential::ApiKey { key: "k".into() })
            .unwrap();
        let mode = std::fs::metadata(dir.join("auth/anthropic.json"))
            .unwrap()
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(mode, 0o600, "file expected 0600, got {mode:o}");
        let dmode = std::fs::metadata(dir.join("auth"))
            .unwrap()
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(dmode, 0o700, "dir expected 0700, got {dmode:o}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn expiry_logic() {
        let now = chrono::Utc::now().timestamp_millis();
        let fresh = Credential::OAuth {
            access: "a".into(),
            refresh: "r".into(),
            expires_ms: now + 3_600_000,
            account_id: None,
        };
        let stale = Credential::OAuth {
            access: "a".into(),
            refresh: "r".into(),
            expires_ms: now - 1000,
            account_id: None,
        };
        assert!(!fresh.is_expired(60));
        assert!(stale.is_expired(60));
        assert!(!Credential::ApiKey { key: "k".into() }.is_expired(60));
    }
}
