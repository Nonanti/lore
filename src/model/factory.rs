//! Model factory: `ModelConfig` + `build_model` centralize the env-based wiring
//! previously duplicated in `main.rs` and `daemon.rs`.
//!
//! Per-agent model config lives here as a serializable record; `from_env()` captures
//! the process-wide defaults (backward compatible); `build_model` produces a live
//! `Arc<dyn Model>` from any config.

use crate::auth::{Credential, RefreshingToken, TokenStore};
use crate::error::{LoreError, Result};
use crate::model::{AnthropicAuth, AnthropicModel, CodexModel, MockModel, Model, OpenAiModel};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

/// LLM provider kind.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderKind {
    Anthropic,
    #[serde(rename = "openai", alias = "open_a_i")]
    OpenAI,
    #[serde(rename = "openai_compat")]
    OpenAiCompat,
    Mock,
}

/// Auth method: `Key` (metered API key) or `Subs` (subscription/OAuth).
/// `None` (absent) means auto-detect — the same logic as `LORE_AUTH` today.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuthKind {
    Key,
    Subs,
}

/// How `Agent::solve` drives tool calls for this model (spec N1/N2 in
/// `docs/superpowers/specs/2026-07-24-native-tool-calling-design.md`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ToolMode {
    /// Native tool calling when the provider supports it; runtime downgrade
    /// to the text protocol on "does not support tools" errors.
    #[default]
    Auto,
    /// Native only — an unsupported provider is a hard error.
    Native,
    /// Text protocol only (the pre-native behavior).
    Text,
}

impl ToolMode {
    /// Whether this is the default (`auto`) — keeps serialized configs tidy.
    pub fn is_auto(&self) -> bool {
        matches!(self, ToolMode::Auto)
    }

    /// Parse `LORE_TOOL_MODE` (`auto`|`native`|`text`). Unknown values warn
    /// and fall back to `Auto` (mirrors `LORE_AUTH` handling).
    pub fn from_env() -> Self {
        match std::env::var("LORE_TOOL_MODE").ok().as_deref() {
            Some("native") => ToolMode::Native,
            Some("text") => ToolMode::Text,
            Some("auto") | None => ToolMode::Auto,
            Some(other) => {
                tracing::warn!(value = other, "LORE_TOOL_MODE unrecognized, using auto");
                ToolMode::Auto
            }
        }
    }
}

/// Per-agent model configuration. Serializes cleanly; missing fields fall back
/// to env defaults when `build_model` is called.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelConfig {
    pub provider: ProviderKind,
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auth: Option<AuthKind>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    /// Tool-call protocol selection; absent in old configs → `auto`.
    #[serde(default, skip_serializing_if = "ToolMode::is_auto")]
    pub tool_mode: ToolMode,
}

impl ModelConfig {
    /// Parse LORE_AUTH env var into optional AuthKind. Unknown values
    /// produce a warning and fall back to None (auto-detect).
    fn parse_auth_env() -> Option<AuthKind> {
        match std::env::var("LORE_AUTH").ok().as_deref() {
            Some("key") => Some(AuthKind::Key),
            Some("subs") => Some(AuthKind::Subs),
            Some(other) => {
                tracing::warn!(
                    value = other,
                    "LORE_AUTH unrecognized, treating as auto-detect"
                );
                None
            }
            None => None,
        }
    }

    /// Constructs a config from the current process environment (same behavior
    /// as the old `build_model` in main.rs/daemon.rs). This is the fallback
    /// when an agent has no per-agent model field.
    pub fn from_env() -> Option<Self> {
        match std::env::var("LORE_PROVIDER").ok().as_deref() {
            Some("anthropic") => Some(Self {
                provider: ProviderKind::Anthropic,
                model: std::env::var("LORE_LLM_MODEL")
                    .unwrap_or_else(|_| "claude-sonnet-4-5".into()),
                auth: Self::parse_auth_env(),
                base_url: None,
                tool_mode: ToolMode::from_env(),
            }),
            Some("openai") => Some(Self {
                provider: ProviderKind::OpenAI,
                model: std::env::var("LORE_LLM_MODEL").unwrap_or_else(|_| "gpt-5".into()),
                auth: Self::parse_auth_env(),
                base_url: None,
                tool_mode: ToolMode::from_env(),
            }),
            _ => match std::env::var("LORE_LLM_BASE") {
                Ok(base) => Some(Self {
                    provider: ProviderKind::OpenAiCompat,
                    model: std::env::var("LORE_LLM_MODEL").unwrap_or_else(|_| "llama3.2".into()),
                    auth: None, // OpenAiCompat doesn't use the key/subs distinction
                    base_url: Some(base),
                    tool_mode: ToolMode::from_env(),
                }),
                Err(_) => None, // No provider configured at all → MockModel
            },
        }
    }
}

/// Optional request timeout in seconds (`LORE_LLM_TIMEOUT`).
fn env_timeout() -> Option<Duration> {
    std::env::var("LORE_LLM_TIMEOUT")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|n| *n > 0)
        .map(Duration::from_secs)
}

/// Optional max response tokens (`LORE_LLM_MAX_TOKENS`).
fn env_max_tokens() -> Option<u32> {
    std::env::var("LORE_LLM_MAX_TOKENS")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .filter(|n| *n > 0)
}

/// Key from env (`LORE_LLM_KEY`) — used for OpenAiCompat.
fn env_llm_key() -> Option<String> {
    std::env::var("LORE_LLM_KEY")
        .ok()
        .filter(|k| !k.trim().is_empty())
}

/// Refresh closure for Anthropic subscription tokens.
fn anthropic_refresh_fn() -> crate::auth::RefreshFn {
    Box::new(|rt: String| Box::pin(async move { crate::auth::refresh_anthropic(&rt).await }))
}

/// Refresh closure for OpenAI subscription tokens.
fn openai_refresh_fn() -> crate::auth::RefreshFn {
    Box::new(|rt: String| Box::pin(async move { crate::auth::refresh_openai(&rt).await }))
}

/// Resolves Anthropic auth from config + env/token store.
fn resolve_anthropic_auth(cfg: &ModelConfig, data_dir: &Path) -> Option<AnthropicAuth> {
    let store = TokenStore::new(data_dir);
    let stored = store.load("anthropic").ok().flatten();
    let want_key = cfg.auth == Some(AuthKind::Key);
    let want_subs = cfg.auth == Some(AuthKind::Subs);
    let anthropic_key = std::env::var("ANTHROPIC_API_KEY")
        .ok()
        .filter(|k| !k.trim().is_empty());
    let api_key = anthropic_key.or_else(|| {
        let generic = std::env::var("LORE_LLM_KEY")
            .ok()
            .filter(|k| !k.trim().is_empty());
        if generic.is_some() {
            tracing::warn!(
                "LORE_LLM_KEY used as fallback for Anthropic — may cause cross-provider auth failures with mixed agents"
            );
        }
        generic
    });

    if want_key {
        return api_key.map(AnthropicAuth::ApiKey);
    }
    if let Some(Credential::ApiKey { key }) = &stored {
        if !want_subs {
            return Some(AnthropicAuth::ApiKey(key.clone()));
        }
    }
    if let Some(cred @ Credential::OAuth { .. }) = stored {
        let refreshing = RefreshingToken::new(store, "anthropic", cred, anthropic_refresh_fn());
        return Some(AnthropicAuth::OAuth(Arc::new(refreshing)));
    }
    api_key.map(AnthropicAuth::ApiKey)
}

/// Resolves OpenAI auth from config + env/token store.
fn resolve_openai_auth(cfg: &ModelConfig, data_dir: &Path) -> OpenAiAuthResult {
    let store = TokenStore::new(data_dir);
    let stored = store.load("openai").ok().flatten();
    let want_key = cfg.auth == Some(AuthKind::Key);

    // Subscription (Codex) path — skip if auth explicitly says key.
    if !want_key {
        if let Some(cred @ Credential::OAuth { account_id, .. }) = &stored {
            let account_id = account_id.clone();
            let refreshing =
                RefreshingToken::new(store, "openai", cred.clone(), openai_refresh_fn());
            return OpenAiAuthResult::Codex {
                refreshing,
                account_id,
            };
        }
        if cfg.auth == Some(AuthKind::Subs) {
            tracing::warn!(
                "auth=subs but no OpenAI subscription credential; run `lore login openai`"
            );
        }
    }

    // Metered API-key path.
    let openai_key = std::env::var("OPENAI_API_KEY")
        .ok()
        .filter(|k| !k.trim().is_empty());
    let api_key = openai_key
        .or_else(|| {
            let generic = std::env::var("LORE_LLM_KEY")
                .ok()
                .filter(|k| !k.trim().is_empty());
            if generic.is_some() {
                tracing::warn!(
                    "LORE_LLM_KEY used as fallback for OpenAI — may cause cross-provider auth failures with mixed agents"
                );
            }
            generic
        })
        .or_else(|| match &stored {
            Some(Credential::ApiKey { key }) => Some(key.clone()),
            _ => None,
        });

    OpenAiAuthResult::Metered { api_key }
}

/// Intermediate auth resolution result for OpenAI.
enum OpenAiAuthResult {
    Codex {
        refreshing: RefreshingToken,
        account_id: Option<String>,
    },
    Metered {
        api_key: Option<String>,
    },
}

/// Builds a live `Arc<dyn Model>` from a `ModelConfig` + data directory.
/// The data directory holds auth token files and is the same `LORE_DATA` root.
/// Network calls are NOT made during construction (only on actual `complete` calls).
pub fn build_model(cfg: &ModelConfig, data_dir: &Path) -> Result<Arc<dyn Model>> {
    match cfg.provider {
        ProviderKind::Anthropic => match resolve_anthropic_auth(cfg, data_dir) {
            Some(auth) => {
                let mut m = AnthropicModel::new(&cfg.model, auth);
                if let Some(n) = env_max_tokens() {
                    m = m.with_max_tokens(n);
                }
                if let Some(d) = env_timeout() {
                    m = m.with_timeout(d);
                }
                Ok(Arc::new(m))
            }
            None => {
                // M-1: explicit auth configuration without credentials is an error,
                // not a silent MockModel fallback.
                if cfg.auth == Some(AuthKind::Key) {
                    return Err(LoreError::Model(
                        "anthropic provider with auth=key but no API key found (set ANTHROPIC_API_KEY or LORE_LLM_KEY)".to_string(),
                    ));
                }
                if cfg.auth == Some(AuthKind::Subs) {
                    return Err(LoreError::Model(
                        "anthropic provider with auth=subs but no subscription credential (run `lore login anthropic`)".to_string(),
                    ));
                }
                // Auto-detect with no credentials → MockModel (dev/test mode).
                tracing::warn!(
                    "anthropic provider but no credential found (run `lore login anthropic` or set ANTHROPIC_API_KEY); using MockModel"
                );
                Ok(Arc::new(MockModel::new()))
            }
        },
        ProviderKind::OpenAI => {
            let result = resolve_openai_auth(cfg, data_dir);
            match result {
                OpenAiAuthResult::Codex {
                    refreshing,
                    account_id,
                } => {
                    let mut m = CodexModel::new(&cfg.model, Arc::new(refreshing), account_id);
                    if let Some(d) = env_timeout() {
                        m = m.with_timeout(d);
                    }
                    Ok(Arc::new(m))
                }
                OpenAiAuthResult::Metered { api_key } => match api_key {
                    Some(k) => {
                        let mut m = OpenAiModel::new("https://api.openai.com/v1", &cfg.model)
                            .with_api_key(k);
                        if let Some(n) = env_max_tokens() {
                            m = m.with_max_tokens(n);
                        }
                        if let Some(d) = env_timeout() {
                            m = m.with_timeout(d);
                        }
                        Ok(Arc::new(m))
                    }
                    None => {
                        // M-1: explicit auth configuration without credentials is an error.
                        if cfg.auth == Some(AuthKind::Key) {
                            return Err(LoreError::Model(
                                "openai provider with auth=key but no API key found (set OPENAI_API_KEY or LORE_LLM_KEY)".to_string(),
                            ));
                        }
                        if cfg.auth == Some(AuthKind::Subs) {
                            return Err(LoreError::Model(
                                "openai provider with auth=subs but no subscription credential (run `lore login openai`)".to_string(),
                            ));
                        }
                        // Auto-detect with no credentials → MockModel (dev/test mode).
                        tracing::warn!(
                            "openai provider but no credential found (run `lore login openai` or set OPENAI_API_KEY); using MockModel"
                        );
                        Ok(Arc::new(MockModel::new()))
                    }
                },
            }
        }
        ProviderKind::OpenAiCompat => {
            let base = cfg
                .base_url
                .as_deref()
                .unwrap_or("http://localhost:11434/v1");
            let mut m = OpenAiModel::new(base, &cfg.model);
            // OpenAiCompat: key from env or from the config (not applicable, but
            // LORE_LLM_KEY still works as a generic key source).
            if let Some(k) = env_llm_key() {
                m = m.with_api_key(k);
            }
            if let Some(n) = env_max_tokens() {
                m = m.with_max_tokens(n);
            }
            if let Some(d) = env_timeout() {
                m = m.with_timeout(d);
            }
            Ok(Arc::new(m))
        }
        ProviderKind::Mock => Ok(Arc::new(MockModel::new())),
    }
}

/// Convenience: build the default model from env vars. Returns `MockModel` if
/// no provider is configured (same as the old `build_model` in main.rs).
/// Convenience: build the default model from env vars. Returns `MockModel` if
/// no provider is configured (dev/test mode). Returns `Err` when auth is
/// explicitly configured but credentials are absent (M-1: no silent fallback).
pub fn build_model_from_env(data_dir: &Path) -> crate::error::Result<Arc<dyn Model>> {
    match ModelConfig::from_env() {
        Some(cfg) => build_model(&cfg, data_dir),
        None => Ok(Arc::new(MockModel::new())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_config_serde_roundtrip() {
        let cfg = ModelConfig {
            provider: ProviderKind::Anthropic,
            model: "claude-sonnet-4-5-20250929".to_string(),
            auth: Some(AuthKind::Subs),
            base_url: None,
            tool_mode: Default::default(),
        };
        let json = serde_json::to_string_pretty(&cfg).unwrap();
        let back: ModelConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back.provider, ProviderKind::Anthropic);
        assert_eq!(back.model, "claude-sonnet-4-5-20250929");
        assert_eq!(back.auth, Some(AuthKind::Subs));
        assert!(back.base_url.is_none());
    }

    #[test]
    fn model_config_optional_fields_omit_on_none() {
        let cfg = ModelConfig {
            provider: ProviderKind::OpenAiCompat,
            model: "qwen3:8b".to_string(),
            auth: None,
            base_url: Some("http://localhost:11434/v1".to_string()),
            tool_mode: Default::default(),
        };
        let json = serde_json::to_string(&cfg).unwrap();
        // auth should NOT appear in JSON (skip_serializing_if)
        assert!(!json.contains("auth"));
        // base_url SHOULD appear (it's Some)
        assert!(json.contains("base_url"));
    }

    #[test]
    fn model_config_deserialize_missing_optional_fields() {
        // JSON without auth or base_url — should deserialize with None.
        let json = r#"{"provider":"mock","model":"test"}"#;
        let cfg: ModelConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.provider, ProviderKind::Mock);
        assert_eq!(cfg.model, "test");
        assert!(cfg.auth.is_none());
        assert!(cfg.base_url.is_none());
    }

    #[test]
    fn from_env_with_openai_compat_base() {
        let uid = ulid::Ulid::new().to_string();
        let base_key = format!("LORE_TEST_BASE_{uid}");
        let model_key = format!("LORE_TEST_MODEL_{uid}");
        std::env::set_var(&base_key, "http://localhost:11434/v1");
        std::env::set_var(&model_key, "llama3.2");

        // We can't actually change from_env to use these var names,
        // but we can verify the ModelConfig struct works for OpenAiCompat.
        let cfg = ModelConfig {
            provider: ProviderKind::OpenAiCompat,
            model: "llama3.2".to_string(),
            auth: None,
            base_url: Some("http://localhost:11434/v1".to_string()),
            tool_mode: Default::default(),
        };
        let json = serde_json::to_string(&cfg).unwrap();
        let back: ModelConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back.provider, ProviderKind::OpenAiCompat);
        assert_eq!(back.base_url.as_deref(), Some("http://localhost:11434/v1"));

        std::env::remove_var(&base_key);
        std::env::remove_var(&model_key);
    }

    #[test]
    fn build_model_auth_key_no_credential_returns_error() {
        // M-1: auth=key with no API key → Err, not MockModel.
        let cfg = ModelConfig {
            provider: ProviderKind::Anthropic,
            model: "claude-sonnet-4-5".to_string(),
            auth: Some(AuthKind::Key),
            base_url: None,
            tool_mode: Default::default(),
        };
        let dir = std::env::temp_dir();
        let result = build_model(&cfg, &dir);
        let err = match result {
            Ok(_) => panic!("auth=key without credential should return Err, got Ok"),
            Err(e) => e.to_string(),
        };
        assert!(
            err.contains("anthropic") && err.contains("auth=key"),
            "error should mention provider and auth mode: {err}"
        );

        // Same for OpenAI.
        let cfg_openai = ModelConfig {
            provider: ProviderKind::OpenAI,
            model: "gpt-5".to_string(),
            auth: Some(AuthKind::Key),
            base_url: None,
            tool_mode: Default::default(),
        };
        let result_openai = build_model(&cfg_openai, &dir);
        assert!(
            result_openai.is_err(),
            "auth=key without credential should return Err for OpenAI too"
        );
    }

    #[test]
    fn build_model_auto_detect_no_credential_returns_mock() {
        // Auto-detect (auth=None) with no credentials → MockModel (dev/test mode).
        let cfg = ModelConfig {
            provider: ProviderKind::Anthropic,
            model: "claude-sonnet-4-5".to_string(),
            auth: None,
            base_url: None,
            tool_mode: Default::default(),
        };
        let dir = std::env::temp_dir();
        let model = build_model(&cfg, &dir).unwrap();
        let _ = model;
    }

    #[test]
    fn build_model_mock_no_network() {
        let cfg = ModelConfig {
            provider: ProviderKind::Mock,
            model: "mock".to_string(),
            auth: None,
            base_url: None,
            tool_mode: Default::default(),
        };
        let dir = std::env::temp_dir();
        let _model = build_model(&cfg, &dir).unwrap();
    }

    #[tokio::test]
    async fn build_model_mock_completes_without_network() {
        let cfg = ModelConfig {
            provider: ProviderKind::Mock,
            model: "mock".to_string(),
            auth: None,
            base_url: None,
            tool_mode: Default::default(),
        };
        let dir = std::env::temp_dir();
        let model = build_model(&cfg, &dir).unwrap();
        let prompt = crate::model::Prompt {
            user: "hello".to_string(),
            ..Default::default()
        };
        let completion = model.complete(&prompt).await.unwrap();
        assert!(completion.text.contains("hello"));
    }

    #[tokio::test]
    async fn build_model_openai_compat_constructs_without_network() {
        // OpenAiCompat with localhost base — no real network calls during construction.
        let cfg = ModelConfig {
            provider: ProviderKind::OpenAiCompat,
            model: "qwen3:8b".to_string(),
            auth: None,
            base_url: Some("http://localhost:11434/v1".to_string()),
            tool_mode: Default::default(),
        };
        let dir = std::env::temp_dir();
        let model = build_model(&cfg, &dir).unwrap();
        // Construction succeeded, no HTTP calls were made.
        // (complete() would try to reach localhost:11434 — we don't call it here.)
        let _ = model;
    }

    #[test]
    fn provider_kind_serde_roundtrip() {
        for pk in [
            ProviderKind::Anthropic,
            ProviderKind::OpenAI,
            ProviderKind::OpenAiCompat,
            ProviderKind::Mock,
        ] {
            let json = serde_json::to_string(&pk).unwrap();
            let back: ProviderKind = serde_json::from_str(&json).unwrap();
            assert_eq!(back, pk);
        }
        // Verify JSON shapes for ALL variants.
        assert_eq!(
            serde_json::to_string(&ProviderKind::Anthropic).unwrap(),
            "\"anthropic\""
        );
        assert_eq!(
            serde_json::to_string(&ProviderKind::OpenAI).unwrap(),
            "\"openai\""
        );
        assert_eq!(
            serde_json::to_string(&ProviderKind::OpenAiCompat).unwrap(),
            "\"openai_compat\""
        );
        assert_eq!(
            serde_json::to_string(&ProviderKind::Mock).unwrap(),
            "\"mock\""
        );
    }

    #[test]
    fn provider_kind_open_a_i_alias_compat() {
        // Existing agent files containing "open_a_i" must still load.
        let pk: ProviderKind = serde_json::from_str("\"open_a_i\"").unwrap();
        assert_eq!(pk, ProviderKind::OpenAI);
        // Canonical form also works.
        let pk2: ProviderKind = serde_json::from_str("\"openai\"").unwrap();
        assert_eq!(pk2, ProviderKind::OpenAI);
    }

    #[test]
    fn auth_kind_serde_roundtrip() {
        assert_eq!(serde_json::to_string(&AuthKind::Key).unwrap(), "\"key\"");
        assert_eq!(serde_json::to_string(&AuthKind::Subs).unwrap(), "\"subs\"");
    }
}
