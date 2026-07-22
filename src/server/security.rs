//! Security middleware: API key validation + fixed-window rate limiting.
//!
//! The key is carried via `X-API-Key` or `Authorization: Bearer <key>`.
//! Comparison is constant-time; the rate-limit key is never derived from
//! a raw attacker-controlled header.

use super::state::AppState;

use axum::{
    extract::{ConnectInfo, Request, State},
    http::{header, HeaderMap, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
    Json,
};
use std::net::SocketAddr;

/// Default wait time for `Retry-After` when rate limit is exceeded (seconds).
const DEFAULT_RETRY_AFTER_SECS: u64 = 60;

/// Auth + rate-limit middleware. Error responses have JSON bodies and standard
/// headers (`WWW-Authenticate` / `Retry-After`).
///
/// Failed auth is rate-limited BEFORE the 401 response — brute-force attempts
/// on the API key are throttled per client IP ("fail:{ip}" key in the hits table).
/// This prevents an attacker from sending unlimited invalid keys, which would
/// bypass the normal key-based rate limit entirely.
pub(super) async fn security_mw(State(st): State<AppState>, req: Request, next: Next) -> Response {
    let provided = extract_key(req.headers());
    // Failed auth: apply IP-based fail-rate BEFORE returning 401.
    // Without this, an attacker could brute-force the API key with unlimited
    // requests (the normal rate limit never fires because auth rejects early).
    if !st.authorized(provided.as_deref()) {
        let ip = req
            .extensions()
            .get::<ConnectInfo<SocketAddr>>()
            .map(|ci| ci.0.ip().to_string())
            .unwrap_or_else(|| "anon".to_string());
        let fail_key = format!("fail:{ip}");
        if !st.allow_fail(&fail_key) {
            let retry = st
                .fail_rate
                .map(|r| r.per.as_secs())
                .unwrap_or(DEFAULT_RETRY_AFTER_SECS)
                .to_string();
            return (
                StatusCode::TOO_MANY_REQUESTS,
                [(header::RETRY_AFTER, retry)],
                Json(serde_json::json!({ "error": "rate limit exceeded" })),
            )
                .into_response();
        }
        return (
            StatusCode::UNAUTHORIZED,
            [(header::WWW_AUTHENTICATE, "Bearer")],
            Json(serde_json::json!({ "error": "unauthorized: valid API key required" })),
        )
            .into_response();
    }
    // Rate-limit key: when auth is enabled, the verified API key; otherwise, the
    // client IP (never trust an attacker-controlled header).
    let rl_key = match (&st.api_key, provided) {
        (Some(_), Some(k)) => k,
        _ => req
            .extensions()
            .get::<ConnectInfo<SocketAddr>>()
            .map(|ci| ci.0.ip().to_string())
            .unwrap_or_else(|| "anon".to_string()),
    };
    if !st.allow(&rl_key) {
        let retry = st
            .rate
            .map(|r| r.per.as_secs())
            .unwrap_or(DEFAULT_RETRY_AFTER_SECS)
            .to_string();
        return (
            StatusCode::TOO_MANY_REQUESTS,
            [(header::RETRY_AFTER, retry)],
            Json(serde_json::json!({ "error": "rate limit exceeded" })),
        )
            .into_response();
    }
    next.run(req).await
}

/// Constant-time string comparison (defense against timing attacks): all bytes of
/// equal-length strings are unconditionally XOR'd, with no early exit.
pub(super) fn ct_eq(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

/// Extracts the key from the `X-API-Key` or `Authorization: Bearer <key>` header
/// (scheme name is case-insensitive — RFC 7235).
pub(super) fn extract_key(h: &HeaderMap) -> Option<String> {
    if let Some(v) = h.get("x-api-key").and_then(|v| v.to_str().ok()) {
        return Some(v.to_string());
    }
    h.get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| {
            let scheme = s.get(..7)?;
            scheme
                .eq_ignore_ascii_case("bearer ")
                .then(|| s[7..].to_string())
        })
}

#[cfg(test)]
mod prop_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// Constant-time comparison is exactly equivalent to `==` for correctness
        /// (including unicode).
        #[test]
        fn ct_eq_equals_operator(a in "\\PC{0,64}", b in "\\PC{0,64}") {
            prop_assert_eq!(ct_eq(&a, &b), a == b);
        }

        /// Header extractor must not panic with arbitrary input (including
        /// char-boundary slicing).
        #[test]
        fn extract_key_never_panics(s in "\\PC{0,64}") {
            let mut headers = axum::http::HeaderMap::new();
            if let Ok(v) = axum::http::HeaderValue::from_str(&s) {
                headers.insert(axum::http::header::AUTHORIZATION, v);
            }
            let _ = extract_key(&headers);
        }
    }
}
