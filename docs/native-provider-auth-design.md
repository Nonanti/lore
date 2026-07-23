# Lore — Native Provider Auth (OpenAI + Anthropic, API-key + Subscription)

Date: 2026-07-20
Author: Berkant
Status: Approved (design), implementing

## Goal

Add first-class OpenAI and Anthropic providers to Lore, usable **both** with a
metered API key **and** with a consumer subscription (ChatGPT Plus/Pro, Claude
Pro/Max). Lore stays **fully self-contained**: it implements its own OAuth login,
its own token store, and its own refresh — it does **not** read any other tool's
credentials (no Codex/Claude Code files).

## Non-goals

- No reading/sharing of external credential stores.
- No TLS (unchanged: reverse proxy's job).
- Tool-calling parity with the OpenAI path is out of scope for the Anthropic
  provider's first cut (text completion + streaming first).

## Architecture

New self-contained `src/auth/` module + provider clients under `src/model/`.

### `src/auth/` — Lore's own credential subsystem
- `TokenStore`: persists credentials at `LORE_DATA/auth/<provider>.json`,
  `0600`, atomic tmp+rename (same pattern as persona writes).
- `Credential`: `ApiKey(String)` | `OAuth { access, refresh, expires_ms, extra }`.
- `oauth.rs`: PKCE (S256) helpers, per-provider login (build authorize URL,
  exchange code, refresh). Two login flows:
  - **loopback** (default): spin a one-shot `127.0.0.1:PORT` listener, open the
    browser, capture `?code=`.
  - **device/manual** (`--device`): print URL, user pastes the code back.
- Refresh-on-expiry: if `expires_ms` is within a skew window, refresh before use
  and write back.

### Providers
- **OpenAI API-key**: existing `OpenAiModel` (`/chat/completions`) — just supply
  the key. ✅ already works.
- **OpenAI subscription (Codex)**: `src/model/codex.rs`, ChatGPT backend
  Responses API, `Bearer access` + `chatgpt-account-id`. (Phase 2)
- **Anthropic (key + subs)**: `src/model/anthropic.rs`, `AnthropicModel` on the
  `Model` trait. `/v1/messages`. Auth modes:
  - api-key: `x-api-key` + `anthropic-version`.
  - subs: `Authorization: Bearer` + `anthropic-version` +
    `anthropic-beta: claude-code-20250219,oauth-2025-04-20,...` +
    system prompt forced to begin with
    `You are Claude Code, Anthropic's official CLI for Claude.`
  - streaming SSE (Anthropic event format), idle-timeout discipline reused.

### Known OAuth constants (see Alaz note z14fo311ua03fkq05l0ufzwg)
- Anthropic: authorize `https://claude.ai/oauth/authorize`, token
  `https://console.anthropic.com/v1/oauth/token`, client_id
  `9d1c250a-e61b-44d9-88ed-5944d1962f5e`.
- OpenAI Codex: `https://chatgpt.com/backend-api/codex/responses`, token
  `https://auth.openai.com/oauth/token` (Phase 2, exact client_id TBD).

### Selection (env, backward compatible)
- `LORE_PROVIDER=anthropic|openai` + `LORE_AUTH=key|subs` (default: auto-detect
  from the token store; `subs` if an OAuth cred exists, else `key`).
- `LORE_LLM_MODEL` model id (e.g. `claude-sonnet-4-5`).
- `LORE_LLM_BASE` legacy OpenAI-compatible path stays untouched.
- API keys may also come from `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` /
  `LORE_LLM_KEY`.

### CLI
- `lore login <provider> [--device]`, `lore logout <provider>`,
  `lore auth` (status: which providers are configured, expiry).

## Dependencies
- Add `sha2` (PKCE S256). base64url encode is hand-rolled (no dep). Randomness
  for the PKCE verifier reuses two `ulid` values (256 bits). README's dependency
  count is updated.

## Error handling
- All through `LoreError`; **no `unwrap`/`expect`** in non-test code; `tracing`
  for diagnostics; secrets never logged.

## Testing
- Pure-function unit tests: PKCE challenge vector, authorize-URL build, token
  JSON parse, Anthropic payload build (system-prompt injection under oauth,
  `x-api-key` vs Bearer selection), Anthropic SSE parse, TokenStore round-trip +
  `0600` + no tmp leftover, expiry/refresh decision.
- No network in unit tests. Real API validated manually against a live token.

## Phasing
1. `src/auth/` + `AnthropicModel` (api-key + subs) + login/logout/auth CLI +
   wiring. Claude works end-to-end.
2. OpenAI subscription (Codex Responses) + its refresh.
3. Docs (README/CHANGELOG/DESIGN), fragility notes, provider matrix.

## Risks
- Subscription OAuth constants are unofficial and may break; documented as such.
- Anthropic OAuth is scoped to Claude Code/claude.ai; using it elsewhere may
  violate ToS.
