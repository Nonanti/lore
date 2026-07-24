# Web Dashboard — Design (approved 2026-07-24)

Sub-project 3 of B → D → C. A browser surface for the coworkers platform:
task board, approval inbox, agents + memory browser — over the EXISTING
HTTP API, with zero new dependencies.

## 1. Locked decisions

| # | Decision | Rationale |
|---|----------|-----------|
| U1 | **Single embedded HTML file** (inline CSS+JS, `include_str!`), served at `GET /ui` from the binary. No framework, no npm, no build step, no CDN, system font stack | "Everything lives inside the binary" (DESIGN §2) applies to the UI too; a node toolchain would be the project's first external build dependency |
| U2 | `/ui` sits in the **public route group** (like `/health`): the shell is static and secret-free; every data call is made client-side with the user-supplied Bearer key | The HTML contains nothing an attacker gains from; data endpoints keep their existing auth/rate-limit wall |
| U3 | API key entered in the UI, kept in `localStorage`, sent as `Authorization: Bearer` | Matches the existing auth exactly; standard self-hosted-dashboard tradeoff, documented |
| U4 | **Polling (4s), no SSE/WS** in v1 | D20 discipline: streams add surface without a demonstrated need at this scale; the JSON endpoints are cheap |
| U5 | No new server endpoints | The Phase-D task surface + agents/recall/ask cover every view; the dashboard is a pure consumer |

## 2. Views

- **Tasks** — board of `GET /tasks` (status badges: Queued/Running/
  WaitingApproval/WaitingSubtasks/Completed/Failed), detail drawer with
  `GET /tasks/{id}` (report, children) + `GET /tasks/{id}/log`; new-task
  form → `POST /tasks` (agent, goal, workspace, verify commands).
- **Inbox** — `GET /inbox` pending approvals (action JSON + reason),
  Approve/Deny → `POST /approvals/{id}/approve|deny`.
- **Agents** — `GET /agents` (name/role/traits/version); per agent:
  memory search `GET /agents/{id}/recall?q=..&semantic=true` (score +
  summary), quick ask `POST /agents/{id}/ask`.
- **Header** — API key field (localStorage), connection indicator fed by
  `GET /health` + an authed probe, auto-refresh toggle.

## 3. Testing

- `GET /ui` → 200 `text/html` WITHOUT auth even when an API key is set.
- Shell self-containment: body contains no `http://`/`https://`
  references (no CDN/external assets can creep in unnoticed).
- Existing protected-endpoint auth tests keep passing (no route-group
  regression).

## 4. Addendum (2026-07-24): review fixes

Independent review (2 major, 4 minor, 4 nits):

- **M2 fixed**: `/ui` now sends `X-Frame-Options: DENY` (the approve/deny
  buttons must not be clickjackable), `X-Content-Type-Options: nosniff`,
  and a CSP that allows exactly what the shell is (`default-src 'none';
  style-src/script-src 'unsafe-inline'; connect-src 'self'`) — a CDN
  reference would be browser-blocked even if the self-containment test
  were dodged. Tests assert all three.
- **M1 decided (documented, no code change)**: `ApprovalEntry.action` may
  contain command arguments/env values. The dashboard adds **no new
  exposure** — `GET /inbox` has returned this field to API-key holders
  since the Phase-D task surface; the key IS the operator boundary
  (single-operator threat model, README security notes). Server-side
  redaction would be heuristic (the policy `Action` has no structured
  env-vs-literal distinction) and is deferred until a multi-operator
  story exists.
- Minors fixed: `verify` input now comma-separated (matches
  `Vec<String>`), polling pauses in hidden tabs, stale in-flight detail
  fetches are dropped via a generation counter, decision path segment
  URL-encoded, toast grammar, score esc() uniformity.
- Self-containment test hardened: the shell has NO `src=`/`href=`
  attributes at all (catches protocol-relative and `data:` URIs).
- N4 (space injection via status CSS class) acknowledged, not changed:
  `TaskStatus` is a server-side enum; `esc()` already covers the
  injection-relevant characters.
- CSP side effect (follow-up review #4): `connect-src 'self'` also means
  the shell can only talk to the Lore instance that SERVED it — pointing
  one dashboard at a remote/other instance is deliberately impossible.
  Cross-instance administration is a non-goal; run each instance's own
  `/ui`.

## 5. Out of scope

- Live streams (U4), persona editing UI, task cancellation (no such
  endpoint), charts/metrics visualization, multi-server switching.
