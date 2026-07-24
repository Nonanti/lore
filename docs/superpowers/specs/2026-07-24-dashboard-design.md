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

## 4. Out of scope

- Live streams (U4), persona editing UI, task cancellation (no such
  endpoint), charts/metrics visualization, multi-server switching.
