# Security Policy

## Supported versions

Lore is pre-1.0; security fixes land on the latest `main`.

## Reporting a vulnerability

Please report security issues privately by email to **nonantiy1@gmail.com**.
Do not open a public issue for an undisclosed vulnerability.

Include:

- A description of the issue and its impact
- Steps to reproduce or a proof of concept
- The affected version / commit

You can expect an initial response within a few days. Once a fix is available,
we will coordinate disclosure.

## Threat model

Lore is a single-operator, trusted-network-first core. Internet exposure should
always go through a reverse proxy with `LORE_API_KEY` + rate limiting + TLS
termination. See the "Security notes (threat model)" section in
[README.md](README.md) for the full model, including:

- No built-in TLS (terminate at the proxy)
- A single API key, compared in constant time
- Input size limits (HTTP body, WebSocket frames, session names, query limits)
- Federation trust (peers authenticated by a shared secret; peer-declared agent
  names are not signed)
- Log hygiene (user text is sanitized; internal errors are not leaked to clients)
- Supply chain: `cargo audit` runs on every CI build
