# Contributing to Lore

Thanks for your interest in contributing! Lore is a self-contained, native-Rust
core — no external services, everything in one binary. Contributions should keep
that spirit.

## Development setup

```bash
git clone https://github.com/nonantiy/lore
cd lore
cargo build
cargo test
```

## Quality gates

Every change must pass the same checks CI runs:

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
```

Conventions:

- No `unwrap()` / `expect()` on production code paths — return `Result` and handle errors.
- Every module documents the "why", not just the "what".
- New features ship with tests.
- Keep the default build fully offline.

## Optional neural feature

```bash
cargo build --features neural   # downloads an ONNX model on first run
```

The default build stays fully offline; please keep it that way.

## Pull requests

1. Fork and create a topic branch.
2. Make your change with tests.
3. Ensure the quality gates above pass.
4. Open a PR with a clear description of the change and its motivation.

## Reporting bugs

Open an issue with steps to reproduce, expected vs. actual behavior, and your
environment (OS, Rust version). For security issues, see [SECURITY.md](SECURITY.md).
