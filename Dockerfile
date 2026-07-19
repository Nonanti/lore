# Lore — self-contained AI agent infrastructure.
# Multi-stage build: compilation in the Rust image, runtime on minimal Debian.
#
#   docker build -t lore .
#   docker run -p 3777:3777 -v lore-data:/data \
#     -e LORE_LLM_BASE=http://host.docker.internal:11434/v1 lore

FROM rust:1-slim AS builder
WORKDIR /build
COPY Cargo.toml Cargo.lock ./
COPY src ./src
RUN cargo build --release --locked

FROM debian:bookworm-slim
RUN useradd --system --home /data lore
COPY --from=builder /build/target/release/lore /usr/local/bin/lore
USER lore
ENV LORE_DATA=/data
VOLUME /data
EXPOSE 3777
ENTRYPOINT ["lore"]
CMD ["serve", "--addr", "0.0.0.0:3777"]
