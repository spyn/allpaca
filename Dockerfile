# syntax=docker/dockerfile:1

# CPU-only OpenAI-compatible server. HIP/ROCm stays on the host (see README).
FROM rust:bookworm AS builder
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY crates ./crates
COPY tools ./tools
COPY hip ./hip
RUN cargo build -p allpaca-server --release

FROM debian:bookworm-slim AS runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=builder /src/target/release/allpaca-server /usr/local/bin/allpaca-server
EXPOSE 8080
# Synthetic model needs no weights; mount a GGUF and drop --synthetic for real loads.
ENTRYPOINT ["allpaca-server"]
CMD ["--synthetic", "--cpu-only", "--host", "0.0.0.0", "--port", "8080"]
