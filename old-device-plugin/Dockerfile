# Stage 1: Build
FROM golang:1.24 AS builder

WORKDIR /app

COPY go.mod go.sum ./
RUN go mod download

COPY . .

RUN go build -o gvirtus-gpu-plugin main.go

# Stage 2: Runtime
FROM debian:bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/gvirtus-gpu-plugin /usr/local/bin/gvirtus-gpu-plugin

ENTRYPOINT ["/usr/local/bin/gvirtus-gpu-plugin"]
