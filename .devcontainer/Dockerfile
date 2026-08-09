# MFH Studio in a browser — the image a Codespace boots from.
#
# Base: the Microsoft devcontainer image, for the non-root `vscode` user, git,
# and the shell setup Codespaces expects. Julia is added from the official
# tarball rather than a package manager, because the version has to match what
# `Project.toml` declares (`julia = "1.12"`) and distributions lag.

FROM mcr.microsoft.com/devcontainers/base:bookworm

# Keep in step with `julia = "1.12"` in Project.toml and with .github/workflows/CI.yml.
ARG JULIA_VERSION=1.12.6

# `python3` is what MFH Studio's server is written in — standard library only,
# so there is nothing to pip install. `ca-certificates` is needed by Pkg to
# reach the General registry.
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

RUN set -eux; \
    series="$(echo "${JULIA_VERSION}" | cut -d. -f1,2)"; \
    arch="$(dpkg --print-architecture)"; \
    case "${arch}" in \
      amd64) jarch=x86_64; jdir=x64 ;; \
      arm64) jarch=aarch64; jdir=aarch64 ;; \
      *) echo "unsupported architecture ${arch}" >&2; exit 1 ;; \
    esac; \
    url="https://julialang-s3.julialang.org/bin/linux/${jdir}/${series}/julia-${JULIA_VERSION}-linux-${jarch}.tar.gz"; \
    curl -fsSL "${url}" -o /tmp/julia.tar.gz; \
    mkdir -p /opt/julia; \
    tar -xzf /tmp/julia.tar.gz -C /opt/julia --strip-components=1; \
    rm /tmp/julia.tar.gz; \
    ln -s /opt/julia/bin/julia /usr/local/bin/julia; \
    julia --version

# The depot lives in the user's home so that `onCreateCommand` can populate it
# and a Codespaces *prebuild* can bake the result into the image — which is the
# difference between a studio that opens in seconds and one that precompiles
# for several minutes on first use.
ENV JULIA_DEPOT_PATH=/home/vscode/.julia
# Plots' GR backend must never look for a display in a container.
ENV GKSwstype=100
