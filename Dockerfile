# syntax=docker/dockerfile:1.27@sha256:4edf897a3ffa55b89f906fc8cc78afdb3f1834cc9c7083565e611a8a7d5fe99e
#
# efs-migrate: bit-for-bit reproducible, non-root runtime image.
#
# Every input is pinned by content digest (frontend, base, uv) or hash (Python
# dependencies, from uv.lock), and every timestamp is clamped to
# SOURCE_DATE_EPOCH. Two cold builds of the same commit with the same epoch
# export identical layers and an identical image digest:
#
#   docker buildx build --no-cache --provenance=false --sbom=false \
#     --build-arg SOURCE_DATE_EPOCH="$(git log -1 --format=%ct)" \
#     --output type=oci,dest=image.tar,rewrite-timestamp=true .
#
# scripts/verify_reproducible_image.sh runs exactly that twice and compares.

ARG PYTHON_IMAGE=python:3.12-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.11.7@sha256:240fb85ab0f263ef12f492d8476aa3a2e4e1e333f7d67fbdd923d00a506a516a

FROM ${UV_IMAGE} AS uv

# ---------------------------------------------------------------------------
# builder: wheel -> hash-locked virtualenv -> checked-hash bytecode -> clamp
# ---------------------------------------------------------------------------
FROM ${PYTHON_IMAGE} AS builder

ARG SOURCE_DATE_EPOCH=0
ENV SOURCE_DATE_EPOCH=${SOURCE_DATE_EPOCH} \
    PYTHONHASHSEED=0 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_NO_CACHE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=0

COPY --from=uv /uv /usr/local/bin/uv

WORKDIR /build
COPY pyproject.toml uv.lock README.md ./
COPY src ./src

# 1. Build the project wheel with uv's bundled build backend (pinned with uv).
RUN uv build --wheel --out-dir /build/dist

# 2. Install the locked runtime dependencies (hash-verified, no resolution) and
#    then the wheel into /opt/venv, with no uv cache. uv still records a
#    wall-clock timestamp in <dist-info>/uv_cache.json for any local-path
#    install (it only serves uv's own staleness check), so it is removed along
#    with its RECORD entry, and the build fails if one survives.
RUN uv export --frozen --no-dev --no-emit-project --no-header \
        --format requirements.txt --output-file /build/requirements.txt \
 && uv venv /opt/venv --python /usr/local/bin/python3.12 \
 && uv pip install --python /opt/venv/bin/python --no-deps --require-hashes \
        --requirement /build/requirements.txt \
 && uv pip install --python /opt/venv/bin/python --no-deps /build/dist/*.whl \
 && for cache in /opt/venv/lib/python3.12/site-packages/*.dist-info/uv_cache.json; do \
        [ -e "$cache" ] || continue; \
        sed -i '/\/uv_cache\.json,/d' "$(dirname "$cache")/RECORD"; \
        rm "$cache"; \
    done \
 && ! grep -rqs uv_cache.json /opt/venv/lib/python3.12/site-packages/*.dist-info \
 && [ -z "$(find /opt/venv -name uv_cache.json)" ]

# 3. Pre-compile bytecode. checked-hash .pyc files embed a hash of the source,
#    not its mtime, so they are reproducible and stay valid on a read-only root.
RUN /opt/venv/bin/python -m compileall -q -j 1 --invalidation-mode checked-hash \
        /opt/venv/lib/python3.12/site-packages

# 4. The runtime user, as plain entries (useradd would stamp /etc/shadow with
#    today's date). No home directory and no login shell.
RUN cp /etc/passwd /etc/group /build/ \
 && printf 'appuser:x:10001:10001:efs-migrate:/nonexistent:/usr/sbin/nologin\n' >> /build/passwd \
 && printf 'appuser:x:10001:\n' >> /build/group

# 5. Clamp every timestamp, so the layers do not depend on when they were built.
RUN find /opt/venv /build/passwd /build/group -newermt "@${SOURCE_DATE_EPOCH}" \
        -exec touch --no-dereference --date="@${SOURCE_DATE_EPOCH}" {} +

# ---------------------------------------------------------------------------
# runtime: the same pinned base, the venv, and nothing else
# ---------------------------------------------------------------------------
FROM ${PYTHON_IMAGE} AS runtime

ARG SOURCE_DATE_EPOCH=0

COPY --from=builder /build/passwd /etc/passwd
COPY --from=builder /build/group /etc/group
COPY --from=builder /opt/venv /opt/venv

# Root filesystem can be mounted read-only: nothing is written at runtime
# except, optionally, a --report file on a writable mount such as /tmp.
ENV PATH=/opt/venv/bin:/usr/local/bin:/usr/bin:/bin \
    PYTHONHASHSEED=0 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp

LABEL org.opencontainers.image.title="embedding-feature-store" \
      org.opencontainers.image.description="Blue/green embedding migration pipeline (efs-migrate)"

USER 10001:10001
WORKDIR /tmp

# SIGINT/SIGTERM are handled by efs-migrate itself (exit 130/143 after a
# compensating rollback), so it runs directly as PID 1.
ENTRYPOINT ["/opt/venv/bin/efs-migrate"]
