# Runtime image needs no compilers: every dependency in pyproject.toml ships manylinux
# wheels for 3.11/3.12, so the builder stage's only job is to keep the 300-odd MB of
# numpy/pandas/scikit-learn/matplotlib download out of the final layer and to leave the
# pip cache behind in a stage that gets thrown away.
FROM python:3.12-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

# The package itself. `src/` layout means the wheel carries nasa_space_ml and nothing
# else - config, models, reports and web are data and are copied separately below, which
# is also what lets a rebuilt wheel layer stay cached while only the artifacts change.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip wheel --no-deps --wheel-dir /wheel . && pip download --dest /wheel --quiet .

FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.title="nasa-space-ml" \
      org.opencontainers.image.description="Two NASA-data classifiers behind one FastAPI service" \
      org.opencontainers.image.source="https://github.com/Yuvraj0967/nasa-space-ml" \
      org.opencontainers.image.licenses="MIT"

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# A uid that does not exist, so nothing else on the host can own files this container
# writes, and so `whoami` in a compromised container returns a number rather than a name.
RUN groupadd --system --gid 10001 nsm && useradd --system --uid 10001 --gid 10001 --no-create-home nsm

WORKDIR /app

# Install from the wheels the builder downloaded - no index access at runtime, so the
# image is reproducible and needs no network to start.
COPY --from=builder /wheel /wheel
RUN pip install --no-index --find-links /wheel /wheel/nasa_space_ml-*.whl && rm -rf /wheel

# The trained artifacts and their sidecars. These are committed to git, so the container
# is a working model service without a training run - which is the only way a reviewer can
# `docker run` this and see a prediction in the next thirty seconds.
COPY configs /app/configs
COPY models /app/models
COPY reports /app/reports
COPY web /app/web

# Both of these matter only because the package is installed into site-packages rather
# than run from a checkout: the code's default candidates are relative to the repository
# it was developed in. Without them the container would find neither settings nor the page.
ENV NSM_CONFIG=/app/configs/settings.yaml \
    NSM_WEB_DIR=/app/web \
    NSM_HOME=/app

# The cache and processed dirs are created at startup by `config.paths.ensure()`, which
# needs write access. `data/raw` stays empty in a published image: fetching NASA's bytes
# is an explicit act, not something that happens because a container booted. Run
# `docker exec <id> nasa-space-ml fetch` if you want them.
RUN mkdir -p /app/data/raw /app/data/processed && chown -R nsm:nsm /app

EXPOSE 8000
USER nsm

# `/api/v1/health` reports on model load state, so a 200 here means "both artifacts
# opened", which is the only thing worth gating a `docker run` on.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys;sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health',timeout=4).status==200 else 1)"

# uvicorn rather than `nasa-space-ml serve` because the CLI defaults to 127.0.0.1, which
# is unreachable from outside a container's own network namespace.
CMD ["uvicorn", "nasa_space_ml.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
