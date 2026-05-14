# ── Build arguments ───────────────────────────────────────────────────────────
# ARGs declared before the first FROM are global and available in every stage.
ARG UV_VERSION=0.11.14
# Pin UV_VERSION to a specific release tag for reproducible builds,
# e.g. UV_VERSION=0.11.14. See https://github.com/astral-sh/uv/releases
ARG BASE_IMAGE=python:3.12.10-slim

# ── Stage 1: uv-bin ───────────────────────────────────────────────────────────
# Pull the official uv image so we can copy its binary into the runtime image.
# Using a dedicated stage avoids installing uv via pip or curl scripts.
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv-bin

# ── Stage 2: final image ───────────────────────────────────────────────────────
# No USER instruction: Docker runs as root; Apptainer ignores USER anyway and
# runs as the calling user. WORKDIR is /opt/kedro rather than a path under
# /home to avoid Apptainer's default $HOME auto-bind shadowing project files.
FROM ${BASE_IMAGE}

# Copy the uv and uvx binaries from Stage 1 into /bin so they are on PATH.
COPY --from=uv-bin /uv /uvx /bin/

# Instruct uv to compile .py files to .pyc bytecode during install so the
# container starts faster (no compilation on first import).
ENV UV_COMPILE_BYTECODE=1 \
    # By default Dask binds its dashboard to 127.0.0.1 (loopback), which is
    # unreachable from outside the container even with -p. :8787 (no host
    # prefix) makes it listen on all interfaces so -p 8787:8787 works.
    DASK_DISTRIBUTED__SCHEDULER__DASHBOARD_ADDRESS=:8787

# Publish this port with -p 8787:8787 at runtime, then open
# http://localhost:8787/status in your browser to view the Dask dashboard.
EXPOSE 8787

WORKDIR /opt/kedro

# Install system libraries required by the geospatial stack.
# rasterio/pyogrio/exactextract bundle their own GDAL, but that bundled GDAL
# still links against the host libexpat (XML parser). libexpat1 is absent from
# the slim base image, which causes an ImportError on first rasterio import.
# libgomp1 is needed by numba's OpenMP-based parallelism at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libexpat1 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy only the dependency manifest first. This layer is cached independently
# of the source code, so rebuilds after code-only changes skip the slow sync.
COPY pyproject.toml uv.lock ./

# Install all third-party dependencies into .venv but skip the laurel package
# itself (--no-install-project) because its source code isn't copied yet.
# --locked: require uv.lock to be up to date. --no-dev: exclude dev extras.
# --mount=type=cache: persist the uv download cache across builds without
# baking it into the image layer.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project

# Copy the full project source. This invalidates the layer cache, but the
# dependency layer above is already cached so only the package install reruns.
COPY . .

# Install the laurel package itself as a non-editable wheel (--no-editable).
# All dependencies are already present from the previous sync, so only the
# package build-and-install step runs here.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable

# uv run --no-sync: execute kedro using the project's .venv without attempting
# to sync the environment first (it was fully installed in the build above).
# Arguments passed to `docker run` / `apptainer run` are appended here, e.g.:
#   docker run laurel --pipeline=electrify_trips --env=scenarios/sense_512/0
#   apptainer run --bind /data:/opt/kedro/data laurel.sif --pipeline=electrify_trips
# Declare the two directories that must be bind-mounted at runtime.
# data/ holds all pipeline inputs and outputs (excluded from the image by
# .dockerignore). conf/scenarios/ holds generated per-SoW parameter files
# (gitignored); conf/base/ is already baked into the image.
#
# Docker:
#   docker run \
#     --mount type=bind,src=/host/path/to/data,dst=/opt/kedro/data \
#     --mount type=bind,src=/host/path/to/conf/scenarios,dst=/opt/kedro/conf/scenarios \
#     laurel --pipeline=electrify_trips --env=scenarios/sense_512/0
#
# Apptainer:
#   apptainer run \
#     --bind /host/path/to/data:/opt/kedro/data \
#     --bind /host/path/to/conf/scenarios:/opt/kedro/conf/scenarios \
#     laurel.sif --pipeline=electrify_trips --env=scenarios/sense_512/0
VOLUME ["/opt/kedro/data", "/opt/kedro/conf/scenarios"]

ENTRYPOINT ["uv", "run", "--no-sync", "kedro", "run"]
# Default to --help so running the container with no arguments prints usage
# rather than attempting to run all pipelines.
CMD ["--help"]
