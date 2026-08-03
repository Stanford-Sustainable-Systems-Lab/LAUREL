# syntax=docker/dockerfile:1.7
#
# Pins BOTH CPython and uv to exact versions -- via TWO images, not one. Astral's
# combined `ghcr.io/astral-sh/uv:<pyver>-<os>` tags only exist for uv's *latest*
# release; there is no per-patch-version `0.11.32-python3.13-bookworm-slim` tag
# (confirmed: `docker manifest inspect` 404s on it). The official Python image
# does publish exact patch tags, so: pin Python from there, and pin uv by
# COPY-ing its binary out of the version-tagged (but Python-less) uv image.
# Match both tags to your .python-version and installed `uv --version`.
FROM python:3.13.12-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:0.11.32 /uv /uvx /usr/local/bin/

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

WORKDIR /app

# =============================================================================
# Layer 1 -- third-party dependencies ONLY. Slow (~1.2 GB, minutes) and cached.
#
# VERIFIED: `uv sync --locked --no-install-project` succeeds with only these two
# files present -- no src/, no README -- even when the project version is
# `dynamic`. So this layer references no package name, and neither a source edit
# nor a version bump can invalidate it.
# =============================================================================
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-group docs

# =============================================================================
# Layer 2 -- the project itself, installed EDITABLE. Fast (seconds).
#
# `COPY . ./` deliberately copies "everything .dockerignore allows", making
# .dockerignore the single source of truth for image contents. This keeps the
# Dockerfile agnostic to whether a given project happens to have tests/,
# conf/logging.yml, etc. -- a missing optional directory cannot break the build.
# =============================================================================
COPY . ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-group docs

# Kedro's standard project directories, created so they exist as bind targets:
# a read-only SIF cannot create a mount point without an overlay. conf/local is
# required because Kedro's default_run_env is "local". .viz is NOT a bind
# target -- it is kedro-viz's own session-stats directory, written directly
# under the CWD by a hook that runs on every `kedro run` if kedro-viz is
# installed. chmod 777 for the same reason as /cache below: the wrapper always
# runs as the invoking uid (confirmed: `docker run -u $(id -u):$(id -g)`
# without this produced "Permission denied: '/app/.viz'", since WORKDIR /app
# is owned by root at the default build-time umask).
RUN mkdir -p /app/conf/local /app/data /app/logs /app/scripts /app/notebooks /app/.viz \
    && chmod -R 777 /app/conf/local /app/data /app/logs /app/scripts /app/notebooks /app/.viz

# Cache root. Deliberately NOT under /tmp: `apptainer --contain` gives an empty,
# session-scoped, possibly RAM-backed /tmp, which would make every cache cold and
# charge it against the job's memory limit. Created here so the bind target
# exists; 777 because the image is built as root but Apptainer runs as the
# invoking user, whose uid is unknown at build time.
RUN mkdir -p /cache/home /cache/xdg /cache/mpl /cache/numba /cache/pyc \
             /cache/pytensor /cache/uv && chmod -R 777 /cache

# =============================================================================
# GENERIC RUN-TIME ENV -- applies to any containerized Kedro project.
#
# Apptainer runs as the *invoking user* against an *immutable* SIF, so anything
# that writes must be pointed at the bound /cache. Split into commented groups
# because a Dockerfile cannot take per-line comments inside a continued ENV.
# =============================================================================

# --- Where "the user's home" is ----------------------------------------------
# HOME: with `--contain` the host $HOME is not mounted, so an inherited HOME
#   points at a nonexistent path; under `docker run -u $(id -u)` there is no
#   /etc/passwd entry for the uid, so HOME may be unset or "/". Either way,
#   anything writing to $HOME (ipython, jupyter, pip) fails. (Without
#   `--contain`, Apptainer overrides this with the real host $HOME -- harmless,
#   since that is also writable.)
# XDG_CACHE_HOME: defaults to $HOME/.cache. Set explicitly so XDG-respecting
#   libraries (fontconfig, parts of the Jupyter stack) land in /cache even if
#   something overrides HOME behind our back.
# USER: with no /etc/passwd entry for a foreign/mapped uid, `getpass.getuser()`
#   raises OSError outright rather than falling back gracefully -- it only
#   short-circuits successfully if LOGNAME/USER/LNAME/USERNAME is already set.
#   Kedro itself guards its one getuser() call, but CONFIRMED this still broke
#   pytest's built-in tmp_path fixture (tmp_path_factory names its base temp
#   dir from getuser(), unguarded) with "OSError: No username set in the
#   environment" under `docker run -u $(id -u):$(id -g)`. Generic to any
#   foreign-uid run, not specific to this project's tests.
ENV HOME=/cache/home \
    XDG_CACHE_HOME=/cache/xdg \
    USER=container

# --- Keep the host's Python out of our sys.path ------------------------------
# PYTHONNOUSERSITE: drops ~/.local/lib/pythonX.Y/site-packages from sys.path.
#   This is the important one. `--contain` already hides the host $HOME, but if
#   anyone ever drops that flag, a cluster-side user install of numpy or numba
#   silently shadows the version pinned in uv.lock. Defence in depth against the
#   worst failure mode: an image that reports the wrong versions.
ENV PYTHONNOUSERSITE=1

# --- Bytecode ----------------------------------------------------------------
# PYTHONPYCACHEPREFIX: writes __pycache__ trees under /cache instead of beside
#   the source. Two reasons: with src/ bind-mounted read-write, CPython would
#   otherwise litter the *host* tree with files owned by the container uid; and
#   when /app is read-only, bytecode is silently re-compiled on every run.
ENV PYTHONPYCACHEPREFIX=/cache/pyc

# --- Observability under a batch scheduler -----------------------------------
# PYTHONUNBUFFERED: when stdout is a pipe or file (always, under SLURM) CPython
#   block-buffers it, so log lines appear only at job exit -- or are lost
#   entirely when a job is killed at its time limit. Unbuffered output is what
#   makes a failed array task diagnosable at all.
# DO_NOT_TRACK / KEDRO_DISABLE_TELEMETRY: if kedro-telemetry is installed, this
#   avoids an outbound network call on every kedro invocation -- which on a
#   compute node with no egress means a timeout, not an error -- and avoids
#   writing .telemetry into a possibly read-only /app. Both names are set
#   because which one is honoured varies by kedro-telemetry version.
ENV PYTHONUNBUFFERED=1 \
    DO_NOT_TRACK=1 \
    KEDRO_DISABLE_TELEMETRY=1

# --- uv guardrail ------------------------------------------------------------
# Run time uses /opt/venv/bin directly, never `uv run`. These make a stray
# `uv run` degrade to a pass-through instead of failing while trying to re-lock a
# read-only filesystem. Set AFTER the syncs: they conflict with `--locked`.
# UV_CACHE_DIR is needed because `uv run --no-sync` still writes a cache and
# takes a .lock there (verified).
ENV UV_FROZEN=1 \
    UV_NO_SYNC=1 \
    UV_CACHE_DIR=/cache/uv

# --- Thread oversubscription -------------------------------------------------
# BLAS and OpenMP runtimes size their thread pools from the number of *visible*
# cores. A container on a shared cluster node sees every physical core on the box
# (~128), NOT the cgroup's cpus-per-task. Left alone, each of N array tasks
# spawns ~128 threads and the node thrashes on context switching rather than
# computing. 1 is the correct *default*; the wrapper raises it to
# $SLURM_CPUS_PER_TASK for runs that genuinely want threads.
# Separate variables because each library reads only its own: numpy links either
# OpenBLAS or MKL, pandas/numexpr read NUMEXPR_*, scipy and numba's OpenMP layer
# read OMP_*. NUMBA_NUM_THREADS additionally caps numba's own parallel=True
# pool, which ignores OMP_NUM_THREADS.
ENV OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    NUMBA_NUM_THREADS=1

# =============================================================================
# PROJECT-SPECIFIC RUN-TIME ENV -- EDIT THIS BLOCK for your stack.
# Everything above is generic; everything below assumes a particular scientific
# Python stack. Delete what you do not use.
# =============================================================================

# --- matplotlib / seaborn ----------------------------------------------------
# MPLBACKEND=Agg: matplotlib's default backend probes for a GUI toolkit. There is
#   no display in a container, so this either warns or hard-fails. Agg is the
#   headless raster backend.
# MPLCONFIGDIR: matplotlib builds a font cache on first import. If the location
#   is unwritable it prints "Matplotlib created a temporary cache directory" on
#   *every* import and rebuilds the font list each time (seconds). Across a large
#   array that is minutes of pure waste.
ENV MPLBACKEND=Agg \
    MPLCONFIGDIR=/cache/mpl

# --- numba -------------------------------------------------------------------
# NUMBA_CACHE_DIR: destination for numba's on-disk JIT cache (for functions
#   decorated cache=True). May be inert if the project's kernels do not opt in,
#   but must already be writable for that optimization to become possible, and
#   numba probes the path regardless.
ENV NUMBA_CACHE_DIR=/cache/numba

# --- PyMC / PyTensor ---------------------------------------------------------
# Two settings in one variable, per PyTensor's own format:
#   cxx=            -> empty value tells PyTensor there is NO C++ compiler, so it
#                      never attempts compilation and never emits a warning
#                      storm. This is the agreed no-g++ decision. It is safe ONLY
#                      if sampling goes through a non-C backend (e.g. nutpie with
#                      the numba backend). A plain pm.sample() would fall back to
#                      PyTensor's slow Python linker.
#   base_compiledir -> where PyTensor caches compiled graph modules. Defaults to
#                      ~/.pytensor, a well-known source of lock contention when
#                      several processes share it, so it must be node-local.
ENV PYTENSOR_FLAGS=cxx=,base_compiledir=/cache/pytensor

# =============================================================================
ARG GIT_SHA=unknown
ENV IMAGE_GIT_SHA=$GIT_SHA
LABEL org.opencontainers.image.revision=$GIT_SHA

# No ENTRYPOINT on purpose: `apptainer exec` bypasses an ENTRYPOINT while
# `apptainer run` honours it, so an ENTRYPOINT makes the two diverge.
CMD ["bash"]
