#!/usr/bin/env bash
# Launch a command inside this project's container (Docker locally, Apptainer on
# an HPC cluster), or execute it directly if already inside one.
#
# Nothing here names a specific project. Every per-project value is derived from
# pyproject.toml and the git remote, each with a KEDRO_CONTAINER_* override -- see
# docs/source/containers.md for the full knob table and adoption checklist.
#
# Usage:
#   ./scripts/setup/run-in-container.sh kedro run --pipelines=evaluate
#   ./scripts/setup/run-in-container.sh bash
#   ./scripts/setup/run-in-container.sh --print-image     # for use in build/push commands
#   ./scripts/setup/run-in-container.sh --print-package
#
# Written for bash 3.2 (macOS's shipped /bin/bash) as well as newer bash on Linux
# clusters: no associative arrays, no `mapfile`.
set -euo pipefail

# -----------------------------------------------------------------------------
# Recursion short-circuit: if we are already inside the container (the wrapper
# set this itself before invoking docker/apptainer), just run the command. This
# is what lets a generated SLURM script call this wrapper unchanged whether it
# is invoked from the host or from an interactive shell already inside the
# container.
# -----------------------------------------------------------------------------
if [ "${KEDRO_CONTAINER_INSIDE:-0}" = "1" ]; then
  exec "$@"
fi

# -----------------------------------------------------------------------------
# Must run from the project root: that is where pyproject.toml lives, and it is
# what the default bind paths (./conf, ./data, ./logs, ./scripts) are relative
# to -- consistent with Kedro's own convention that `kedro run` is invoked from
# the project root.
# -----------------------------------------------------------------------------
if [ ! -f pyproject.toml ]; then
  echo "run-in-container.sh: no pyproject.toml in the current directory." >&2
  echo "  Run this from the project root (where pyproject.toml lives)." >&2
  exit 1
fi

# -----------------------------------------------------------------------------
# Derive the package name from pyproject.toml's [tool.kedro] table.
#
# Uses awk, not `python -c 'import tomllib'`: this script also runs on cluster
# login/compute nodes, whose system python3 may predate tomllib (3.11). awk has
# no such dependency. Scoped to the [tool.kedro] section (not a blind grep) so a
# same-named key elsewhere in the file can't be picked up by accident.
# -----------------------------------------------------------------------------
PKG="${KEDRO_CONTAINER_PACKAGE:-}"
if [ -z "$PKG" ]; then
  PKG=$(awk '
    /^\[tool\.kedro\]/ { in_section=1; next }
    /^\[/               { in_section=0 }
    in_section && /^package_name[[:space:]]*=/ {
      match($0, /"[^"]*"/)
      print substr($0, RSTART+1, RLENGTH-2)
      exit
    }
  ' pyproject.toml)
fi
if [ -z "$PKG" ]; then
  echo "run-in-container.sh: could not derive the package name from" >&2
  echo "  pyproject.toml's [tool.kedro] package_name. Set KEDRO_CONTAINER_PACKAGE." >&2
  exit 1
fi

# -----------------------------------------------------------------------------
# Derive the registry owner from the git remote, tolerating a missing remote
# (e.g. a fresh clone with no origin yet, or no git repo at all).
# -----------------------------------------------------------------------------
GIT_OWNER=""
if REMOTE_URL=$(git remote get-url origin 2>/dev/null); then
  GIT_OWNER=$(echo "$REMOTE_URL" | sed -E 's#^(https://github\.com/|git@github\.com:)([^/]+)/.*#\2#')
fi

# -----------------------------------------------------------------------------
# All per-project knobs. Every one has a KEDRO_CONTAINER_* override.
#
# Docker/OCI image references MUST be lowercase; a GitHub org or user name is
# under no such constraint (this repo's own org is
# "Stanford-Sustainable-Systems-Lab"). Lowercase only for the derived image
# ref -- `tr`, not bash 4's ${var,,}, since this must also run on macOS's
# shipped bash 3.2.
# -----------------------------------------------------------------------------
if [ -n "${KEDRO_CONTAINER_IMAGE:-}" ]; then
  IMAGE="$KEDRO_CONTAINER_IMAGE"
elif [ -n "$GIT_OWNER" ]; then
  GIT_OWNER_LC=$(echo "$GIT_OWNER" | tr '[:upper:]' '[:lower:]')
  PKG_LC=$(echo "$PKG" | tr '[:upper:]' '[:lower:]')
  IMAGE="ghcr.io/${GIT_OWNER_LC}/${PKG_LC}:dev"
else
  IMAGE=$(echo "${PKG}:dev" | tr '[:upper:]' '[:lower:]')
fi

SIF="${KEDRO_CONTAINER_SIF:-./${PKG}.sif}"

# --print-* flags: host-side introspection for use in other commands
# (`docker build -t "$(./scripts/setup/run-in-container.sh --print-image)"`,
# `apptainer build "$PKG.sif" "docker://$(... --print-image)"`), so those
# commands never need to hardcode the project name either.
case "${1:-}" in
  --print-image)   echo "$IMAGE"; exit 0 ;;
  --print-package) echo "$PKG"; exit 0 ;;
esac

if [ "$#" -eq 0 ]; then
  echo "usage: $0 <command> [args...]" >&2
  echo "       $0 --print-image | --print-package" >&2
  exit 1
fi

PROJECT_ROOT="$PWD"
CONF_DIR="${KEDRO_CONTAINER_CONF_DIR:-$PROJECT_ROOT/conf}"
DATA_DIR="${KEDRO_CONTAINER_DATA_DIR:-$PROJECT_ROOT/data}"
LOGS_DIR="${KEDRO_CONTAINER_LOGS_DIR:-$PROJECT_ROOT/logs}"
SCRIPTS_DIR="${KEDRO_CONTAINER_SCRIPTS_DIR:-$PROJECT_ROOT/scripts}"
MOUNT_SRC="${KEDRO_CONTAINER_MOUNT_SRC:-1}"
THREADS="${KEDRO_CONTAINER_THREADS:-1}"
# $SCRATCH, not $L_SCRATCH: the caches redirected here (numba, PyTensor, matplotlib,
# $HOME) are only worth redirecting if they SURVIVE. $L_SCRATCH is node-local and
# destroyed when the job ends, so every array task would start cold -- and on a login
# node it is unset entirely, silently falling back to /tmp. It is also a hard failure
# rather than a slow one: `apptainer --bind` errors out if the source path is missing,
# which is what a stale $L_SCRATCH path becomes the moment its job exits. $TMPDIR/tmp
# remains the last resort for a machine with no $SCRATCH at all, i.e. Docker on a Mac.
CACHE_ROOT="${KEDRO_CONTAINER_CACHE_ROOT:-${SCRATCH:-${TMPDIR:-/tmp}}/${PKG}-cache}"
EXTRA_ARGS="${KEDRO_CONTAINER_EXTRA_ARGS:-}"

# -----------------------------------------------------------------------------
# Create host-side directories that will be bind targets. A read-only SIF
# without an overlay cannot create a mount point, and a missing bind *source*
# is a hard error for `apptainer --bind` (Docker is more forgiving but the
# behaviour should not differ between the two). conf/local is required because
# Kedro's default_run_env is "local". Every /cache subdirectory must be
# pre-created because binding $CACHE_ROOT onto /cache replaces the image's own
# /cache wholesale, hiding the subdirectories baked into the image -- and $HOME
# in particular is not auto-created by anything that writes to it.
# -----------------------------------------------------------------------------
mkdir -p \
  "$CONF_DIR/local" \
  "$DATA_DIR" \
  "$LOGS_DIR/slurm" \
  "$SCRIPTS_DIR/scenarios" \
  "$CACHE_ROOT/home" "$CACHE_ROOT/xdg" "$CACHE_ROOT/mpl" \
  "$CACHE_ROOT/numba" "$CACHE_ROOT/pyc" "$CACHE_ROOT/pytensor" "$CACHE_ROOT/uv" \
  "$CACHE_ROOT/viz"

# -----------------------------------------------------------------------------
# Defensive cleanup: a `*.egg-info/` directory under src/ is disposable build
# metadata (setuptools writes it as a side effect of an editable install;
# nothing at runtime reads it -- confirmed the host's own `import <pkg>` and
# venv keep working the moment it is removed). It is regenerated by ordinary
# host-side `uv sync` / `uv run` activity, so this is not a one-time cleanup:
# left in place, a MOUNT_SRC=1 bind re-imports it into /app/src on every run
# alongside the image's own /opt/venv/.../<pkg>-*.dist-info, and
# importlib.metadata then reports the package as TWO distributions. Kedro
# resolves hooks and plugins via entry points, so that duplication makes
# registration order-dependent -- a real, silent correctness bug, not cosmetic.
# Only relevant when src/ is actually bind-mounted in.
# -----------------------------------------------------------------------------
if [ "$MOUNT_SRC" = 1 ]; then
  rm -rf "$PROJECT_ROOT"/src/*.egg-info
fi

# -----------------------------------------------------------------------------
# Provenance banner. A run with MOUNT_SRC=1 is NOT reproducible from the image
# tag alone -- the tag then describes only the dependencies, not the code that
# actually ran. Printed to stderr on every invocation so it lands in SLURM logs
# rather than being discovered months later.
# -----------------------------------------------------------------------------
HOST_GIT_SHA=unknown
HOST_GIT_STATE=unknown
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  HOST_GIT_SHA=$(git rev-parse HEAD 2>/dev/null || echo unknown)
  if [ -n "$(git status --porcelain 2>/dev/null)" ]; then
    HOST_GIT_STATE=dirty
  else
    HOST_GIT_STATE=clean
  fi
fi
echo "[run-in-container] pkg=$PKG image=$IMAGE mount_src=$MOUNT_SRC" \
     "host_git_sha=$HOST_GIT_SHA ($HOST_GIT_STATE)" >&2
if [ "$MOUNT_SRC" = 1 ] && [ "$HOST_GIT_STATE" = dirty ]; then
  echo "[run-in-container] WARNING: source is bind-mounted from a DIRTY working" \
       "tree; this run is not reproducible from the image tag alone." >&2
fi

# -----------------------------------------------------------------------------
# Dispatch: Apptainer if a built .sif is present and apptainer is on PATH,
# otherwise Docker. A generated SLURM script always reaches this branch (it
# runs on the host, not inside the container); interactive Mac use always takes
# the Docker branch, since there is no .sif on a Mac.
# -----------------------------------------------------------------------------
if [ -f "$SIF" ] && command -v apptainer >/dev/null 2>&1; then
  binds=(
    --bind "$CONF_DIR:/app/conf"
    --bind "$DATA_DIR:/app/data"
    --bind "$LOGS_DIR:/app/logs"
    --bind "$SCRIPTS_DIR:/app/scripts"
    --bind "$CACHE_ROOT:/cache"
    # kedro-viz writes .viz/ under the CWD on every `kedro run`. Making /app/.viz
    # world-writable in the image is enough for Docker (writable container layer)
    # but not for Apptainer: a SIF is an immutable squashfs, so the write fails
    # with ENOTDIR/EROFS regardless of mode. Bind it, or every array task logs two
    # WARNINGs. Non-fatal either way -- the pipeline itself completes.
    --bind "$CACHE_ROOT/viz:/app/.viz"
  )
  if [ "$MOUNT_SRC" = 1 ]; then
    binds+=(--bind "$PROJECT_ROOT/src:/app/src")
  fi

  # --contain: empty, session-scoped /tmp and $HOME instead of Apptainer's
  #   default auto-bind of the host's -- otherwise a cluster-side
  #   ~/.local/lib/pythonX.Y/site-packages would join sys.path and could shadow
  #   a pinned dependency.
  # --cleanenv: drop the host environment (module-loaded PYTHONPATH /
  #   LD_LIBRARY_PATH) and keep only the image's own ENV plus what is passed
  #   with --env below.
  # --pwd /app: mandatory under --contain, since the host CWD is no longer
  #   auto-mounted and Apptainer's default working directory is the host's CWD.
  # Not --containall: the extra PID/IPC namespace containment and --no-init
  #   buy nothing for a single-user batch job, and --no-init risks orphaning
  #   multiprocessing children if a Kedro ParallelRunner is ever used.
  # --home: NOT redundant with the image's ENV HOME=/cache/home. Apptainer
  #   translates a Docker ENV into `export VAR="${VAR:-default}"`, so the image's
  #   value only applies when the variable is otherwise unset -- and Apptainer
  #   always pre-sets HOME to the host's home path, so the image's default never
  #   wins. Under --contain that path is an empty 64 MB RAM-backed tmpfs, so every
  #   $HOME write would be lost per run and charged to the job's memory limit:
  #   exactly what redirecting caches to /cache exists to prevent. `--env HOME=`
  #   cannot fix it either -- Apptainer refuses it ("Overriding HOME environment
  #   variable with APPTAINERENV_HOME is not permitted"); --home is the only
  #   supported route, and its src:dest form both binds and sets HOME.
  exec apptainer exec --contain --cleanenv --pwd /app \
    --home "$CACHE_ROOT/home:/cache/home" \
    "${binds[@]}" \
    --env "KEDRO_CONTAINER_INSIDE=1" \
    --env "OMP_NUM_THREADS=$THREADS" \
    --env "OPENBLAS_NUM_THREADS=$THREADS" \
    --env "MKL_NUM_THREADS=$THREADS" \
    --env "NUMEXPR_NUM_THREADS=$THREADS" \
    --env "NUMBA_NUM_THREADS=$THREADS" \
    "$SIF" "$@"
fi

tty_flags=()
if [ -t 0 ] && [ -t 1 ]; then
  tty_flags+=(-it)
fi

binds=(
  -v "$CONF_DIR:/app/conf"
  -v "$DATA_DIR:/app/data"
  -v "$LOGS_DIR:/app/logs"
  -v "$SCRIPTS_DIR:/app/scripts"
  -v "$CACHE_ROOT:/cache"
)
if [ "$MOUNT_SRC" = 1 ]; then
  binds+=(-v "$PROJECT_ROOT/src:/app/src")
fi

# -u $(id -u):$(id -g): matches the Apptainer uid model (runs as the invoking
#   user, not root), so a bug that only reproduces under a foreign uid shows up
#   locally too, rather than only on the cluster.
#
# "${tty_flags[@]+"${tty_flags[@]}"}", not the plain "${tty_flags[@]}": under
# `set -u`, bash < 4.4 (including macOS's shipped bash 3.2) treats a reference
# to a zero-element array as an unbound variable. The `+` form only expands
# the array if it is set, empty or not, which is safe on every bash this
# script targets. (tty_flags is the only array here that can be empty --
# binds always has at least five elements.)
# shellcheck disable=SC2086
exec docker run --rm "${tty_flags[@]+"${tty_flags[@]}"}" \
  -w /app \
  -u "$(id -u):$(id -g)" \
  "${binds[@]}" \
  -e "KEDRO_CONTAINER_INSIDE=1" \
  -e "OMP_NUM_THREADS=$THREADS" \
  -e "OPENBLAS_NUM_THREADS=$THREADS" \
  -e "MKL_NUM_THREADS=$THREADS" \
  -e "NUMEXPR_NUM_THREADS=$THREADS" \
  -e "NUMBA_NUM_THREADS=$THREADS" \
  $EXTRA_ARGS \
  "$IMAGE" "$@"
