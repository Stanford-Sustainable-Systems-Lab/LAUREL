# Running the project in a container

This project ships a container image that runs interactively (Docker, e.g. on a Mac) and as
SLURM job arrays on an HPC cluster (Apptainer / Singularity), from the same `Dockerfile`.

## What this is

Three files:

- **`Dockerfile`** — builds one image, in two layers. Layer 1 installs third-party
  dependencies only (slow, cached across every code change); layer 2 installs this project
  itself, editable, on top (fast). The virtual environment lives at `/opt/venv`, *outside*
  `/app`, so bind-mounting host directories into `/app` at run time can never hide the
  interpreter. Because the install is editable, bind-mounting a live `src/` over `/app/src`
  swaps in your working tree with zero rebuild.
- **`.dockerignore`** — an allowlist. Since the `Dockerfile`'s second layer is `COPY . ./`,
  this file is the single source of truth for what enters the image.
- **`scripts/setup/run-in-container.sh`** — a wrapper that derives this project's identity
  (package name, registry image, cache paths) from `pyproject.toml` and the git remote, then
  launches Docker or Apptainer with the right binds, or executes directly if already inside
  the container. Writable state (JIT caches, `$HOME`, font caches, …) is redirected to a
  single bound directory, `/cache`, because Apptainer's `/tmp` is empty and possibly
  RAM-backed under `--contain`. It also removes any stray `src/*.egg-info/` before a
  bind-mounted run — see the caveat below.

None of the three files name this project. They can be copied into another uv-managed Kedro
project unchanged — see *Adopting this elsewhere* below.

## Adopting this in another Kedro project

Copy `Dockerfile`, `.dockerignore`, and `scripts/setup/run-in-container.sh` unchanged, then:

1. Match the `Dockerfile`'s two version-pinned tags to the new project: the `FROM
   python:<version>-slim-bookworm` tag to `.python-version`, and the
   `COPY --from=ghcr.io/astral-sh/uv:<version>` tag to the `uv --version` this project is
   built with. (There is no single combined `uv:<version>-python<x.y>-<os>` tag per uv
   release — only for uv's `latest` — hence the two separate images.)
2. Review the **PROJECT-SPECIFIC RUN-TIME ENV** block near the bottom of the `Dockerfile`
   (matplotlib, numba, Dask); delete whichever groups the new project does not use, and add
   a group for any library the new project has that writes to a fixed path under `$HOME`.
   The block above it is generic and should not need editing.
3. Extend `.dockerignore`'s allowlist if the project keeps source outside `src/` and
   `conf/base/`.
4. Confirm `scripts/setup/**` is not gitignored — the wrapper must be committed even in a
   project where `scripts/` otherwise holds generated output.
5. If the project generates its own SLURM scripts (as this one's `build_scenarios` pipeline
   does), point their launcher prefix at `./scripts/setup/run-in-container.sh`. A batch
   script runs on the compute node *outside* the container, so each array task must launch
   the container itself.

Nothing else is project-specific: the wrapper derives the package name from
`pyproject.toml`'s `[tool.kedro] package_name` and the registry owner from
`git remote get-url origin`, with a `KEDRO_CONTAINER_*` override for every derived value (see
the knob table below). There are no placeholders to find-and-replace.

## Everyday usage

### Locally, with Docker

```bash
# after a dependency change -- the only thing that needs a rebuild
uv add "some-pkg>=1.2"
docker build -t "$(./scripts/setup/run-in-container.sh --print-image)" \
  --build-arg GIT_SHA="$(git rev-parse HEAD)" .

# run a pipeline against LIVE source -- no rebuild, ever
./scripts/setup/run-in-container.sh kedro run --pipeline=evaluate_impacts --env=scenarios/test/task_0

# interactive shell
./scripts/setup/run-in-container.sh bash

# kedro-viz. --host 0.0.0.0 is required: it defaults to 127.0.0.1, which
# inside a container is unreachable from the host.
KEDRO_CONTAINER_EXTRA_ARGS="-p 4141:4141" \
  ./scripts/setup/run-in-container.sh kedro viz run --host 0.0.0.0 --port 4141 --no-browser

# tests. --no-cov / -p no:cacheprovider because pytest's configured addopts
# write .coverage and .pytest_cache into /app, which is not bound by default.
./scripts/setup/run-in-container.sh pytest -m "not slow" --no-cov -p no:cacheprovider
```

### On the cluster, with Apptainer

```bash
# One-time setup. Redirect Apptainer's cache and temp space off $HOME: the build
# unpacks the entire root filesystem before squashing it, which will otherwise
# exhaust the default quota. Both go on $SCRATCH, never $L_SCRATCH -- that one
# exists only for the lifetime of a job, so anything pointing at it breaks the
# moment the job ends, and on a login node it is unset entirely.
mkdir -p "$SCRATCH"/.apptainer/{cache,tmp}
export APPTAINER_CACHEDIR=$SCRATCH/.apptainer/cache
export APPTAINER_TMPDIR=$SCRATCH/.apptainer/tmp
command -v apptainer                 # on Sherlock it is already on $PATH, with no module

# Only if the registry package is private -- which is the default for a package
# published by a workflow authenticating with GITHUB_TOKEN. Log in with a classic
# PAT carrying `read:packages` (authorized for the org, if it enforces SAML SSO):
docker login ghcr.io                 # writes ~/.docker/config.json

# Apptainer does NOT read ~/.docker/config.json on its own -- it looks only in
# ~/.apptainer/docker-config.json -- so `docker login` alone is not enough. Point
# at the file explicitly with --authfile, which takes that same Docker format.
# On a cluster with no Docker client, copy config.json over from a host that has one.

# Build on a compute node, not the login node. Prefer the digest over a tag: `:dev`
# is mutable and moves on every push to the default branch, so only `…@sha256:…`
# pins what actually went into the SIF.
PKG=$(./scripts/setup/run-in-container.sh --print-package)
mkdir -p "$GROUP_HOME/$PKG"
apptainer build --authfile "$HOME/.docker/config.json" \
  "$GROUP_HOME/$PKG/$PKG.sif" \
  "docker://$(./scripts/setup/run-in-container.sh --print-image)"

# The SIF lives outside the repo because a 15 GB $HOME cannot hold one, and it is
# symlinked in because the wrapper's default KEDRO_CONTAINER_SIF is ./<pkg>.sif and
# its `-f` test follows symlinks. $GROUP_HOME rather than $SCRATCH: a SIF that is
# only ever read would eventually hit the 90-day purge.
ln -s "$GROUP_HOME/$PKG/$PKG.sif" "./$PKG.sif"

# one interactive pipeline
export KEDRO_CONTAINER_CONF_DIR=$GROUP_HOME/$PKG/conf
export KEDRO_CONTAINER_DATA_DIR=$SCRATCH/$PKG/data
./scripts/setup/run-in-container.sh kedro run --pipeline=evaluate_impacts --env=scenarios/test/task_0

# a SLURM array, using a scenario script that build_scenarios generated. The name is
# the builder's `display_name` param, not the env directory name: conf/scenario_builders/
# boot_scale sets display_name: boot_scale_small, so it emits boot_scale_small.sh.
KEDRO_CONTAINER_MOUNT_SRC=0 sbatch scripts/scenarios/boot_scale_small.sh
```

`KEDRO_CONTAINER_MOUNT_SRC=0` for `sbatch` is the recommended discipline: it runs the code
baked into the image rather than a live bind-mount, so the image tag alone identifies what
produced the results. `sbatch` exports the submitting environment by default, so it reaches
every array task.

**Tagging.** Tag each build with the git SHA and keep a moving `:dev` tag for
local iteration. `.github/workflows/docker-publish.yml` does both on every push to the
default branch — `type=sha` and `type=raw,value=dev` in its `metadata-action` step — so
`:dev` is what `run-in-container.sh --print-image` resolves to and `sha-<short>` is the
immutable-per-commit alternative. Registry tags are mutable; the image digest is not, so
cite `…@sha256:…`
for anything that needs to be traced back later. The `GIT_SHA` build-arg lands in an OCI
label, and `apptainer build` carries OCI labels into the `.sif`, so
`apptainer inspect <pkg>.sif` recovers the code state from the artifact alone, with no
registry lookup needed.

**The live-mount caveat.** A run with `KEDRO_CONTAINER_MOUNT_SRC=1` (the default, for
interactive iteration) is *not* reproducible from the image tag — the tag then describes
only the dependencies, not the code. The wrapper prints a one-line banner on every
invocation naming the image, whether source is mounted, and the host's git SHA and dirty
state, so this is visible in every SLURM log rather than discovered months later.

## Configuration knobs

All prefixed `KEDRO_CONTAINER_`; every one has a sensible derived default.

| Variable | Default | Meaning |
| --- | --- | --- |
| `KEDRO_CONTAINER_PACKAGE` | `[tool.kedro] package_name` in `pyproject.toml` | the project's package name |
| `KEDRO_CONTAINER_IMAGE` | `ghcr.io/<git-owner>/<pkg>:dev` | image reference for `docker build`/`run` |
| `KEDRO_CONTAINER_SIF` | `./<pkg>.sif` | path to the built Apptainer image; if present (and `apptainer` is on `PATH`) the wrapper uses Apptainer instead of Docker |
| `KEDRO_CONTAINER_CONF_DIR` | `$PWD/conf` | host dir bound onto `/app/conf` |
| `KEDRO_CONTAINER_DATA_DIR` | `$PWD/data` | host dir bound onto `/app/data` |
| `KEDRO_CONTAINER_LOGS_DIR` | `$PWD/logs` | host dir bound onto `/app/logs` |
| `KEDRO_CONTAINER_SCRIPTS_DIR` | `$PWD/scripts` | host dir bound onto `/app/scripts` |
| `KEDRO_CONTAINER_MOUNT_SRC` | `1` | `1` binds host `src/` onto `/app/src` (live code); `0` runs the code baked into the image |
| `KEDRO_CONTAINER_CACHE_ROOT` | `${SCRATCH:-${TMPDIR:-/tmp}}/<pkg>-cache` | host dir bound onto `/cache` (JIT caches, `$HOME`, font cache, …). `$SCRATCH` so the caches persist between jobs; `$TMPDIR`/`/tmp` only as a fallback for a machine with no `$SCRATCH`, i.e. Docker on a Mac |
| `KEDRO_CONTAINER_THREADS` | `1` | overrides the image's default single-threaded BLAS/OpenMP/numba settings; raise to `$SLURM_CPUS_PER_TASK` when a run should use more than one core |
| `KEDRO_CONTAINER_EXTRA_ARGS` | *(empty)* | extra flags passed straight to `docker run`, e.g. port publishing for `kedro viz` |
| `KEDRO_CONTAINER_INSIDE` | *(unset)* | set to `1` by the wrapper itself once inside the container; short-circuits recursion so a script can call the wrapper unconditionally whether it is already inside or not |

## Caveats worth reading before debugging

- **No C++ compiler is installed in the image** (`build-essential` is not among the apt
  packages). Nothing this project depends on compiles at run time — the geospatial stack
  ships wheels with a bundled GDAL, and numba emits machine code through LLVM rather than a
  C toolchain — but a dependency added later that falls back to run-time compilation would
  need one.
- **A `KEDRO_CONTAINER_MOUNT_SRC=1` run is not reproducible from the image tag** — see
  above. Use `MOUNT_SRC=0` for anything you plan to cite.
- **`pytest` needs `--no-cov -p no:cacheprovider`** unless `/app` itself is bound writable,
  since the project's configured `addopts` write `.coverage` and `.pytest_cache` into the
  current directory.
- **Apptainer's `--contain` gives an empty, RAM-backed `/tmp`** — measured at 64 MB of
  `tmpfs` on Sherlock. This is why every writable cache in the image is redirected to
  `/cache` instead — a cache under `/tmp` would be cold on every single run and would count
  against the job's memory limit.
- **The image's `ENV HOME=/cache/home` does not survive into Apptainer, and cannot be
  restored with `--env`.** Apptainer compiles a Docker `ENV` down to
  `export VAR="${VAR:-default}"`, so the image's value applies only when the variable is
  otherwise unset — and Apptainer *always* pre-sets `HOME` to the host's home path, so the
  image default never wins. Under `--contain` that path is the same 64 MB `tmpfs` as above,
  which silently defeats the whole `/cache` design for anything writing to bare `$HOME`.
  `--env HOME=…` is rejected outright (`Overriding HOME environment variable with
  APPTAINERENV_HOME is not permitted`); `--home <host>:<dest>`, which both binds and sets
  `HOME`, is the only supported route. The wrapper passes it. Verify with
  `./scripts/setup/run-in-container.sh sh -c 'df -h $HOME'` — a `tmpfs` line means it
  regressed.
- **`kedro-viz`, if installed (it is, in the `dev` group), writes a `.viz/` session-stats
  directory directly under the current directory on every `kedro run`.** The image
  pre-creates `/app/.viz` world-writable, which is sufficient under Docker (writable
  container layer) but **not** under Apptainer: a SIF is an immutable squashfs, so the write
  fails on the read-only filesystem no matter what its mode is —
  `WARNING: Failed writing events: [Errno 30] Read-only file system: '/app/.viz/…'`, twice
  per run. Non-fatal — the pipeline still completes — but it is noise in every array task's
  log, so the wrapper binds `$CACHE_ROOT/viz` onto `/app/.viz` on the Apptainer path.
- **`getpass.getuser()` raises outright when a foreign/mapped uid has no `/etc/passwd`
  entry**, rather than falling back gracefully — it only succeeds without one if
  `LOGNAME`/`USER`/`LNAME`/`USERNAME` is already set. Kedro guards its one internal call to
  this, but **confirmed** it still breaks pytest's own built-in `tmp_path` fixture (which
  names its base temp directory from `getuser()`, unguarded), failing every test that uses
  `tmp_path` with `OSError: No username set in the environment`. The image sets `USER` to
  cover this — generic to any foreign-uid run, not specific to this project's test suite.
- **A stray `src/*.egg-info/` on the host will reappear** any time `uv sync` or `uv run` runs
  locally against a setuptools-backend project — it is a normal side effect of an editable
  install, not a one-time mistake. With `KEDRO_CONTAINER_MOUNT_SRC=1`, that directory rides
  the bind mount into `/app/src`, and `importlib.metadata` then reports the package as *two*
  distributions (the egg-info and the image's real `dist-info`), which makes entry-point-based
  registration (e.g. Kedro hooks) order-dependent. The wrapper removes it from the host before
  every bind-mounted run, so this should never surface — but if `importlib.metadata` ever
  reports more than one distribution for this package, check for it first.

## Verification checklist

Run in order; each step isolates a different failure class.

```bash
IMAGE=$(./scripts/setup/run-in-container.sh --print-image)
PKG=$(./scripts/setup/run-in-container.sh --print-package)

# 1. build succeeds, and the layer split actually works
docker build -t "$IMAGE" --build-arg GIT_SHA="$(git rev-parse HEAD)" .
touch src/*/some_module.py && time docker build -t "$IMAGE" .
#    -> expect seconds, and "CACHED" on the `uv sync --no-install-project` step

# 2. the image alone is sane, with no mounts at all. The imports worth naming are the
#    ones with compiled extensions or a bundled GDAL -- a pure-Python dependency does
#    not fail differently inside a container than outside one.
docker run --rm "$IMAGE" python -c \
  "import geopandas, rasterio, pyogrio, exactextract, osmium, dask_geopandas, \
          numba, openturns, skfda, h3, pandas; print('stack ok')"
docker run --rm "$IMAGE" kedro registry list          # uses the baked conf/base

# 3. the Apptainer uid model: works as a non-root, passwd-less user?
docker run --rm -u 65534:65534 "$IMAGE" python -c "import sys; print('foreign uid ok')"

# 4. duplicate-metadata check -- must print exactly ONE path
./scripts/setup/run-in-container.sh python -c \
  "import importlib.metadata as m, os; \
   print([d._path for d in m.distributions() if d.metadata['Name']==os.environ.get('PKG','$PKG')])"

# 5. live source really is live
./scripts/setup/run-in-container.sh python -c "import $PKG; print($PKG.__file__)"
#    -> must print /app/src/<pkg>/__init__.py

# 6. a real pipeline end to end, writing to the bound conf/ and scripts/ dirs.
#    build_scenarios is the cheapest end-to-end check: its inputs are parameters and
#    conf/base/catalog.yml, so it needs no staged data and finishes in well under a
#    second. The env must name a directory that exists under conf/scenario_builders/.
./scripts/setup/run-in-container.sh kedro run --pipeline=build_scenarios --env=scenario_builders/boot_scale
./scripts/setup/run-in-container.sh pytest -m "not slow" --no-cov -p no:cacheprovider

# 7. on the cluster, after apptainer build
apptainer --version                                   # --contain semantics need >= 1.1
apptainer inspect "$PKG.sif" | grep revision          # label == the sha pushed
apptainer exec "$PKG.sif" readlink -f /opt/venv/bin/python   # venv symlink survived SIF flattening

# 7a. the single most important cluster check: does --cleanenv preserve the IMAGE's ENV?
apptainer exec --contain --cleanenv "$PKG.sif" sh -c \
  'echo HOME=$HOME; echo PATH=$PATH; command -v kedro'
#    If PATH is not preserved and bare `kedro` is not found, the wrapper must invoke
#    /opt/venv/bin/kedro by absolute path instead.

# 7b. confirm the /cache design was necessary and works
apptainer exec --contain --cleanenv "$PKG.sif" sh -c 'ls /tmp; df -h /tmp | tail -1'
./scripts/setup/run-in-container.sh python -c \
  "import matplotlib.pyplot, numba; print('cache writes ok')"

# 7c. host-state leakage really is closed (no /home or ~/.local entry expected)
apptainer exec --contain --cleanenv "$PKG.sif" python -c "import sys; print(sys.path)"

# 7d. one array task before sixty
KEDRO_CONTAINER_MOUNT_SRC=0 sbatch --array=0-0 scripts/scenarios/test.sh
```
