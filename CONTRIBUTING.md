# Contributing to LAUREL

Thank you for your interest in LAUREL. This guide covers everything you need to set up a development environment, run tests, and submit changes.

---

## Development setup

```bash
git clone <repo-url>
cd LAUREL

# Install all dependencies including dev extras
uv sync

# Install pre-commit hooks (runs ruff automatically on each commit)
pre-commit install
```

---

## Running tests

```bash
pytest                  # run all tests
pytest --cov=laurel     # with coverage report
```

Most pipeline integration tests require intermediate datasets that are not distributed with this repository. Tests that need external data are skipped automatically when the data is absent. The `validate` scenario (`conf/scenarios/validate/`) provides a minimal, hand-built configuration that can be used to exercise the full pipeline without running all 512 SoWs.

---

## Linting and formatting

```bash
ruff check .                  # lint
ruff format .                 # format
pre-commit run --all-files    # run all hooks across the entire repo
```

The pre-commit configuration runs `ruff` automatically on staged files at commit time. The CI workflow for documentation also runs on push to `main`.

---

## Building documentation

```bash
cd docs
make html
open build/html/index.html
```

Documentation is built with Sphinx and autodoc. Docstrings in `src/laurel/` are pulled in automatically. The GitHub Actions workflow in `.github/workflows/docs.yml` deploys the built docs to GitHub Pages on every push to `main`.

---

## Docstring standard

All public functions and non-trivial private helpers use **Google Style** docstrings (rendered via the Sphinx `napoleon` extension).

Every `nodes.py` file follows a two-level standard:

**Module docstring** — four required sections:

1. One-sentence summary (pipeline name and paper module)
2. **Pipeline overview** — numbered list of all nodes in execution order, one line each
3. **Key design decisions** — prose explaining non-obvious algorithmic choices (the *why*, not just the *what*)
4. **References** — cite the paper and any external works

**Function docstring** — Google Style with:

- Summary line (active voice, one sentence)
- Extended description explaining the algorithm and why it was chosen
- `Args` — one entry per parameter; for `params: dict`, list every expected key as a sub-bullet
- `Returns` — describe what is returned, including columns added/removed for `DataFrame`/`DwellSet` returns
- `Raises` — if applicable

See any existing `nodes.py` (e.g., [src/laurel/pipelines/evaluate_impacts/nodes.py](src/laurel/pipelines/evaluate_impacts/nodes.py)) for a concrete example.

---

## Bringing your own data

LAUREL requires two proprietary datasets that cannot be redistributed:

- **Telematics** (`trips_raw`): contact International, Inc. for access, or supply a comparable dataset. Required columns: `vehicle_id`, `veh_type`, `vin_gvw`, `start_timestamp_utc`, `end_timestamp_utc`, `starting_h3_8`, `ending_h3_8`, `trip_miles`, `trip_hrs`.
- **Business establishments**: contact Data Axle, Inc. for access, or supply a comparable dataset. Required schema: three tables joined on `ESTAB_ID` — a core table (`COMPANY`, `PRIMARY_NAICS_CODE`, `EMPLOYEE_SIZE__5____LOCATION`, `BUSINESS_STATUS_CODE`, `STATE`), a geo table (`LATITUDE`, `LONGITUDE`), and a relationships table (`PARENT_NUMBER`).

See the [Input Data](README.md#input-data) section of README.md for the full list of required datasets and their sources.
