# AI Coding Agent Instructions for laurel

Concise, project-specific guidance so an AI agent can contribute productively within minutes.

## 1. Project Essence
laurel is a Kedro 1.x data & simulation project modeling heavy-vehicle trip electrification. Core domain object: `DwellSet` (see `src/laurel/models/dwell_sets.py`) which represents sequential vehicle dwells/trips and supports masked accumulation, forward/backward propagation, and reset-aware transforms (Numba + Dask compatible). Pipelines (directory: `src/laurel/pipelines/*`) orchestrate transformations configured via YAML in `conf/base/*.yml`.

## 2. Architecture & Data Flow
- Entry point: `kedro run` assembles pipelines via `pipeline_registry.register_pipelines()` (auto-discovers each `pipeline.py`).
- Typical flow for electrification (`electrify_trips`): load trips → build `DwellSet` → enrich with vehicle/mode parameters → compute dwell durations & energy → mark refresh / critical days → filter dwells (masked accumulation) → compute shift power envelopes → simulate charging decisions → apply charging management.
- Configuration layering: `conf/base/parameters_*.yml` (shared), `conf/local/` (ignored, developer overrides), scenario definitions in `conf/scenarios/` & scenario runners in `conf/scenario_runners/` (used for batch experimentation).
- Data folders follow Kedro convention (`data/01_raw` → `08_reporting`). Never commit large outputs.

## 3. Key Conventions
- Always prefer operating on `DwellSet` methods (`accum_masked`, `reset_masked`, `sort_by_veh_time`) rather than re-implementing cumulative logic.
- `DwellSet` assumes vehicle ID is the index for ordering-sensitive ops; if modifying columns that affect grouping, re-run `sort_by_veh_time()`.
- For Dask: functions must be partition-safe (avoid relying on global ordering unless `DwellSet.sort_by_veh_time()` already enforced). Use `.map_partitions` with `meta` for schema stability.
- Parameters are passed into nodes under `params:` in pipeline catalog; new node parameters must be added to the appropriate `parameters_<pipeline>.yml` with stable, descriptive names.
- Column naming: time spans generally suffixed with `_hrs`, energy with `_kwh`, boolean flags `is_*` or `has_*`, availability arrays `_bool`.
- Masking pattern: create a boolean keep column → call `accum_masked` (optionally forward + reverse accumulation) → transform `_col_keep` suffixed columns back to original names → drop mask.

## 4. Performance & Scaling
- Use NumPy / vectorized Pandas first; rely on Numba-decorated cores (e.g. `_accum_masked_core`) for tight loops—do not introduce Python loops over rows.
- When adding partition operations, supply `meta=dd.utils.make_meta(df)` to avoid Dask graph expansion errors.
- Maintain `threads_per_worker: 1` per config (`parameters_electrify_trips.yml`) because Numba & some numpy sections are not thread-safe.

## 5. Charging Simulation Pattern
- Strategy object: `ForwardLookingChargingChoiceStrategy` consumes column names via `params['simulate_charging_choice']['input_cols']`.
- Precompilation option only used when `precompile: True`;
- After simulation, mode selection merging uses `merge_chosen_mode` then charging power application via a manager (e.g. `MinPowerChargingManager`). New managers should accept `energy`, `duration`, `max_power` kwargs and return columns consistent with existing tests.

## 6. Adding a New Pipeline Node (Example)
1. Implement pure function in `nodes.py` operating on `DwellSet` or DataFrame.
2. Accept `(dw: DwellSet, params: dict)` if it mutates dwell state; return updated `DwellSet`.
3. Register in `pipeline.py` with `node(func, inputs=["dwell_set", "params:my_step"], outputs="dwell_set")`.
4. Add `my_step:` block to `conf/base/parameters_<pipeline>.yml`.
5. If new columns created, document in that YAML and (if reused) add to downstream parameter mappings.

## 7. Testing & Quality
- Tests live under `tests/`; prefer fast unit tests for new accumulation or simulation logic (construct tiny in-memory DataFrames).
- Run: `pytest -q` (coverage governed by `.coveragerc`).
- Lint: Ruff configured in `pyproject.toml` (rules: F,E,W,UP,I,PL; long lines ignored). Keep imports sorted; run Ruff before committing if possible.

## 8. Developer Workflows
- Install deps (project uses uv-managed pyproject): `uv sync` then extras from `[tool.uv].dev-dependencies` for docs/perf tooling.
- Visualize pipelines: `kedro viz` (experiment tracking supported via kedro-viz—log params/metrics when expanding simulation components).
- Notebooks: use `kedro jupyter lab` so context objects (`catalog`, `session`) are injected; avoid committing executed outputs (nbstripout hook recommended—already a dev dependency).

## 9. Safe Extension Guidelines
- Reuse existing parameter names where semantic overlap exists (e.g., new energy columns should extend pattern `*_kwh`).
- When introducing new accumulation semantics, extend `CumAggFunc` only if necessary—prefer composing existing functions.
- Keep Dask vs Pandas parity: add branches similar to existing nodes (`if dw.is_dask:`) and ensure deterministic column ordering.

## 10. Gotchas
- Some reverse accumulations require matching `reverse` list length to `accum_cols`; validate lengths to avoid silent misalignment.
- `simulate_charging_choice` assumes prior sorting unless `dw.is_dask` and partitions already respect ordering; forcing sort on very large Dask frames can be costly.
- Boolean columns may be converted to unsigned bytes in record arrays (`_replace_dtypes`)—do not assume `bool` dtype when working inside Numba cores.
- Critical day logic depends on both `is_refresh` and battery capacity; altering threshold params without updating tests may mask infeasible days.

## 11. Where to Look First
- Domain logic: `src/laurel/models/dwell_sets.py`
- Charging strategy: `src/laurel/models/charging_algorithms.py`
- Pipeline example: `src/laurel/pipelines/electrify_trips/nodes.py`
- Parameters: `conf/base/parameters_electrify_trips.yml`
- Registry: `src/laurel/pipeline_registry.py`

## 12. When Unsure
Prefer inspecting existing analogous nodes/pipelines and mirroring their signature & parameterization before inventing new patterns.

(End of instructions — provide feedback on any unclear section to refine.)
