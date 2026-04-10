# LAUREL: Location-Attuned Uncertainty-Robust Electric-vehicle Load-simulator

This repository implements the LAUREL model described in:

> Passow, F. H., & Rajagopal, R. (2026). Identifying indicators to inform proactive substation upgrades for charging electric heavy-duty trucks. *Applied Energy* (submitted March 2026).

The specific use of the LAUREL model demonstrated here estimates e-HDT charging load profiles for each of the ~52,000 electrical substations in the continental U.S. across 512 plausible future states of the world (SoWs) representing 2035 conditions. It is used to identify which substations grid operators should consider proactively upgrading for e-HDT charging, and what techno-economic indicators signal when such upgrades may become necessary.

---

## Table of Contents

- [Background](#background)
- [Repository Structure](#repository-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Input Data](#input-data)
- [Model Overview](#model-overview)
- [Running the Model](#running-the-model)
- [Scenarios](#scenarios)
- [Configuration](#configuration)
- [HPC Execution (Sherlock)](#hpc-execution-sherlock)
- [Output Data](#output-data)
- [Development](#development)
- [Citation](#citation)

---

## Background

Electric heavy-duty trucks (e-HDTs) will require geographically concentrated, high-power charging that will stress electric distribution infrastructure — particularly substations, which can take years and millions of dollars to upgrade. This model was built to answer: *which substations should grid operators proactively upgrade, and what observable conditions should trigger that decision?*

Our approach uses a continent-scale telematics dataset of ~69,000 diesel HDTs (International, Inc., April–November 2023) as the behavioral foundation. We simulate electrified versions of these vehicles across 512 quasi-random combinations of key uncertain parameters (adoption rates, energy consumption, charger power, battery reserve). For each combination, we assemble 30-minute load profiles for every substation in the continental U.S.

Key findings:

- Under a "duty-to-serve-robust" policy (80th-percentile peak load), 7.4% of substations would exceed 3 MW from e-HDT charging alone by 2035.
- Presence of a truck stop is the strongest geographic predictor; almost no substations exceed 3 MW at adoption rates below 5%.
- Adoption level is by far the strongest techno-economic indicator, followed distantly by truck-stop charger power.

---

## Repository Structure

```text
LAUREL/
├── conf/                      # Kedro configuration
│   ├── base/                  # Shared parameters and data catalog
│   │   ├── catalog.yml        # ~860 dataset definitions
│   │   └── parameters_*.yml   # One parameter file per pipeline
│   ├── build_scenarios/       # Hand-written build specs (input to build_scenarios pipeline)
│   └── scenarios/             # Generated per-task parameter overrides (gitignored)
├── data/                      # Data layers (Kedro convention; not committed)
│   ├── 01_raw/                # External source datasets
│   ├── 02_intermediate/       # Processed/formatted datasets
│   ├── 07_model_output/       # Per-scenario charging results
│   └── 08_reporting/          # Final visualizations and summaries
├── docs/                      # Sphinx documentation source
├── notebooks/                 # Exploratory Jupyter notebooks
├── scripts/                   # Generated SLURM batch scripts (gitignored)
├── src/
│   ├── laurel/              # Main Python package
│   │   ├── datasets/          # Custom Kedro dataset classes (geospatial formats)
│   │   ├── models/            # Core algorithms (charging, dwell sets, sampling)
│   │   ├── pipelines/         # Nine Kedro pipelines (one per model module)
│   │   ├── routing/           # GraphHopper routing client and server management
│   │   ├── scenario_builders/ # Concrete ScenarioBuilder subclasses (one per scenario family)
│   │   ├── scenario_framework/# Abstract base classes, shell-script generator, I/O helpers
│   │   └── utils/             # Shared utilities (geo, H3, NAICS, time, ...)
├── tests/                     # pytest test suite
├── pyproject.toml             # Project metadata and dependencies
└── uv.lock                    # Locked dependency versions
```

---

## Prerequisites

| Requirement | Notes |
| ----------- | ----- |
| Python ≥ 3.12 | Tested on 3.12 |
| [uv](https://docs.astral.sh/uv/) | Preferred package manager |
| Docker | Required only for the `compute_routes` pipeline (GraphHopper routing engine) |
| ~200 GB disk space | For raw inputs + intermediate outputs for all 512 SoWs |
| 64 GB RAM | Minimum for running a single SoW through `electrify_trips` / `evaluate_impacts` |
| 128 GB RAM | Recommended for `compute_routes` |

For large-scale runs across all 512 SoWs, an HPC cluster is strongly recommended (see [HPC Execution](#hpc-execution-sherlock)).

---

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd LAUREL

# Install dependencies with uv (creates .venv automatically)
uv sync

# Or with pip into an existing environment
pip install -e .
```

To verify the installation:

```bash
kedro info
kedro pipeline list
```

You should see the nine pipelines: `preprocess`, `describe_vehicles`, `describe_dwells`, `compute_routes`, `describe_locations`, `prepare_totals`, `electrify_trips`, `evaluate_impacts`, `build_scenarios`.

---

## Input Data

The model requires several external datasets, placed under `data/01_raw/`. The data catalog (`conf/base/catalog.yml`) defines where each dataset is expected.

| Dataset | Source | Pipeline(s) |
| --- | --- | --- |
| International, Inc. telematics | Proprietary (contact International, Inc.) | `describe_dwells`, `describe_vehicles`, `compute_routes`, `electrify_trips` |
| VIUS (Vehicle Inventory and Use Survey) | [BTS](https://www.bts.gov/vius) | `preprocess` |
| NREL Ledna adoption scenarios | [iScience 27 (2024) 109385](https://doi.org/10.1016/j.isci.2024.109385) | `prepare_totals` |
| HIFLD Electrical Substations | [gem.anl.gov](https://gem.anl.gov) | `evaluate_impacts` |
| PG&E ICA maps | [grip.pge.com](https://grip.pge.com) | `evaluate_impacts` (PG&E territory only) |
| NLCD 2023 (National Land Cover Database) | [USGS](https://doi.org/10.5066/P94UXNTS) | `describe_locations` |
| Data Axle business establishments | Proprietary (contact Data Axle, Inc.) | `describe_locations` |
| Jason's Law truck parking | [BTS geodata](https://geodata.bts.gov/datasets/fff36e0c37c748a5a1773b5784d4d9a5_0) | `describe_locations`, `compute_routes` |
| OpenStreetMap (continental U.S.) | [Geofabrik](https://download.geofabrik.de) | `compute_routes`, `describe_locations` |

> **Note on telematics data:** The International, Inc. dataset is proprietary and cannot be redistributed. Researchers wishing to replicate this work should contact International, Inc. to request access, or adapt the pipeline to use a comparable telematics dataset with the same schema (vehicle ID, dwell TAZ, dwell start/end times, trip distance).

> **Note on business establishments data:** The Data Axle, Inc. dataset is proprietary and cannot be redistributed. Researchers wishing to replicate this work should contact Data Axle, Inc. to request access, or adapt the pipeline to use a comparable business establishment dataset with the same schema. The pipeline expects three tables joined on a shared establishment ID: a core table (`ESTAB_ID`, `COMPANY`, `PRIMARY_NAICS_CODE`, `EMPLOYEE_SIZE__5____LOCATION`, `BUSINESS_STATUS_CODE`, `STATE`), a geo table (`ESTAB_ID`, `LATITUDE`, `LONGITUDE`), and a relationships table (`ESTAB_ID`, `PARENT_NUMBER`).

---

## Model Overview

The model has six modules that map to Kedro pipelines:

```text
Module 1: Select SoWs         ← prepare_totals
Module 2: Augment dwell data  ← describe_dwells + compute_routes
Module 3: Augment TAZs        ← describe_locations
                                 (also: preprocess, describe_vehicles)
─── repeat for each SoW ───
Module 4: Simulate e-HDT dwells  ← electrify_trips
Module 5: Estimate expected dwells by TAZ  ← evaluate_impacts (first half)
Module 6: Assemble load profiles  ← evaluate_impacts (second half)
```

### Module 1 — Select States of the World (`prepare_totals`)

Generates 512 quasi-random SoWs using Sobol' sequences (via OpenTURNS). Adoption rates by vehicle primary operating distance class are drawn from Beta distributions fit to NREL scenarios via a Gaussian copula. Other parameters (energy consumption rate, charger power at truck stops / depots / destinations, battery reserve) are sampled uniformly.

### Module 2 — Augment Dwell Data (`describe_dwells`, `compute_routes`)

Coalesces spurious short dwells, marks driver shifts (≥6.9 hr breaks per FMCSA rules), and inserts optional dwells at truck stops along shortest-path routes computed by GraphHopper. Optional dwells are inserted between existing dwells separated by >50 miles, if a truck stop falls within 1 mile of the shortest path. This grew our dwell count by ~35%.

### Module 3 — Augment TAZs (`describe_locations`)

Classifies each H3 resolution-8 hexagon (~1/4-mile diameter) in the continental U.S. into one of 22 freight activity classes:

- Undeveloped / No establishments / No freight-intensive establishments / Truck stops
- 18 K-Means clusters of freight-intensive TAZs (based on NAICS employee counts)

Deploys chargers by freight activity class: truck-stop charging at truck-stop TAZs, destination charging at freight-intensive TAZs, depot charging per-vehicle based on a 30-day rolling dwell-time threshold.

### Module 4 — Simulate Electrified Dwells (`electrify_trips`)

Runs a utility-maximization charging choice algorithm (inspired by Liu et al. 2022) for each vehicle's full dwell history. The algorithm selects charging mode and energy amount at each dwell, trading off SoC maintenance against incurred delay, with look-ahead to the end of the current driver shift. Uses Numba JIT compilation for performance. Each SoW runs in ~25 minutes on a 4-core/64 GB machine.

### Module 5 — Estimate Expected Electrified Dwells (`evaluate_impacts`)

Estimates the expected number of electrified dwells per TAZ on a typical weekday by fusing SoW adoption rates (from Module 1) with freight-activity-class-specific vehicle visit statistics from the observed data. Uses logistic regression with a numeric correction term to ensure consistency with known fleet-level adoption rates.

### Module 6 — Assemble Load Profiles (`evaluate_impacts`)

Bootstrap-samples electrified dwells (100 draws, 95th percentile) to assemble 30-minute load profiles for each TAZ, then aggregates across all TAZs within each substation territory. Uses inverse propensity score weighting to correct for sampling bias in the telematics dataset.

---

## Running the Model

### Full pipeline (all modules, default scenario)

```bash
kedro run
```

### Individual pipelines

```bash
# Data preparation (run once)
kedro run --pipeline=preprocess
kedro run --pipeline=describe_vehicles
kedro run --pipeline=describe_dwells
kedro run --pipeline=compute_routes        # Requires Docker (GraphHopper)
kedro run --pipeline=describe_locations

# SoW generation (run once)
kedro run --pipeline=prepare_totals

# Per-SoW simulation (run once per scenario)
kedro run --pipeline=electrify_trips
kedro run --pipeline=evaluate_impacts
```

### Running a specific scenario

```bash
kedro run --pipeline=electrify_trips --params="scenario:sense_512"
kedro run --pipeline=evaluate_impacts --params="scenario:sense_512"
```

### Running a single SoW from the 512-SoW set

The `sense_512` scenario set is designed to run one SoW at a time, identified by a `task_id` parameter. This is how the SLURM array jobs work:

```bash
kedro run --pipeline=electrify_trips --params="scenario:sense_512,task_id:42"
kedro run --pipeline=evaluate_impacts --params="scenario:sense_512,task_id:42"
```

### Interactive development

```bash
kedro jupyter lab    # JupyterLab with full Kedro context
kedro jupyter notebook
kedro ipython        # IPython REPL with Kedro context
```

---

## Scenarios

Scenario definitions live in `conf/scenarios/`. Each scenario directory contains YAML files that override base parameters.

| Scenario | Description |
| --------- | ------------- |
| `sense_512` | Main paper scenario: 512 SoWs, adoption from NREL Beta+copula |
| `validate` | Validation run matching Broga et al. (2025) assumptions |
| `test` | Fast smoke-test scenario |

### SoW parameter ranges (sense_512)

| Parameter | Range | Distribution |
| --------- | ----- | ------------ |
| Adoption, 0–99 mi class | 0–29% | Beta (median ~4%) |
| Adoption, 100–249 mi class | 0–30% | Beta (median ~3%) |
| Adoption, 250–499 mi class | 0–19% | Beta (median ~2%) |
| Adoption, 500+ mi class | 0–11% | Beta (median ~0%) |
| Energy consumption | 1.4–2.0 kWh/mile | Uniform |
| Charger power at truck stops | 350–1,500 kW | Uniform |
| Charger power at depots/destinations | 40–350 kW | Uniform |
| Battery reserve (target SoC) | 10–50% | Uniform |

Adoption rates across distance classes are correlated via a Gaussian copula fit to NREL scenarios (see Figure A.10 in the paper).

---

## Configuration

### Data catalog (`conf/base/catalog.yml`)

Defines all ~860 datasets with their file paths and formats. Uses standard Kedro layers:

- `01_raw` — source data (never modified)
- `02_intermediate` — cleaned/formatted data
- `07_model_output` — per-scenario simulation outputs
- `08_reporting` — final outputs and visualizations

Per-scenario outputs are stored under `data/07_model_output/<scenario_name>/<task_id>/` as partitioned Parquet datasets.

### Pipeline parameters

Each pipeline has a corresponding parameter file in `conf/base/`:

| File | Controls |
| ---- | --------- |
| `parameters_preprocess.yml` | VIUS scaling weights |
| `parameters_describe_dwells.yml` | Dwell coalescing thresholds, shift detection |
| `parameters_describe_vehicles.yml` | Vehicle classification, depot detection window |
| `parameters_compute_routes.yml` | GraphHopper server config, optional dwell radius |
| `parameters_describe_locations.yml` | K-Means cluster count, NLCD thresholds, NAICS codes |
| `parameters_prepare_totals.yml` | SoW count, Sobol' seed, Beta distribution parameters |
| `parameters_electrify_trips.yml` | Charging algorithm weights, delay caps |
| `parameters_evaluate_impacts.yml` | Bootstrap count, percentile, electrifiability criteria |
| `parameters_build_scenarios.yml` | SLURM configuration for HPC job arrays |

### GraphHopper routing engine

The `compute_routes` pipeline uses GraphHopper via Docker. Before running:

```bash
# Pull the GraphHopper Docker image
docker pull graphhopper/graphhopper

# The pipeline manages container startup/shutdown automatically
kedro run --pipeline=compute_routes
```

The OSM road network file for the continental U.S. must be placed at the path specified in `conf/base/parameters_graphhopper.yml`.

---

## HPC Execution (Sherlock)

The full 512-SoW run was computed on the [Sherlock cluster](https://www.sherlock.stanford.edu/) at Stanford University. Each SoW takes ~25 minutes on 4 cores / 64 GB RAM.

### Generating SLURM scripts

```bash
kedro run --pipeline=build_scenarios --env=build_scenarios/sense_512
```

This writes SLURM batch scripts to `scripts/` and per-task config files to `conf/scenarios/`. Each array index corresponds to one SoW.

### Submitting jobs

```bash
sbatch scripts/sense_512.sh
```

### GraphHopper on Sherlock (Apptainer)

HPC environments like Sherlock do not support Docker; use [Apptainer](https://apptainer.org/) instead. Loading the GraphHopper container requires a few manual steps:

1. **Pull the container in sandbox mode:**

   ```bash
   apptainer pull --sandbox docker://israelhikingmap/graphhopper:10.2
   ```

2. **Edit the container's runscript** to add `cd /graphhopper` at the very beginning. This works around Apptainer's lack of support for Docker's `WORKDIR` directive.

3. **Edit `/graphhopper/graphhopper.sh`** (inside the container) to restrict the `.jar` file search to the `/graphhopper` directory. The relevant line is near the bottom of the file where the `JAR` environment variable is set.

Once patched, point `conf/base/parameters_compute_routes.yml` at the Apptainer sandbox path instead of a Docker image name.

---

## Output Data

After running `evaluate_impacts` for all 512 SoWs, the outputs are organized as:

```text
data/07_model_output/sense_512/
└── <task_id>/
    ├── dwells_with_charging_partition/   # Per-vehicle charging decisions
    ├── events_partition/                 # Charging events (power, time)
    ├── vehicles_with_params_partition/   # Vehicle design ranges + parameters
    └── load_profile_quantiles/           # 30-min load profiles by substation
```

The final reporting outputs (maps, policy analysis, validation figures) are in `data/08_reporting/`.

### Cross-SoW aggregation

To compute the 80th/20th percentile ("duty-to-serve-robust" / "used-and-useful-robust") peak loads across all SoWs, aggregate the `load_profile_quantiles` datasets across task IDs. Example notebooks for this analysis are in `notebooks/`.

---

## Development

### Running tests

```bash
pytest
pytest --cov=laurel   # with coverage report
```

### Linting and formatting

```bash
ruff check .      # lint
ruff format .     # format
pre-commit run --all-files  # all hooks
```

### Building documentation

```bash
cd docs
make html
open build/html/index.html
```

## Citation

If you use this model or code in your research, please cite:

```bibtex
@article{passow2026laurel,
  title   = {Identifying indicators to inform proactive substation upgrades
             for charging electric heavy-duty trucks},
  author  = {Passow, Fletcher H. and Rajagopal, Ram},
  journal = {Applied Energy},
  year    = {2026},
  note    = {Submitted March 2026}
}
```

---

## Acknowledgments

The authors thank International, Inc. (especially Srinivas Gowda and Tobias Glitterstam) for sharing vehicle behavior data. This work was supported by the Bits & Watts Initiative at the Stanford Precourt Institute for Energy.

Computational resources were provided by the Sherlock cluster at Stanford University.
