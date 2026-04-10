# conf/scenarios/

> **This directory is generated. Do not hand-edit its contents.**

Subdirectories here are created by the `build_scenarios` Kedro pipeline and contain per-task parameter overrides used when running individual scenario tasks:

```
kedro run --pipeline=build_scenarios --env=build_scenarios/<name>
```

Each subdirectory maps to one SLURM array task:

```
conf/scenarios/<scenario_name>/task_<N>/parameters.yml
```

The corresponding hand-written build specifications (which builder class to use, what parameter ranges to sweep) live in [`conf/build_scenarios/`](../build_scenarios/).

To regenerate configs for a scenario family, run:

```bash
kedro run --pipeline=build_scenarios --env=build_scenarios/<name>
```

The matching SLURM batch script is written to `scripts/<name>.sh`.
