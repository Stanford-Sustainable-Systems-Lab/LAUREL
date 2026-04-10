"""Kedro pipeline nodes for the ``build_scenarios`` pipeline (scenario configuration generation).

Generates the per-scenario configuration files (YAML partitions) that drive
HPC batch execution of the States-of-the-World (SoW) array on the
Stanford Sherlock cluster.  Each generated partition encodes the full
parameter override set for one scenario run of the ``electrify_trips`` and
``evaluate_impacts`` pipelines.

Pipeline overview
-----------------
1. **generate_scenario_configs** — Discovers the appropriate
   ``ScenarioBuilder`` subclass by name, instantiates it, and returns a dict
   of YAML partition callables—one per scenario—that Kedro writes to
   ``conf/scenarios/``.

Key design decisions
--------------------
- **Registry-based dispatch**: :func:`generate_scenario_configs` is imported
  from :mod:`laurel.scenario_framework.nodes` and uses
  :attr:`~laurel.scenario_framework.build.ScenarioBuilder._registry`, which is
  populated automatically when builder modules are imported.  The
  ``import laurel.scenario_builders`` line below triggers registration of all
  bundled builders as a side effect; no manual registry update is required
  when a new builder is added to that package.
- **Separation of config generation from execution**: the ``build_scenarios``
  pipeline only produces configuration files and a shell script; actual
  scenario execution is handled by scripts in ``scripts/`` that read those
  files.

References
----------
Passow, F., & Rajagopal, R. (2026). Identifying indicators to inform proactive
substation upgrades for charging electric heavy-duty trucks. *Applied Energy*.
"""

import laurel.scenario_builders  # noqa: F401 — registers all bundled builders
from laurel.scenario_framework.nodes import generate_scenario_configs  # noqa: F401
