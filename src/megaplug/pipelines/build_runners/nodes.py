"""Kedro pipeline nodes for the ``build_runners`` pipeline (scenario configuration generation).

Generates the per-scenario configuration files (YAML partitions) that drive
HPC batch execution of the 512 States-of-the-World (SoW) array on the
Stanford Sherlock cluster.  Each generated partition encodes the full
parameter override set for one scenario run of the ``electrify_trips`` and
``evaluate_impacts`` pipelines.

Pipeline overview
-----------------
1. **generate_scenario_configs** — Discovers the appropriate
   ``ScenarioBuilder`` subclass by name, instantiates it, and returns a dict
   of YAML partition callables—one per scenario—that Kedro writes to
   ``conf/scenario_runners/``.

Key design decisions
--------------------
- **Dynamic dispatch via globals()**: Rather than a hard-coded ``if/elif``
  chain, the node inspects the module's global namespace at runtime to find
  all ``ScenarioBuilder`` subclasses.  New builders are automatically
  available once they are imported at the top of the module; no registry
  update is required.
- **Separation of config generation from execution**: The ``build_runners``
  pipeline only produces configuration files; actual scenario execution is
  handled by shell scripts in ``src/runners/`` that read those files.

References
----------
Passow, F., & Rajagopal, R. (2026). Identifying indicators to inform proactive
substation upgrades for charging electric heavy-duty trucks. *Applied Energy*.
"""

import inspect

from megaplug.scenarios.build import (  # noqa: F401
    ScenarioBuilder,
    TestScenarioBuilder,
)

from .batt_dual_pow import BatteryDualPowerScenarioBuilder  # noqa: F401
from .batt_man import BatteryManageScenarioBuilder  # noqa: F401
from .batt_pow import BatteryPowerScenarioBuilder  # noqa: F401
from .ca_eight import CalifClass8ScenarioBuilder  # noqa: F401
from .ca_eight_adopt import CalifClass8AdoptionScenarioBuilder  # noqa: F401
from .ranger import RangeScenarioBuilder  # noqa: F401
from .scale_test import ScaleTestScenarioBuilder  # noqa: F401
from .sense import SenseScenarioBuilder  # noqa: F401


def generate_scenario_configs(scen_params: dict, all_params: dict) -> dict:
    """Call the appropriate scenario configuration builder.

    This function is meant to be called from a kedro pipeline directly.

    Args:
        scen_params: just the item in the "parameters" dictionary dedicated to giving
            parameters for scenario building (e.g. builder name)
        all_params: the whole "parameters" input dictionary, usually passed directly from
            a kedro pipeline input.

    Returns: The dictionary of scenario configuration partitions. Usually this would be
    saved to a `kedro` partitioned dataset.
    """
    bldr_map = {}
    base_cls = ScenarioBuilder
    for name, obj in globals().items():
        if inspect.isclass(obj) and issubclass(obj, base_cls) and obj is not base_cls:
            bldr_map.update({name: obj})

    bldr_name = scen_params["builder"]
    try:
        builder_cls = bldr_map[bldr_name]
    except KeyError:
        raise NotImplementedError(f"Scenario builder {bldr_name} not imported.")
    builder = builder_cls(scen_params=scen_params, all_params=all_params)
    parts = builder.build_configs()
    return (parts, builder)
