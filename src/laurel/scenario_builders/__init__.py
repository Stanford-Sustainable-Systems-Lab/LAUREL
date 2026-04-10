"""Concrete ScenarioBuilder subclasses for LAUREL scenario generation.

Importing this package registers all bundled builders in
:attr:`~laurel.scenario_framework.build.ScenarioBuilder._registry`, making them
available to :func:`~laurel.scenario_framework.nodes.generate_scenario_configs`
without any further configuration.
"""

from .batt_dual_pow import BatteryDualPowerScenarioBuilder  # noqa: F401
from .batt_man import BatteryManageScenarioBuilder  # noqa: F401
from .batt_pow import BatteryPowerScenarioBuilder  # noqa: F401
from .ca_eight import CalifClass8ScenarioBuilder  # noqa: F401
from .ca_eight_adopt import CalifClass8AdoptionScenarioBuilder  # noqa: F401
from .ranger import RangeScenarioBuilder  # noqa: F401
from .scale_test import ScaleTestScenarioBuilder  # noqa: F401
from .sense import SenseScenarioBuilder  # noqa: F401
