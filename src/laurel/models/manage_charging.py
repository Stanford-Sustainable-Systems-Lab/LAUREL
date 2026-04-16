"""Charging management classes that convert dwell records to load-profile events.

After the charging-choice simulation (:mod:`laurel.models.charging_algorithms`)
determines *how much* energy each vehicle charges at each dwell, the charging
managers in this module determine *when* that energy flows.  They translate
per-dwell charging assignments into a sequence of timestamped power events
that can be assembled into load profiles.

The class hierarchy is:

- :class:`AbstractChargingManager` — stores column bindings and the abstract
  :meth:`get_events` interface.
- :class:`IndependentDwellChargingManager` — handles the common case where
  each dwell is managed independently (no cross-dwell optimisation).  Defines
  the ``seq_names`` pattern and the dwell-to-event pipeline.
- :class:`MinPowerChargingManager` — spreads energy at the minimum constant
  power needed to deliver the required kWh over the full dwell duration.
- :class:`ImmediateChargingManager` — charges at maximum available power as
  early as possible, computing the exact charge-end timestamp.

Key design decisions
--------------------
- **``seq_names`` event structure**: each concrete manager defines a list of
  named event-sequence prefixes (e.g. ``["dwell_start", "dwell_end"]``).  For
  each sequence name, the manager adds columns ``{seq_name}_time``,
  ``{seq_name}_duration``, ``{seq_name}_power_kw``, and
  ``{seq_name}_plugged``.  The :meth:`~DwellSet.to_events` method then pivots
  from wide (one dwell per row) to long (one event per row) format.
- **``ProfileType`` enum**: ``OBSERVATIONS`` mode writes absolute power at each
  event boundary (suitable for step-function integration); ``DIFFERENCES`` mode
  writes the *change* in power (positive at plug-in, negative at plug-out),
  which enables efficient sparse load profile construction via cumulative sum.
- **Property renaming pattern**: the ``energy``, ``duration``, ``max_power``,
  ``region``, ``scale_up``, and ``cost`` properties delegate to
  :meth:`~DwellSet._rename_idx_col`, so assigning a new column name
  transparently renames the underlying DataFrame column.
- **``_MANAGER_MAP``**: populated at module load time by introspecting
  ``globals()`` for subclasses of :class:`AbstractChargingManager`.  Allows
  pipeline nodes to instantiate managers by name without a static registry.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from enum import IntEnum, auto
from itertools import product
from typing import Self

import dask.dataframe as dd
import numpy as np
import pandas as pd

from laurel.models.dwell_sets import DwellSet

logger = logging.getLogger(__name__)


class ProfileType(IntEnum):
    """Output encoding for power events in :class:`IndependentDwellChargingManager`.

    Members:
        OBSERVATIONS: Each event records the absolute charging power at that
            timestamp.  Suitable for step-function load profiles.
        DIFFERENCES: Each event records the *change* in charging power (positive
            at plug-in, negative at plug-out).  Enables sparse cumulative-sum
            profile assembly.
    """

    OBSERVATIONS = auto()
    DIFFERENCES = auto()


class AbstractChargingManager(ABC):
    """Abstract base class for charging management strategies.

    Stores column-name bindings for the dwell features needed to build load
    profiles and defines the :meth:`get_events` interface that concrete
    managers must implement.  Column bindings are exposed as properties that
    rename the underlying :attr:`dw` DataFrame columns transparently when
    assigned.

    Attributes:
        dw: :class:`~laurel.models.dwell_sets.DwellSet` over which this
            manager operates.
    """

    _dw: DwellSet = None
    _energy: str = None
    _duration: str = None
    _max_power: str = None
    _cost: str = None
    _scale_up: str = None
    _region: str = None
    _id_cols: list[str] = None
    _prof_type: ProfileType = ProfileType.OBSERVATIONS

    def __init__(
        self,
        dw: DwellSet,
        energy: str,
        duration: str,
        max_power: str,
        region: str = None,
        scale_up: str = None,
        cost: str = None,
        id_cols: list[str] = None,
        prof_type: ProfileType = ProfileType.OBSERVATIONS,
    ) -> None:
        """Initialise the charging manager with column bindings.

        Args:
            dw: :class:`~laurel.models.dwell_sets.DwellSet` containing
                dwell records with charging decisions already assigned.
            energy: Column name for the energy to deliver at each dwell (kWh).
            duration: Column name for the available dwell duration (hours).
            max_power: Column name for the maximum available charging power at
                each dwell (kW).
            region: Column name for the spatial aggregation grouping variable
                (e.g. substation or hex ID).  ``None`` if not used.
            scale_up: Column name for the population-scaling weight applied to
                power values before aggregation.  ``None`` if not used.
            cost: Column name for the per-kWh cost promised to the vehicle
                during the choice model.  ``None`` if not used.
            id_cols: Additional columns to carry through to the event output.
                ``None`` uses the DwellSet default.
            prof_type: :class:`ProfileType` controlling whether output events
                encode absolute power or power differences.  Defaults to
                ``ProfileType.OBSERVATIONS``.
        """
        self.dw = dw
        self.energy = energy
        self.duration = duration
        self.max_power = max_power
        self.region = region
        self.scale_up = scale_up
        self.cost = cost
        self.id_cols = id_cols
        self.prof_type = prof_type

    @abstractmethod
    def get_events(self) -> pd.DataFrame:
        """Build and return a long-format event DataFrame from the dwell records.

        Returns:
            DataFrame with one event per row, columns include at minimum a
            timestamp, a power value, a plugged indicator, and the duration of
            the event.  Exact column names depend on the concrete manager's
            ``seq_names`` and ``suffixes``.
        """
        pass

    @property
    def energy(self):
        """Name of the column within dw containing the required energy for each dwell"""
        return self._energy

    @energy.setter
    def energy(self, value):
        self.dw._rename_idx_col(value, self._energy)
        self._energy = value

    @property
    def duration(self):
        """Name of the column within dw containing the duration of the dwell"""
        return self._duration

    @duration.setter
    def duration(self, value):
        self.dw._rename_idx_col(value, self._duration)
        self._duration = value

    @property
    def max_power(self):
        """Name of the column within dw containing the maximum charging power for this dwell"""
        return self._max_power

    @max_power.setter
    def max_power(self, value):
        self.dw._rename_idx_col(value, self._max_power)
        self._max_power = value

    @property
    def region(self):
        """Name of the column containing the grouping variable for regions"""
        return self._region

    @region.setter
    def region(self, value):
        self.dw._rename_idx_col(value, self._region)
        self._region = value

    @property
    def scale_up(self):
        """Name of the column giving the factor by which to scale up the power to represent a population of vehicles"""
        return self._scale_up

    @scale_up.setter
    def scale_up(self, value):
        self.dw._rename_idx_col(value, self._scale_up)
        self._scale_up = value

    @property
    def cost(self):
        """Name of the column giving the cost which was promised during the charging choice model"""
        return self._cost

    @cost.setter
    def cost(self, value):
        self.dw._rename_idx_col(value, self._cost)
        self._cost = value

    @property
    def prof_type(self):
        """Name of the pofile type to use when setting events."""
        return self._prof_type

    @prof_type.setter
    def prof_type(self, value):
        self._prof_type = value


class IndependentDwellChargingManager(AbstractChargingManager):
    """Base class for managers that treat each dwell independently.

    Defines the four-suffix column convention (``time``, ``duration``,
    ``power_kw``, ``plugged``) and the shared :meth:`get_events` pipeline:
    add event columns via :meth:`set_dwell_events`, optionally scale power
    by ``scale_up``, then reshape via :meth:`~DwellSet.to_events`.

    Concrete subclasses implement :meth:`set_dwell_events` and declare
    :attr:`seq_names`.

    Attributes:
        suffixes: Dict mapping logical role (``"time"``, ``"duration"``,
            ``"power"``, ``"plugged"``) to the column-name suffix appended
            after each sequence name.
    """

    suffixes = {
        "time": "time",
        "duration": "duration",
        "power": "power_kw",
        "plugged": "plugged",
    }

    @property
    @abstractmethod
    def seq_names(self) -> list[str]:
        """Ordered list of event-sequence prefixes for this manager.

        For example ``["dwell_start", "dwell_end"]`` causes the manager to
        create columns ``dwell_start_time``, ``dwell_start_power_kw``, etc.
        """
        pass  # Implement this in the concrete classes by setting the attribute seq_names

    @abstractmethod
    def set_dwell_events(self) -> Self:
        """Add event-sequence columns to ``self.dw.data`` and return ``self``.

        Must create all columns defined by :attr:`seq_names` × :attr:`suffixes`
        before returning so that :meth:`get_events` can pivot them.
        """
        pass

    def get_events(self: Self) -> pd.DataFrame:
        """Execute the dwell-to-event pipeline.

        1. Calls :meth:`set_dwell_events` to add event columns.
        2. Optionally scales power columns by ``self.scale_up``.
        3. Sets :attr:`~DwellSet.seq_names` on the DwellSet.
        4. Calls :meth:`~DwellSet.to_events` to reshape to long format.

        Logs a warning if any existing columns already use the suffix convention.
        """
        checks = self.check_for_suffixes()
        if len(checks) > 0:
            logger.warning(
                f"The following columns are already using the {type(self).__name__} suffixes: {', '.join(checks)}. If this is unexpected, then please change their names."
            )
        self = self.set_dwell_events()
        if self.scale_up is not None:
            for col in [f"{seqn}_{self.suffixes['power']}" for seqn in self.seq_names]:
                self.dw.data[col] = self.dw.data[col] * self.dw.data[self.scale_up]
        # self.dw.data = self.dw.data.dropna(subset=self.dw.seq_names)
        self.dw.seq_names = self.seq_names
        events = self.dw.to_events(id_cols=self.id_cols)
        return events

    def check_for_suffixes(self) -> list[str]:
        """Return any DwellSet columns that already end with a managed suffix."""
        matches = []
        for col, suf in product(self.dw.data.columns, self.suffixes.values()):
            if col.endswith(suf):
                matches.append(col)
        return matches


class MinPowerChargingManager(IndependentDwellChargingManager):
    """Deliver energy at constant minimum power over the full dwell duration.

    Computes power as ``energy / duration`` (zero when either is zero) and
    schedules a single charge-start event at ``dwell_start`` and a charge-end
    event at ``dwell_end``.

    ``seq_names = ["dwell_start", "dwell_end"]``.
    """

    seq_names = ["dwell_start", "dwell_end"]

    def set_dwell_events(self) -> Self:
        """Populate dwell-start and dwell-end event columns for constant-power charging.

        Computes ``dwell_start_power_kw = energy / duration`` (0 for zero-energy
        or zero-duration dwells), sets ``dwell_end_power_kw = 0`` (or
        ``-dwell_start_power_kw`` for ``DIFFERENCES`` mode), and calculates the
        ``dwell_start_duration`` as the interval between the two event timestamps.

        Returns:
            ``self`` with event columns added to ``self.dw.data``.
        """
        cnames = {}
        for k, v in self.suffixes.items():
            cnames[k] = [f"{seqn}_{v}" for seqn in self.seq_names]

        # Set values at dwell_start
        self.dw.data[cnames["time"][0]] = self.dw.data[self.dw.start]

        zero_energy = self.dw.data[self.energy] == 0.0
        zero_dur = self.dw.data[self.duration] == 0.0
        good_div = ~(zero_energy | zero_dur)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            self.dw.data[cnames["power"][0]] = (
                self.dw.data[self.energy] / self.dw.data[self.duration]
            )
            self.dw.data[cnames["power"][0]] = self.dw.data[cnames["power"][0]].where(
                good_div, 0.0
            )

        nonzero_pwr = self.dw.data[cnames["power"][0]] != 0
        self.dw.data[cnames["plugged"][0]] = 1 * nonzero_pwr

        # Set values at dwell_end
        self.dw.data[cnames["time"][-1]] = self.dw.data[self.dw.end]
        self.dw.data[cnames["duration"][0]] = (
            self.dw.data[cnames["time"][-1]] - self.dw.data[cnames["time"][0]]
        )
        self.dw.data[cnames["duration"][-1]] = pd.NA
        self.dw.data[cnames["duration"][-1]] = pd.to_timedelta(
            self.dw.data[cnames["duration"][-1]]
        )

        if self.prof_type == ProfileType.OBSERVATIONS:
            self.dw.data[cnames["power"][-1]] = 0
            self.dw.data[cnames["plugged"][-1]] = 0
        elif self.prof_type == ProfileType.DIFFERENCES:
            self.dw.data[cnames["power"][-1]] = -self.dw.data[cnames["power"][0]]
            self.dw.data[cnames["plugged"][-1]] = -1 * nonzero_pwr
        else:
            raise NotImplementedError()

        return self


class ImmediateChargingManager(IndependentDwellChargingManager):
    """Deliver energy at maximum available power starting at dwell arrival.

    Computes the charge-end timestamp as
    ``dwell_start + ceil(energy / max_power, "s")``, capped at ``dwell_end``.
    Schedules a charge-start event at ``dwell_start`` and a charge-end event
    at the computed charge-end time.

    ``seq_names = ["dwell_start", "charge_end"]``.
    """

    seq_names = ["dwell_start", "charge_end"]

    def set_dwell_events(self) -> Self:
        """Populate dwell-start and charge-end event columns for immediate charging.

        Computes the charge duration as ``energy / max_power`` (0 when either
        is zero), converts to a timedelta ceiled to the nearest second, and
        clips to the available dwell window.  Sets ``charge_end_power_kw = 0``
        (or ``-max_power`` for ``DIFFERENCES`` mode).

        Returns:
            ``self`` with event columns added to ``self.dw.data``.
        """
        cnames = {}
        for k, v in self.suffixes.items():
            cnames[k] = [f"{seqn}_{v}" for seqn in self.seq_names]

        # Set units at dwell_start
        self.dw.data[cnames["time"][0]] = self.dw.data[self.dw.start]
        self.dw.data[cnames["power"][0]] = self.dw.data[self.max_power]

        nonzero_pwr = self.dw.data[cnames["power"][0]] != 0
        self.dw.data[cnames["plugged"][0]] = 1 * nonzero_pwr

        # Set values at charge_end
        # Assumes that units are kWh, kW, and hours
        zero_energy = self.dw.data[self.energy] == 0.0
        zero_power = self.dw.data[self.max_power] == 0.0
        good_div = ~(zero_energy | zero_power)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            self.dw.data["charge_hrs"] = (
                self.dw.data[self.energy] / self.dw.data[self.max_power]
            )
            self.dw.data["charge_hrs"] = self.dw.data["charge_hrs"].where(good_div, 0.0)

        if self.dw.is_dask:
            charge_time = dd.to_timedelta(self.dw.data["charge_hrs"], unit="hour")
        else:
            charge_time = pd.to_timedelta(self.dw.data["charge_hrs"], unit="hour")
        charge_time = charge_time.dt.ceil(freq="s")  # To avoid messy near-second values

        charge_end = self.dw.data[self.dw.start] + charge_time
        charge_end = charge_end.clip(upper=self.dw.data[self.dw.end])

        # Set values at dwell_end
        self.dw.data[cnames["time"][-1]] = charge_end
        self.dw.data[cnames["duration"][0]] = (
            self.dw.data[cnames["time"][-1]] - self.dw.data[cnames["time"][0]]
        )
        self.dw.data[cnames["duration"][-1]] = pd.NA
        self.dw.data[cnames["duration"][-1]] = pd.to_timedelta(
            self.dw.data[cnames["duration"][-1]]
        )

        if self.prof_type == ProfileType.OBSERVATIONS:
            self.dw.data[cnames["power"][-1]] = 0
            self.dw.data[cnames["plugged"][-1]] = 0
        elif self.prof_type == ProfileType.DIFFERENCES:
            self.dw.data[cnames["power"][-1]] = -self.dw.data[self.max_power]
            self.dw.data[cnames["plugged"][-1]] = -1 * nonzero_pwr
        else:
            raise NotImplementedError()

        self.dw.data = self.dw.data.drop(columns=["charge_hrs"])
        return self


class OptimizerDwellChargingManager(AbstractChargingManager):
    """Placeholder for a future optimisation-based charging manager.

    Intended to support time-of-use rates and joint SoC optimisation across
    dwells within a region.  Not yet implemented.
    """

    # TODO: Eventually, I could make only some very slight modifications to the
    # optimization code I wrote in the winter, which fix the charging energy of each
    # session and then split up the sessions into discrete time intervals, in order to
    # build a fully parallelized optimization for each hexagon, permitting arbitrary
    # ToU rates, vehicle SoC evolutions (within session), etc.

    # TODO: Even if optimization is occurring on regions larger than a single hexagon,
    # you should still report events by hexagon in order to allow the maximum
    # flexibility of summaries later.
    pass


_MANAGER_MAP = {
    k: v
    for k, v in globals().items()
    if isinstance(v, type)
    and issubclass(v, AbstractChargingManager)
    and v is not AbstractChargingManager
}
