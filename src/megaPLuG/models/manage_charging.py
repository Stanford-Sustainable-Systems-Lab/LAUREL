import logging
from abc import ABC, abstractmethod
from enum import IntEnum, auto
from itertools import product
from typing import Self

import dask.dataframe as dd
import numpy as np
import pandas as pd

from megaPLuG.models.dwell_sets import DwellSet

logger = logging.getLogger(__name__)


class ProfileType(IntEnum):
    """Possible profile type to be used by AbstractChargingManager."""

    OBSERVATIONS = auto()
    DIFFERENCES = auto()


class AbstractChargingManager(ABC):
    """This class sets the interface for all concrete charging managers.

    Attributes:
        dw: DwellSet that this ChargingManager will compute over
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
        """Initialize the ChargingManager."""
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
        """Get the load profiles which result from the given charging management."""
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
    """Charge each dwell independently without considering influences within a region."""

    suffixes = {
        "time": "time",
        "duration": "duration",
        "power": "power_kw",
        "plugged": "plugged",
    }

    @property
    @abstractmethod
    def seq_names(self) -> list[str]:
        """Get the column prefixes used to track the sequence of events within a dwell."""
        pass  # Implement this in the concrete classes by setting the attribute seq_names

    @abstractmethod
    def set_dwell_events(self) -> Self:
        """Set the charging management power change events within the dwell."""
        pass

    def get_events(self: Self) -> pd.DataFrame:
        """Take dwells and convert to events."""
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
        """Check if any of the suffixes are being used in DwellSet columns."""
        matches = []
        for col, suf in product(self.dw.data.columns, self.suffixes.values()):
            if col.endswith(suf):
                matches.append(col)
        return matches


class MinPowerChargingManager(IndependentDwellChargingManager):
    """Charge the vehicles at minimum acceptable power for the full dwell duration."""

    seq_names = ["dwell_start", "dwell_end"]

    def set_dwell_events(self) -> Self:
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
    """Charge the vehicles at maximum available power as soon as possible."""

    seq_names = ["dwell_start", "charge_end"]

    def set_dwell_events(self) -> Self:
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
    """Charge each dwell based on an optimization algorithm."""

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
