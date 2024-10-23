import logging
from abc import ABC, abstractmethod
from itertools import product
from typing import Self

import pandas as pd
from megaPLuG.models.dwell_sets import DwellSet

logger = logging.getLogger(__name__)


class AbstractChargingManager(ABC):
    """This class sets the interface for all concrete charging managers.

    Attributes:
        dw: DwellSet that this ChargingManager will compute over
        energy: name of the column within dw containing the required energy for each dwell
        duration: name of the column within dw containing the duration of the dwell
        max_power: name of the column within dw containing the maximum charging power for this dwell
        region: name of the column containing the grouping variable for regions
        cost: name of the column giving the cost which was promised during the charging choice model
    """

    _dw: DwellSet = None
    _energy: str = None
    _duration: str = None
    _max_power: str = None
    _cost: str = None
    _region: str = None

    def __init__(
        self,
        dw: DwellSet,
        energy: str,
        duration: str,
        max_power: str,
        region: str,
        cost: str = None,
    ) -> None:
        """Initialize the ChargingManager."""
        self.dw = dw
        self.energy = energy
        self.duration = duration
        self.max_power = max_power
        self.region = region
        self.cost = cost

    @abstractmethod
    def get_load_profiles(self) -> pd.DataFrame:
        """Get the load profiles which result from the given charging management."""
        pass

    @property
    def energy(self):
        return self._energy

    @energy.setter
    def energy(self, value):
        self.dw._rename_idx_col(value, self._energy)
        self._energy = value

    @property
    def duration(self):
        return self._duration

    @duration.setter
    def duration(self, value):
        self.dw._rename_idx_col(value, self._duration)
        self._duration = value

    @property
    def max_power(self):
        return self._max_power

    @max_power.setter
    def max_power(self, value):
        self.dw._rename_idx_col(value, self._max_power)
        self._max_power = value

    @property
    def region(self):
        return self._region

    @region.setter
    def region(self, value):
        self.dw._rename_idx_col(value, self._region)
        self._region = value

    @property
    def cost(self):
        return self._cost

    @cost.setter
    def cost(self, value):
        self.dw._rename_idx_col(value, self._cost)
        self._cost = value


class IndependentDwellChargingManager(AbstractChargingManager):
    """Charge each dwell independently without considering influences within a region."""

    suffixes = {
        "time": "time",
        "power": "hex_kw_diff",
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

    def get_load_profiles(self, prof_col: str, dur_col: str) -> pd.DataFrame:
        """Take dwells and convert to events."""
        checks = self.check_for_suffixes()
        if len(checks) > 0:
            logger.warning(
                f"The following columns are already using the {type(self).__name__} suffixes: {', '.join(checks)}. If this is unexpected, then please change their names."
            )
        self = self.set_dwell_events()
        # self.dw.data = self.dw.data.dropna(subset=self.dw.seq_names)
        self.dw.seq_names = self.seq_names
        events = self.dw.to_events(id_cols=[self.region])
        # Sort by region and time
        events = DwellSet._sort_by_grp_time(
            df=events,
            grp_col=self.region,
            time_col=self.suffixes["time"],
            drop_cur_idx=True,
        )
        event_grp = events.groupby(self.region, sort=False)
        # Note: The vehicle and hexagon ids are rendered uninterpretable by the cumsum
        events[prof_col] = event_grp[self.suffixes["power"]].cumsum()
        events[dur_col] = event_grp[self.suffixes["time"]].transform(
            lambda ser: ser.shift(-1) - ser
        )
        events = events.drop(columns=[self.suffixes["power"]])
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
        pwr_cols = [f"{seqn}_{self.suffixes['power']}" for seqn in self.seq_names]
        time_cols = [f"{seqn}_{self.suffixes['time']}" for seqn in self.seq_names]
        self.dw.data[time_cols[0]] = self.dw.data[self.dw.start]
        self.dw.data[pwr_cols[0]] = (
            self.dw.data[self.energy] / self.dw.data[self.duration]
        )
        self.dw.data[time_cols[-1]] = self.dw.data[self.dw.end]
        self.dw.data[pwr_cols[-1]] = -self.dw.data[pwr_cols[0]]
        return self


class ImmediateChargingManager(IndependentDwellChargingManager):
    """Charge the vehicles at maximum available power as soon as possible."""

    seq_names = ["dwell_start", "charge_end"]

    def set_dwell_events(self) -> Self:
        pwr_cols = [f"{seqn}_{self.suffixes['power']}" for seqn in self.seq_names]
        time_cols = [f"{seqn}_{self.suffixes['time']}" for seqn in self.seq_names]
        self.dw.data[time_cols[0]] = self.dw.data[self.dw.start]
        self.dw.data[pwr_cols[0]] = self.dw.data[self.max_power]
        # Assumes that units are kWh, kW, and hours
        charge_hrs = self.dw.data[self.energy] / self.dw.data[self.max_power]
        charge_time = pd.to_timedelta(charge_hrs, unit="hour")
        charge_end = (self.dw.data[self.dw.start] + charge_time).dt.round(freq="s")
        charge_end = charge_end.clip(upper=self.dw.data[self.dw.end])
        self.dw.data[time_cols[-1]] = charge_end
        self.dw.data[pwr_cols[-1]] = -self.dw.data[self.max_power]
        return self


class OptimizerDwellChargingManager(AbstractChargingManager):
    """Charge each dwell based on an optimization algorithm."""

    # TODO: Eventually, I could make only some very slight modifications to the
    # optimization code I wrote in the winter, which fix the charging energy of each
    # session and then split up the sessions into discrete time intervals, in order to
    # build a fully parallelized optimization for each hexagon, permitting arbitrary
    # ToU rates, vehicle SoC evolutions (within session), etc.
    pass


_MANAGER_MAP = {
    k: v
    for k, v in globals().items()
    if isinstance(v, type)
    and issubclass(v, AbstractChargingManager)
    and v is not AbstractChargingManager
}
