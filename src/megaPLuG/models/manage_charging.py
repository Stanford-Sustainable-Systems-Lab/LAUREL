import logging
from abc import ABC, abstractmethod
from itertools import product
from typing import Self

import pandas as pd
from megaPLuG.models.dwell_sets import DwellSet

logger = logging.getLogger(__name__)


class AbstractChargingManager(ABC):
    """This class sets the interface for all concrete charging managers."""

    _dw = None
    _energy = None
    _dur = None
    _cost = None

    def __init__(
        self,
        dw: DwellSet,
        energy: str,
        dur: str,
        cost: str = None,
    ) -> None:
        """Initialize the ChargingManager."""
        self.dw = dw
        self.energy = energy
        self.dur = dur
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
    def dur(self):
        return self._dur

    @dur.setter
    def dur(self, value):
        self.dw._rename_idx_col(value, self._dur)
        self._dur = value

    @property
    def cost(self):
        return self._cost

    @cost.setter
    def cost(self, value):
        self.dw._rename_idx_col(value, self._cost)
        self._cost = value


class IndependentDwellChargingManager(AbstractChargingManager):
    """Charge each dwell independently without considering influences within a location."""

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

    def get_load_profiles(self, prof_col: str) -> pd.DataFrame:
        """Take dwells and convert to events."""
        checks = self.check_for_suffixes()
        if len(checks) > 0:
            logger.warning(
                f"The following columns are already using the {type(self).__name__} suffixes: {', '.join(checks)}. If this is unexpected, then please change their names."
            )
        self = self.set_dwell_events()
        # self.dw.data = self.dw.data.dropna(subset=self.dw.seq_names)
        self.dw.seq_names = self.seq_names
        events = self.dw.to_events()
        # Sort by hexagon and time
        events = DwellSet._sort_by_grp_time(
            df=events,
            grp_col=self.dw.hex,
            time_col=self.suffixes["time"],
            drop_cur_idx=True,
        )
        events[prof_col] = events.groupby(self.dw.hex)[self.suffixes["power"]].cumsum()
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
        self.dw.data[pwr_cols[0]] = self.dw.data[self.energy] / self.dw.data[self.dur]
        self.dw.data[time_cols[-1]] = self.dw.data[self.dw.end]
        self.dw.data[pwr_cols[-1]] = -self.dw.data[pwr_cols[0]]
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
