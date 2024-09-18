from abc import ABC, abstractmethod
from typing import Self

import pandas as pd
from megaPLuG.models.dwell_sets import DwellSet


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

    event_col = "hex_kw_diff"

    def __init__(
        self,
        dw: DwellSet,
        energy: str,
        dur: str,
        seq_names: list[str],
        cost: str = None,
    ) -> None:
        super().__init__(dw, energy, dur, cost)
        self.dw.seq_names = seq_names

    @abstractmethod
    def set_dwell_events(self, event_cols: list[str]) -> Self:
        """Set the charging management power change events within the dwell."""
        pass

    def get_load_profiles(self) -> pd.DataFrame:
        """Take dwells and convert to events."""
        hex_kw_cols = [f"{seqn}_{self.event_col}" for seqn in self.dw.seq_names]
        self = self.set_dwell_events(event_cols=hex_kw_cols)
        self.dw.data = self.dw.data.dropna(subset=hex_kw_cols)

        events = self.dw.to_events()
        # Sort by hexagon and time
        events = DwellSet._sort_by_grp_time(
            df=events,
            grp_col=self.dw.hex,
            time_col=DwellSet._get_seq_name_tail(self.dw.seq_names[0], self.dw.start),
            drop_cur_idx=True,
        )
        profs = events.groupby(self.dw.hex)[self.event_col].cumsum()
        profs = profs.to_frame()
        return profs


class MinPowerChargingManager(IndependentDwellChargingManager):
    """Charge the vehicles at minimum acceptable power for the full dwell duration."""

    def __init__(
        self,
        dw: DwellSet,
        energy: str,
        dur: str,
        seq_names: list[str],
        cost: str = None,
    ) -> None:
        super().__init__(dw, energy, dur, seq_names, cost)

    def set_dwell_events(self, event_cols: list[str]) -> Self:
        self.dw.data[event_cols[0]] = self.dw.data[self.energy] / self.dw.data[self.dur]
        self.dw.data[event_cols[-1]] = -self.dw.data[event_cols[0]]
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
