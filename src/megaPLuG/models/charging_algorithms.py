from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np
import pandas as pd
from megaPLuG.models.dwell_sets import DwellSet
from numba import jit
from tqdm import tqdm


class AbstractChargingChoiceStrategy(ABC):
    """Defines how charging choice strategies should be implemented.

    This applies the strategy for a sequence of dwells which may cover multiple vehicles.
    Ensure that the vehicles have a "reset == True" at the beginning of their dwell
    set to ensure isolation.
    """

    consumed_kwh: str
    dwell_hrs: str
    avail_kw: str
    reset: str
    batt_cap: str
    random_seed: str
    _replace_dtypes: dict = {np.bool.__name__: "u1"}
    # See https://numpy.org/doc/stable/user/basics.rec.html#structured-datatype-creation
    # for creation instructions
    _output_records_dtype = np.dtype(
        {
            "names": ["dwell_init_kwh", "charge_kwh", "charge_mode"],
            "formats": [np.float64, np.float64, np.str_],
        }
    )
    _renamer: dict

    def __init__(
        self,
        consumed_kwh: str,
        dwell_hrs: str,
        avail_kw: str,
        reset: str,
        batt_cap: str,
        random_seed: str,
        rng_alpha: str,
        rng_beta: str,
    ) -> None:
        """Set up the column name mappings to be used for this strategy.

        Because of the renaming during the creation of the recarrays, you can refer
        to the variables in the recarrays by the corresponding attribute names from
        the ChargingChoiceStrategy class which you are using.

        At a minimum, we need the columns in the dwells:
            - consumed_kwh
            - dwell_hrs
            - avail_kw (which may end up being a recarray itself to accomodate multiple modes)
            - reset

        And we need the vehicle parameters:
            - battery_capacity
            - rng_alpha
            - rng_beta
        """
        self.consumed_kwh = consumed_kwh
        self.dwell_hrs = dwell_hrs
        self.avail_kw = avail_kw
        self.reset = reset
        self.batt_cap = batt_cap
        self.random_seed = random_seed
        self.rng_alpha = rng_alpha
        self.rng_beta = rng_beta

        self._renamer = {v: k for k, v in self.__dict__.items() if isinstance(v, str)}

    def convert_to_records(self, df: pd.DataFrame) -> np.recarray:
        """Convert a dataframe to a format which can be passed into the JIT-ed simulation.

        This method performs two primary tasks: renaming and type casting.
        """
        rename_key_set = set(self._renamer.keys())
        use_cols = list(rename_key_set.intersection(set(df.columns)))
        df_sel = df.loc[:, use_cols]
        df_sel = df_sel.rename(columns=self._renamer)
        col_dtypes = self._get_recarray_dtypes(df_sel)
        recs = df_sel.to_records(column_dtypes=col_dtypes)
        return recs

    def _get_recarray_dtypes(self, df: pd.DataFrame) -> dict[str, str]:
        """Convert a DataFrame to a Record Array using opinionated type conversions.

        Note: Boolean values seem to not convert as bytes, so I will use a small unsigned
        integer instead
        """
        col_dtypes = {}
        for col, dtype in zip(df.columns, df.dtypes):
            if dtype.name == np.bool.__name__:
                col_dtypes.update({col: self._replace_dtypes[dtype.name]})
        return col_dtypes

    def run(
        self,
        dwells: DwellSet,
        vehs: pd.DataFrame,
        locs: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """Run the simulation for a single vehicle by calling the sub-class-specific JIT-ed simulator."""
        dwl_recs = self.convert_to_records(dwells.data)
        veh_recs = self.convert_to_records(vehs)
        rngs = vehs[self.random_seed].transform(lambda s: np.random.default_rng(seed=s))

        if np.any(vehs.index.duplicated()):
            raise RuntimeError("Duplicate vehicle ids detected")

        veh_idxr = pd.Series(data=np.arange(len(vehs)), index=vehs.index)
        grp_idxs = dwells.data.groupby(dwells.veh).indices
        outs = np.recarray((dwl_recs.shape[0],), dtype=self._output_records_dtype)
        for grp, idxs in tqdm(
            grp_idxs.items()
        ):  # Using pandas groupby indices to move over dwell recarray
            outs[idxs] = self._simulate(
                choice_func=self._choose_charging,
                dwls=dwl_recs[idxs],
                veh=veh_recs[veh_idxr.loc[grp]],
                outs=outs[idxs],
                rng=rngs[grp],
            )
        out_df = pd.DataFrame.from_records(outs, index=dwells.data.index)
        dwells_w_charging = pd.concat([dwells.data, out_df], axis=1)
        return dwells_w_charging

    @staticmethod
    @jit
    def _simulate(
        choice_func: Callable,
        dwls: np.recarray,
        veh: np.recarray,
        outs: np.recarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Simulate the evolution of the SoC with charging choices.

        Because of the renaming during the creation of the recarrays, you should refer
        to the variables in the recarrays by the corresponding attribute names from
        the ChargingChoiceStrategy class which you are using.
        """
        nsteps = dwls.shape[0]
        cur_energy = np.nan
        for i in range(nsteps):
            if dwls["reset"][i]:
                soc = rng.beta(a=veh["rng_alpha"], b=veh["rng_beta"])
                cur_energy = veh["batt_cap"] * soc
            cur_energy -= dwls["consumed_kwh"][i]
            outs["dwell_init_kwh"][i] = cur_energy
            avail_kwh = dwls["dwell_hrs"][i] * dwls["avail_kw"][i]
            if np.isnan(cur_energy) or cur_energy < 0:  # Currently dead
                if (
                    avail_kwh >= veh["batt_cap"]
                ):  # If full recharge is possible, then refresh
                    cur_energy = 0
                    chg = veh["batt_cap"]
                else:  # If not, then become/stay dead
                    chg = np.nan
            else:
                chg = choice_func(cur_energy, dwls[i], veh)
            outs["charge_kwh"][i] = chg
            outs["charge_mode"][i] = "None"  # TODO: Implement mode selection
            cur_energy += chg

        return outs

    @staticmethod
    @abstractmethod
    def _choose_charging(
        cur_energy: float,
        dwl: np.recarray,
    ) -> np.ndarray:
        """Choose charging energy and mode.

        Note: Will be passed to JIT-ed _simulate().
        """
        pass


class SoCThreshChargingChoiceStrategy(AbstractChargingChoiceStrategy):
    """A charging choice strategy where the vehicle charges whenever it falls below
    an SoC threshold.
    """

    charge_soc: str

    def __init__(self, charge_soc: str, **kwargs) -> None:
        self.charge_soc = charge_soc
        super().__init__(**kwargs)

    @staticmethod
    @jit
    def _choose_charging(
        cur_energy: float,
        dwl: np.recarray,
        veh: np.recarray,
    ) -> np.ndarray:
        """Choose charging energy and mode.

        Note: Will be passed to JIT-ed _simulate().
        """
        if cur_energy / veh["batt_cap"] <= veh["charge_soc"]:
            chg = np.minimum(
                veh["batt_cap"] - cur_energy, dwl["dwell_hrs"] * dwl["avail_kw"]
            )
        else:
            chg = 0
        return chg
