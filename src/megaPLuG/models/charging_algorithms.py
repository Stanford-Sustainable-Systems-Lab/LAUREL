from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np
import pandas as pd
from numba import jit
from tqdm import tqdm

from megaPLuG.models.dwell_sets import DwellSet
from megaPLuG.utils.data import to_arrays


class AbstractChargingChoiceStrategy(ABC):
    """Defines how charging choice strategies should be implemented.

    This applies the strategy for a sequence of dwells which may cover multiple vehicles.
    Ensure that the vehicles have a "reset == True" at the beginning of their dwell
    set to ensure isolation.
    """

    consumed_kwh: str
    dwell_hrs: str
    modes_avail: str
    refresh: str
    reset: str
    batt_cap: str
    random_seed: str
    _replace_dtypes: dict = {
        np.bool.__name__: "u1",
        pd.Float64Dtype().name: np.dtypes.Float64DType().name,
    }
    # See https://numpy.org/doc/stable/user/basics.rec.html#structured-datatype-creation
    # for creation instructions
    _output_records_dtype = np.dtype(
        {
            "names": [
                "dwell_init_kwh",
                "charge_kwh",
                "dwell_init_delay_hrs",
                "delay_inc_hrs",
                "delay_dec_hrs",
                "charge_mode_id",
            ],
            "formats": [
                np.float64,
                np.float64,
                np.float64,
                np.float64,
                np.float64,
                np.int32,
            ],
        }
    )
    _renamer: dict

    def __init__(
        self,
        consumed_kwh: str,
        dwell_hrs: str,
        modes_avail: str,
        avail_kw: str,
        refresh: str,
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
            - modes_avail
            - avail_kw
            - reset

        And we need the vehicle parameters:
            - battery_capacity
            - rng_alpha
            - rng_beta
        """
        self.consumed_kwh = consumed_kwh
        self.dwell_hrs = dwell_hrs
        self.modes_avail = modes_avail
        self.avail_kw = avail_kw
        self.refresh = refresh
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
        arrs, names, formats = to_arrays(df=df_sel, column_dtypes=col_dtypes)
        for i, fmt in enumerate(formats):
            if fmt.shape != ():  # If the dtype has some shape, not just scalar
                arrs[i] = np.vstack(arrs[i])
        recs = np.rec.fromarrays(arrs, dtype={"names": names, "formats": formats})
        return recs

    def _get_recarray_dtypes(self, df: pd.DataFrame) -> dict[str, str]:
        """Convert a DataFrame to a Record Array using opinionated type conversions.

        Note: Boolean values seem to not convert as bytes, so I will use a small unsigned
        integer instead
        """
        col_dtypes = {}
        for col, dtype in zip(df.columns, df.dtypes):
            # First, determine if there is internal structure within the column
            if isinstance(dtype, np.dtypes.ObjectDType):
                example = df[col].iloc[0]
                if isinstance(example, np.ndarray):
                    check_dtype = example.dtype
                    shp = example.shape
            else:
                check_dtype = dtype
                shp = None

            # Then, override dtypes incompatible with Numba
            out_fmt = check_dtype.name
            if out_fmt in self._replace_dtypes:
                out_fmt = self._replace_dtypes[out_fmt]

            if shp is not None:
                out_dtype = (out_fmt, shp)
            else:
                out_dtype = out_fmt
            col_dtypes.update({col: np.dtype(out_dtype)})

        return col_dtypes

    def run(
        self,
        dwells: DwellSet,
        vehs: pd.DataFrame,
        modes: pd.DataFrame,
    ) -> pd.DataFrame:
        """Run the simulation for a single vehicle by calling the sub-class-specific JIT-ed simulator."""
        if np.any(vehs.index.duplicated()):
            raise RuntimeError("Duplicate vehicle ids detected")

        # Convert index to integer if it's not already
        try:
            # First check if it's numeric but not integer
            if pd.api.types.is_numeric_dtype(
                modes.index
            ) and not pd.api.types.is_integer_dtype(modes.index):
                modes.index = modes.index.astype(int)
            # Then check if it's not numeric at all
            elif not pd.api.types.is_numeric_dtype(modes.index):
                raise TypeError(
                    f"Modes index must have integer dtype, got {modes.index.dtype}"
                )
        except (ValueError, TypeError) as e:
            # This will catch both non-numeric values and explicit TypeError from above
            raise TypeError(
                f"Modes dataframe index must be convertible to integer. Error: {e}"
            )

        # Additional validation that index is integer type after any conversions
        if not pd.api.types.is_integer_dtype(modes.index):
            raise TypeError(
                f"Modes index must have integer dtype, got {modes.index.dtype}"
            )

        dwl_recs = self.convert_to_records(dwells.data)
        veh_recs = self.convert_to_records(vehs)
        mode_recs = self.convert_to_records(modes)
        rngs = vehs[self.random_seed].transform(lambda s: np.random.default_rng(seed=s))

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
                modes=mode_recs,
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
        modes: np.recarray,
        outs: np.recarray,
        rng: np.random.Generator,
        round_decimals: int = 4,
    ) -> np.ndarray:
        """Simulate the evolution of the SoC with charging choices.

        Because of the renaming during the creation of the recarrays, you should refer
        to the variables in the recarrays by the corresponding attribute names from
        the ChargingChoiceStrategy class which you are using.
        """
        nsteps = dwls.shape[0]
        cur_energy = np.nan
        cur_delay = 0.0
        for i in range(nsteps):
            # Manage hard resets of vehicles, including at the beginning
            if dwls["reset"][i]:
                soc = 1.0  # rng.beta(a=veh["rng_alpha"], b=veh["rng_beta"])
                cur_energy = veh["batt_cap"] * soc

            # Update vehicle status
            cur_energy -= dwls["consumed_kwh"][i]
            outs["dwell_init_kwh"][i] = cur_energy

            # Check if we can decrease delay at this dwell
            outs["dwell_init_delay_hrs"][i] = cur_delay
            if dwls["refresh"][i] and dwls["dwell_hrs"][i] > cur_delay:
                delay_reduction = cur_delay
            else:
                delay_reduction = 0.0
            outs["delay_dec_hrs"][i] = delay_reduction
            cur_delay -= delay_reduction
            avail_hrs = dwls["dwell_hrs"][i] - delay_reduction

            # Manage vehicles running out of energy and resuscitating
            max_power_mode = np.argmax(dwls["modes_avail"][i] * modes["avail_kw"])
            avail_kwh = avail_hrs * modes["avail_kw"][max_power_mode]
            if np.isnan(cur_energy) or cur_energy < 0:  # Currently dead
                if (
                    avail_kwh >= veh["batt_cap"]
                ):  # If full recharge is possible, then revive
                    cur_energy = 0.0
                    cur_delay = 0.0
                    chg = veh["batt_cap"]
                    dly = 0.0
                    mode = max_power_mode
                else:  # If not, then become/stay dead
                    chg = np.nan
                    dly = np.nan
                    mode = np.argmin(modes["avail_kw"])
            else:
                chg, dly, mode = choice_func(cur_energy, avail_hrs, dwls[i], veh, modes)
            outs["charge_kwh"][i] = chg
            outs["delay_inc_hrs"][i] = dly
            outs["charge_mode_id"][i] = mode
            cur_energy += chg
            cur_delay += dly

        return outs

    @staticmethod
    @abstractmethod
    def _choose_charging(
        cur_energy: float,
        dwl: np.recarray,
        veh: np.recarray,
        modes: np.recarray,
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
        modes: np.recarray,
    ) -> np.ndarray:
        """Choose charging energy and mode.

        Note: Will be passed to JIT-ed _simulate().
        """
        # TODO: Convert this to a discrete choice framework
        # TODO: Return the selected mode
        if cur_energy / veh["batt_cap"] <= veh["charge_soc"]:
            chg = np.minimum(
                veh["batt_cap"] - cur_energy, dwl["dwell_hrs"] * modes["avail_kw"][-1]
            )
        else:
            chg = 0
        return chg


class ForwardLookingChargingChoiceStrategy(AbstractChargingChoiceStrategy):
    """A charging choice strategy where the vehicle looks ahead to the next trip and
    the rest of the current shift to determine when to charge.
    """

    soc_buffer_low: str
    soc_buffer_high: str
    plug_in_and_out_delay_hrs: str
    consumed_kwh_next: str
    consumed_kwh_shift: str

    def __init__(
        self,
        soc_buffer_low: str,
        soc_buffer_high: str,
        plug_in_and_out_delay_hrs: str,
        consumed_kwh_next: str,
        consumed_kwh_shift: str,
        **kwargs,
    ) -> None:
        self.soc_buffer_low = soc_buffer_low
        self.soc_buffer_high = soc_buffer_high
        self.plug_in_and_out_delay_hrs = plug_in_and_out_delay_hrs
        self.consumed_kwh_next = consumed_kwh_next
        self.consumed_kwh_shift = consumed_kwh_shift
        super().__init__(**kwargs)

    @staticmethod
    @jit
    def _choose_charging(
        cur_energy: float,
        avail_hrs: float,
        dwl: np.recarray,
        veh: np.recarray,
        modes: np.recarray,
    ) -> np.ndarray:
        """Choose the charging energy, delay, and mode for a dwell."""
        # Set some weighting constants
        EXTREME = 100.0
        BETA_DELAY = 0.01

        # Select energy needed so that we only charge when we cannot make the next trip
        # And we only charge to 100% when needed for the next trip
        buff = veh["soc_buffer_low"] * veh["batt_cap"]
        nrg_needed_trip = np.maximum(dwl["consumed_kwh_next"] + buff - cur_energy, 0)

        if nrg_needed_trip > 0:
            # Energy needed to get to the end of the current shift
            nrg_needed_shift = np.maximum(
                dwl["consumed_kwh_shift"] + buff - cur_energy, 0
            )

            # Maximum energy that can be charged while staying at or below the SoC soft cap
            nrg_max_soft_cap_soc = veh["soc_buffer_high"] * veh["batt_cap"] - cur_energy

            # Charge at least enough for the next trip, and if more energy is needed to
            # finish the shift, then charge up to either the shift need or the soft cap,
            # whichever is lower.
            nrg_needed_final = np.maximum(
                nrg_needed_trip,
                np.minimum(nrg_needed_shift, nrg_max_soft_cap_soc),
            )
        else:
            nrg_needed_final = 0

        # Structure this array so that options up and to the left (smaller indices in the
        # flattened array) are more attractive given a tie. For example, lower power and
        # less delay options should be preferred.
        N_CHG_OPTS = 3
        powers = modes["avail_kw"] * dwl["modes_avail"]
        n_powers = powers.shape[0]
        n_opts = N_CHG_OPTS * n_powers
        e = np.zeros(n_opts)

        # Option 1: No charging
        # e[0:n_powers] remains 0

        # Option 2: Charge for available time
        e[n_powers : 2 * n_powers] = powers * avail_hrs

        # Option 3: Charge to meet needs
        e[2 * n_powers :] = nrg_needed_final

        # Calculate outcomes
        e_cap = np.minimum(e, veh["batt_cap"] - cur_energy)
        powers_flat = np.ravel(
            powers.repeat(N_CHG_OPTS).reshape((-1, N_CHG_OPTS)).T
        )  # TODO: Consider pre-computing this and passing in for speed
        div = np.where(powers_flat == 0.0, EXTREME / BETA_DELAY, e_cap / powers_flat)
        div = np.where(e_cap == 0.0, 0.0, div)
        if not avail_hrs > 0:  # If this is a zero-duration optional stop
            div = np.where(e_cap == 0, div, div + veh["plug_in_and_out_delay_hrs"])
        delta = np.maximum(div - avail_hrs, 0)
        soc_next = (cur_energy + e_cap - dwl["consumed_kwh_next"]) / veh["batt_cap"]

        # Calculate indirect utilities
        # First, for the SoC at the end the next trip
        v_soc_next = np.zeros_like(soc_next)
        EPS = 1e-4
        low_bnd = 0 + EPS
        high_bnd = 1 - EPS
        std_idx = np.logical_and(soc_next > low_bnd, soc_next < high_bnd)
        not_extr = soc_next[std_idx]
        v_soc_next[std_idx] = -np.log(
            (1 - not_extr) / ((1 / veh["soc_buffer_low"] - 1) * not_extr)
        )
        v_soc_next[soc_next <= low_bnd] = -EXTREME
        v_soc_next[soc_next >= high_bnd] = EXTREME

        # Second, for the change in SoC during this dwell
        dsoc = np.minimum(e / veh["batt_cap"], 1 - soc_next)
        v_dsoc = 1 - (dsoc - 1) ** 2

        # Third, for the delay during this dwell
        v_delay = BETA_DELAY * delta

        # Dropped softmax because it is monotonic and we're not using the probabilities
        sel_idx = np.argmax(v_soc_next + v_dsoc - v_delay)
        chg = e_cap[sel_idx]
        delay = delta[sel_idx]
        mode = sel_idx % n_powers
        return (chg, delay, mode)
