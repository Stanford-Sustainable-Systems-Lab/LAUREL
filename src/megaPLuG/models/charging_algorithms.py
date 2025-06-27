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
        pd.Int64Dtype().name: np.dtypes.Int64DType().name,
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
            outs["dwell_init_delay_hrs"][i] = cur_delay
            veh_is_dead = np.isnan(cur_energy) or cur_energy < 0

            # Check dwell status
            dwell_is_refresh = dwls["refresh"][i]

            avail_hrs = dwls["dwell_hrs"][i]

            # Manage vehicles running out of energy and resuscitating
            if veh_is_dead:
                if dwell_is_refresh:  # If we are at a refresh point, then revive
                    cur_energy = 0.0
                    cur_delay = 0.0
                    chg, dly, mode = choice_func(
                        cur_energy, avail_hrs, dwls[i], veh, modes
                    )
                else:  # If not, then become/stay dead
                    chg = np.nan
                    dly = np.nan
                    mode = np.argmin(modes["avail_kw"])
            else:
                chg, dly, mode = choice_func(cur_energy, avail_hrs, dwls[i], veh, modes)

            if dwell_is_refresh:
                dly_lim = np.maximum(
                    dly, -cur_delay
                )  # TODO: Add limits on delay reduction at refreshes
            else:
                dly_lim = np.maximum(dly, 0)  # No delay reduction except at refreshes
            outs["charge_kwh"][i] = chg
            outs["delay_inc_hrs"][i] = np.maximum(dly_lim, 0)
            outs["delay_dec_hrs"][i] = np.abs(np.minimum(dly_lim, 0))
            outs["charge_mode_id"][i] = mode
            cur_energy += chg
            cur_delay += dly_lim

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
    min_soc_charge: str
    plug_in_and_out_delay_hrs: str
    consumed_kwh_next: str
    consumed_kwh_shift: str
    power_kw_shift_max_remaining: str

    def __init__(
        self,
        soc_buffer_low: str,
        soc_buffer_high: str,
        min_soc_charge: str,
        plug_in_and_out_delay_hrs: str,
        consumed_kwh_next: str,
        consumed_kwh_shift: str,
        power_kw_shift_max_remaining: str,
        **kwargs,
    ) -> None:
        self.soc_buffer_low = soc_buffer_low
        self.soc_buffer_high = soc_buffer_high
        self.min_soc_charge = min_soc_charge
        self.plug_in_and_out_delay_hrs = plug_in_and_out_delay_hrs
        self.consumed_kwh_next = consumed_kwh_next
        self.consumed_kwh_shift = consumed_kwh_shift
        self.power_kw_shift_max_remaining = power_kw_shift_max_remaining
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
        EXTREME_DELAY_HRS = 10000.0  # More than a year of delay
        BETA_SOC = 0.5

        # Set the shapes of the evaluation arrays
        N_CHG_OPTS = 6
        n_modes = modes["avail_kw"].shape[0]
        caster = np.ones((N_CHG_OPTS, 1), dtype=float)

        ## Build the set of charging energy options. Structure this array so that
        # options up and to the left (smaller indices in the flattened array) are more
        # attractive given a tie. For example, lower power and less delay options should
        # be preferred.
        e = np.zeros((N_CHG_OPTS, n_modes), dtype=float)

        # Option 1: No charging
        # e[0, :] = 0 # No need to actually call this line

        # Option 2: Charge for available time on available power levels, with a minimum
        #   level charged to avoid tiny charging sessions.
        powers_flat = modes["avail_kw"] * dwl["modes_avail"]
        min_e_chg = veh["min_soc_charge"] * veh["batt_cap"]
        e[1, :] = np.maximum(powers_flat * avail_hrs, min_e_chg)

        # Option 3: Charge for next trip
        buff = veh["soc_buffer_low"] * veh["batt_cap"]
        e[2, :] = np.maximum(dwl["consumed_kwh_next"] + buff - cur_energy, 0)

        # Option 4: Charge for full shift
        nrg_needed_shift = np.maximum(dwl["consumed_kwh_shift"] + buff - cur_energy, 0)
        e[3, :] = nrg_needed_shift

        # Option 5: Charge to optimal SoC, with a minimum level charged to avoid tiny
        #   charging sessions.
        e[4, :] = np.maximum(
            veh["soc_buffer_high"] * veh["batt_cap"] - cur_energy, min_e_chg
        )

        # Option 6: Charge to fill battery
        e[5, :] = np.maximum(veh["batt_cap"] - cur_energy, 0)

        # Zero out charging on unavailable modes
        e = e * (caster * dwl["modes_avail"])

        ## Calculate outcomes
        # Energy at end of charging (i.e., "final" energy)
        e_fin = cur_energy + e

        # Trip successfully completed
        # TODO: Implement a soft lower buffer, to deter insufficient charging
        e_next = e_fin - dwl["consumed_kwh_next"]
        trip_succeeds = np.where(e_next > 0, 0, -np.inf)

        # Battery charged within bounds
        batt_respected = np.where(e_fin <= veh["batt_cap"], 0, -np.inf)

        # Delay at this dwell
        powers = caster * powers_flat
        chg_time = np.where(powers == 0.0, EXTREME_DELAY_HRS, e / powers)
        chg_time = np.where(e == 0.0, 0.0, chg_time)
        # If this is a zero-duration optional stop, add a delay penalty for plugging in
        # and out. This creates a subtle bias for charging later rather than sooner,
        # since a similar bias is not added to the delay-in-remainder-of-shift
        # (delay_shift) variable.
        if not avail_hrs > 0:
            plug_time = veh["plug_in_and_out_delay_hrs"]
            chg_time = np.where(e == 0, chg_time, chg_time + plug_time)
        time_delta = chg_time - avail_hrs  # Will be negative if time can be made up
        delay = np.maximum(time_delta, 0)  # Can only ever count new delay

        # Delay that would result if we charged instead at the highest power dwell
        #  remaining in this shift.
        if dwl["power_kw_shift_max_remaining"] == 0:
            delay_shift = np.ones_like(delay) * EXTREME_DELAY_HRS
        else:
            e_rem = (
                nrg_needed_shift - cur_energy - np.maximum(nrg_needed_shift - e_fin, 0)
            )
            delay_shift = e_rem / dwl["power_kw_shift_max_remaining"]

        soc_targeting = (
            -BETA_SOC * (veh["soc_buffer_high"] - e_fin / veh["batt_cap"]) ** 2
        )

        ## Calculate indirect utility and maximize
        v = soc_targeting - (delay - delay_shift) + trip_succeeds + batt_respected
        flat_best_idx = np.argmax(v, axis=None)
        best_idx = (flat_best_idx // n_modes, flat_best_idx % n_modes)
        chg = e[best_idx]
        dly = time_delta[best_idx]
        mode = best_idx[1]
        return (chg, dly, mode)
