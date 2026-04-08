"""Charging-choice simulation strategies for the ``electrify_trips`` pipeline (Model Module 4).

Implements the per-vehicle utility-maximisation charging-choice model described
in the paper.  Each concrete :class:`AbstractChargingChoiceStrategy` encodes a
different decision rule for *when* and *how much* to charge at each dwell, and
is invoked by :meth:`AbstractChargingChoiceStrategy.run` which iterates over
all vehicles in a :class:`~laurel.models.dwell_sets.DwellSet`.

Pipeline overview
-----------------
1. :meth:`AbstractChargingChoiceStrategy.run` — outer loop: converts
   ``DwellSet.data``, vehicle parameters, and mode tables to NumPy recarrays;
   dispatches per-vehicle to :meth:`_simulate`.
2. :meth:`AbstractChargingChoiceStrategy._simulate` — JIT-compiled vehicle
   loop: evolves SoC step-by-step, calling :meth:`_choose_charging` at each
   eligible dwell.
3. :meth:`AbstractChargingChoiceStrategy._choose_charging` — abstract static
   method; each concrete strategy overrides this with a ``@jit``-decorated
   function expressing its decision logic.

Concrete strategies
-------------------
- :class:`SoCThreshChargingChoiceStrategy`: charges when SoC falls below a
  threshold; simple rule used for baseline and validation runs.
- :class:`ForwardLookingChargingChoiceStrategy`: evaluates six charging-energy
  options × available modes, scores each by an indirect utility function
  (SoC target + delay cost + feasibility penalties), and selects the maximum.

Key design decisions
--------------------
- **Recarray-based Numba interface**: all simulation inputs are converted to
  NumPy structured arrays (recarrays) before being passed to JIT functions.
  Column names are mapped to recarray field names via ``_renamer`` (a reversed
  ``{column_name → attribute_name}`` dict).  Pandas nullable types
  (``Float64``, ``Int64``) and booleans are substituted by
  ``_replace_dtypes`` to avoid Numba incompatibility.
- **Bitmask mode availability**: available charging modes per dwell are encoded
  as a ``uint64`` bitmask in the ``modes_avail`` field and decoded inside the
  JIT function via :func:`~laurel.utils.mode_masks.bits_to_bool_vec`.  This
  avoids passing variable-length arrays across the Python/Numba boundary.
- **``_output_records_dtype``**: a fixed structured dtype shared by all
  strategies ensures that the output recarray can always be concatenated with
  the input DwellSet data without column-name conflicts.
- **Delay accounting**: ``delay_inc_hrs`` and ``delay_dec_hrs`` track
  separately the delay incurred and recovered at each dwell; ``cur_delay`` is
  the running balance.  The ``max_delay_recoverable_hrs`` vehicle parameter
  caps how much accumulated delay can be erased at a single refresh point
  (depot or destination stop).
- **Revive logic**: a vehicle whose SoC goes negative is treated as "broken
  down" and neither charges nor accumulates delay until it reaches the next
  ``refresh`` dwell (a depot or destination stop with sufficient dwell time),
  where it is revived with SoC = 0.

References
----------
Passow, F., & Rajagopal, R. (2026). Identifying indicators to inform proactive
substation upgrades for charging electric heavy-duty trucks. *Applied Energy*.

Liu, Y., et al. (2021). A hierarchical optimization charging strategy for
plug-in hybrid electric vehicles. *IEEE Transactions on Vehicular Technology*.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable

import dask.dataframe as dd
import numpy as np
import pandas as pd
from numba import jit
from tqdm import tqdm

from laurel.models.dwell_sets import DwellSet
from laurel.utils.data import to_arrays
from laurel.utils.mode_masks import bits_to_bool_vec


class AbstractChargingChoiceStrategy(ABC):
    """Abstract base class for per-dwell charging choice strategies.

    Subclasses implement :meth:`_choose_charging` (a static ``@jit``-decorated
    function) to express their decision rule.  The base class provides the
    full simulation driver — recarray conversion, per-vehicle iteration, and
    output assembly — so subclasses only need to supply the decision logic.

    Each strategy instance stores the names of the input columns it expects
    from the dwell DataFrame and vehicle/mode parameter tables.  These names
    are used to build ``_renamer``, a dict mapping *input column names* to the
    *recarray field names* that the JIT functions address.

    Class attributes:
        _replace_dtypes: Dtype substitution map used when building recarrays
            from DataFrames.  Converts pandas nullable types and Python bools
            to Numba-compatible equivalents.
        _output_records_dtype: Fixed NumPy structured dtype for the output
            recarray produced by :meth:`_simulate`.  Fields are
            ``dwell_init_kwh``, ``charge_kwh``, ``dwell_init_delay_hrs``,
            ``delay_inc_hrs``, ``delay_dec_hrs``, ``charge_mode_id``.

    .. note::
        All vehicles in the supplied :class:`~laurel.models.dwell_sets.DwellSet`
        must have a ``reset == True`` at their first dwell to ensure their SoC
        is initialised correctly before any trip energy is subtracted.
    """

    consumed_kwh: str
    dwell_hrs: str
    modes_avail: str
    refresh: str
    reset: str
    critical: str
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
        critical: str,
        batt_cap: str,
        max_delay_recoverable_hrs: str,
        random_seed: str,
        rng_alpha: str,
        rng_beta: str,
    ) -> None:
        """Store column-name mappings for dwell, vehicle, and mode inputs.

        All arguments are *column names* (not values).  They are stored as
        instance attributes and used to build ``_renamer``, a dict mapping
        each input column name to the corresponding recarray field name used
        inside the JIT functions.

        Args:
            consumed_kwh: Dwell-table column: energy consumed by the preceding
                trip (kWh).
            dwell_hrs: Dwell-table column: duration of this dwell (hours).
            modes_avail: Dwell-table column: ``uint64`` bitmask of charging
                modes available at this location.
            avail_kw: Mode-table column: charging power for each mode (kW).
            refresh: Dwell-table column: boolean; ``True`` at depot/destination
                dwells that are eligible for delay recovery and SoC revival.
            reset: Dwell-table column: boolean; ``True`` at the start of each
                vehicle's simulation epoch.
            critical: Dwell-table column: boolean; ``True`` at dwells where
                charging should always be considered (e.g. all stops ≥ some
                minimum dwell time).
            batt_cap: Vehicle-table column: usable battery capacity (kWh).
            max_delay_recoverable_hrs: Vehicle-table column: maximum delay that
                can be erased at a single refresh point (hours).
            random_seed: Vehicle-table column: integer seed for the vehicle's
                RNG (reserved for future stochastic extensions).
            rng_alpha: Vehicle-table column: Beta distribution alpha parameter
                (reserved).
            rng_beta: Vehicle-table column: Beta distribution beta parameter
                (reserved).
        """
        self.consumed_kwh = consumed_kwh
        self.dwell_hrs = dwell_hrs
        self.modes_avail = modes_avail
        self.avail_kw = avail_kw
        self.refresh = refresh
        self.critical = critical
        self.reset = reset
        self.batt_cap = batt_cap
        self.max_delay_recoverable_hrs = max_delay_recoverable_hrs
        self.random_seed = random_seed
        self.rng_alpha = rng_alpha
        self.rng_beta = rng_beta

        self._renamer = {v: k for k, v in self.__dict__.items() if isinstance(v, str)}

    def convert_to_records(self, df: pd.DataFrame) -> np.recarray:
        """Convert a DataFrame to a Numba-compatible recarray.

        Selects only the columns whose names appear in ``_renamer``, renames
        them to their recarray field names (attribute names of this strategy
        instance), substitutes incompatible dtypes, and converts to a NumPy
        structured array.  Array-valued columns (e.g. ``avail_kw`` which holds
        a per-mode power vector per row) are stacked via ``np.vstack``.

        Args:
            df: Input DataFrame (dwell table, vehicle table, or mode table).

        Returns:
            NumPy recarray with field names matching this strategy's attribute
            names.
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
        """Build a per-column dtype dict for recarray construction.

        Inspects each column's dtype.  For scalar columns, substitutes any
        dtype name listed in ``_replace_dtypes``.  For object-dtype columns
        that contain NumPy arrays (e.g. per-mode power vectors), extracts the
        element dtype and shape and constructs a ``(dtype_str, shape)``
        sub-array type.

        Args:
            df: DataFrame after column renaming (field names match strategy
                attributes).

        Returns:
            Dict mapping column names to ``np.dtype`` objects suitable for
            ``np.rec.fromarrays``.
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
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """Simulate charging choices for all vehicles and return an annotated dwell table.

        Converts all inputs to recarrays, builds per-vehicle RNGs from the
        ``random_seed`` column, then iterates over vehicle groups (using
        ``groupby.indices`` for O(1) integer-indexed recarray slicing).  Calls
        :meth:`_simulate` for each vehicle and assembles the output recarray
        back into a DataFrame aligned with ``dwells.data``.

        Args:
            dwells: :class:`~laurel.models.dwell_sets.DwellSet` containing
                dwell-level input features.
            vehs: DataFrame of per-vehicle parameters indexed by vehicle ID.
                Must not contain duplicate vehicle IDs.
            modes: DataFrame of charging modes indexed by integer mode ID.
                The index must be (or be convertible to) integer.
            show_progress: Display a ``tqdm`` progress bar over vehicles.
                Defaults to ``True``.

        Returns:
            ``dwells.data`` concatenated with the output recarray columns
            (``dwell_init_kwh``, ``charge_kwh``, ``dwell_init_delay_hrs``,
            ``delay_inc_hrs``, ``delay_dec_hrs``, ``charge_mode_id``).

        Raises:
            RuntimeError: If ``vehs`` contains duplicate vehicle IDs.
            TypeError: If ``modes.index`` cannot be converted to integer.
        """
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

        itr = grp_idxs.items()
        if show_progress:
            itr = tqdm(itr)

        # Using pandas groupby indices to move over dwell recarray
        for grp, idxs in itr:
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
        """JIT-compiled vehicle-level SoC evolution loop.

        Steps through each dwell in ``dwls`` and:

        1. Resets the vehicle's SoC to full (``batt_cap``) at each ``reset``
           boundary.
        2. Subtracts ``consumed_kwh`` to update the current energy.
        3. Decides whether to charge:

           - If the vehicle is alive and the dwell is ``critical`` or
             ``refresh``: calls ``choice_func`` for a charging decision.
           - If the vehicle is alive but neither: no charging, no delay.
           - If the vehicle is dead (SoC < 0) at a ``refresh`` point: revives
             with SoC = 0 then charges.
           - If the vehicle is dead at any other dwell: records ``NaN`` charge,
             zero delay.

        4. Limits delay recovery at ``refresh`` points by
           ``max_delay_recoverable_hrs``.
        5. Writes results into ``outs``.

        Field names in the recarrays must match the strategy's attribute names
        (set up by :meth:`__init__` and applied by :meth:`convert_to_records`).

        Args:
            choice_func: The concrete :meth:`_choose_charging` static method.
            dwls: Per-dwell recarray for one vehicle.
            veh: Single-vehicle parameter recarray (shape ``(1,)``).
            modes: Charging-mode parameter recarray.
            outs: Pre-allocated output recarray for this vehicle's dwells.
            rng: NumPy random Generator for stochastic extensions (currently
                unused in deterministic strategies).
            round_decimals: Reserved for future rounding of output values.

        Returns:
            ``outs`` filled with simulation results for this vehicle.
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

            # Manage vehicles running out of energy and resuscitating
            veh_is_alive = cur_energy >= 0
            dwell_is_refresh = dwls["refresh"][i]
            dwell_is_critical = dwls["critical"][i]

            if veh_is_alive:  # Business as usual charging choice
                if dwell_is_critical or dwell_is_refresh:
                    chg, dly, mode = choice_func(cur_energy, dwls[i], veh, modes)
                else:
                    chg, dly, mode = (0.0, 0.0, np.argmin(modes["avail_kw"]))
            else:  # noqa: PLR5501
                if dwell_is_refresh:  # If we are at a refresh point, then revive
                    cur_energy = 0.0
                    cur_delay = 0.0
                    chg, dly, mode = choice_func(cur_energy, dwls[i], veh, modes)
                else:  # If not, then become/stay dead
                    chg, dly, mode = (np.nan, 0.0, np.argmin(modes["avail_kw"]))

            # Set limits on the amount of delay which is recoverable
            if dwell_is_refresh:
                max_recover = np.minimum(veh["max_delay_recoverable_hrs"], cur_delay)
            else:
                max_recover = 0.0
            dly_lim = np.maximum(dly, -max_recover)

            outs["charge_kwh"][i] = chg
            outs["delay_inc_hrs"][i] = np.maximum(dly_lim, 0)
            outs["delay_dec_hrs"][i] = np.abs(np.minimum(dly_lim, 0))
            outs["charge_mode_id"][i] = mode
            cur_energy += chg
            cur_delay += dly_lim

        return outs

    @classmethod
    def get_output_schema(cls, input: pd.DataFrame | dd.DataFrame) -> pd.DataFrame:
        """Generate an empty DataFrame matching the output schema of :meth:`run`.

        Used to supply ``meta`` to Dask ``map_partitions`` calls.  Starts from
        the input schema and appends the columns defined in
        ``_output_records_dtype``.

        Args:
            input: Representative DataFrame or Dask DataFrame whose schema
                (columns and dtypes) forms the base of the output schema.

        Returns:
            Empty pandas DataFrame with input columns plus the output columns
            (``dwell_init_kwh``, ``charge_kwh``, etc.) at their correct dtypes.
        """
        # Create empty DataFrame with input columns (these pass through unchanged)
        schema_df = dd.utils.make_meta(input)

        # Add output columns from the charging algorithm using _output_records_dtype
        for name in cls._output_records_dtype.names:
            field_dtype = cls._output_records_dtype.fields[name][0]
            schema_df[name] = pd.Series([], dtype=field_dtype)
        return schema_df

    @staticmethod
    @abstractmethod
    def _choose_charging(
        cur_energy: float,
        dwl: np.recarray,
        veh: np.recarray,
        modes: np.recarray,
    ) -> tuple[float, float, int]:
        """Select charging energy, delay change, and mode for one dwell.

        Called from the JIT-compiled :meth:`_simulate` loop.  Must itself be
        decorated with ``@jit`` in concrete subclasses.

        Args:
            cur_energy: Vehicle's current energy (kWh) after subtracting this
                dwell's consumed energy.
            dwl: Single-dwell recarray row.
            veh: Single-vehicle parameter recarray row.
            modes: Full charging-mode parameter recarray.

        Returns:
            Three-tuple ``(charge_kwh, delay_hrs, mode_id)`` where
            ``delay_hrs`` is positive for new delay and negative for
            recovered delay.
        """
        pass


class SoCThreshChargingChoiceStrategy(AbstractChargingChoiceStrategy):
    """Threshold-based charging strategy: charge when SoC falls below a fixed level.

    At each eligible dwell, charges to full at the maximum available power if
    the current SoC is at or below ``charge_soc``; otherwise does not charge.
    Used for baseline comparisons and validation.

    Additional Args (beyond :class:`AbstractChargingChoiceStrategy`):
        charge_soc: Vehicle-table column containing the SoC threshold
            (fraction of ``batt_cap``) below which charging is triggered.
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
    ) -> tuple[float, float, int]:
        """Charge to full at maximum power if SoC ≤ threshold, else skip.

        Args:
            cur_energy: Current energy (kWh).
            dwl: Dwell recarray row.
            veh: Vehicle recarray row; must contain ``batt_cap`` and
                ``charge_soc`` fields.
            modes: Mode recarray; highest-index mode is used for charging.

        Returns:
            ``(charge_kwh, 0.0, mode_id)`` — no delay is modelled by this
            simple strategy.
        """
        # TODO: Convert this to a discrete choice framework
        # TODO: Return the selected mode
        if cur_energy / veh["batt_cap"] <= veh["charge_soc"]:
            chg = float(
                np.minimum(
                    veh["batt_cap"] - cur_energy,
                    dwl["dwell_hrs"] * modes["avail_kw"][-1],
                )
            )
        else:
            chg = 0.0
        mode = int(np.argmax(modes["avail_kw"]))
        return (chg, 0.0, mode)


class ForwardLookingChargingChoiceStrategy(AbstractChargingChoiceStrategy):
    """Utility-maximising charging strategy with forward-looking shift awareness.

    At each eligible dwell, enumerates six charging-energy options ×
    ``n_modes`` charging modes, evaluates an indirect utility function for each
    combination, and selects the (option, mode) pair with the highest utility.

    The six energy options are:

    0. No charging.
    1. Charge for the full available dwell time.
    2. Charge for the next trip plus a low SoC buffer.
    3. Charge for the remainder of the shift plus a low SoC buffer.
    4. Charge to the target (high) SoC.
    5. Charge to full battery.

    The utility function penalises:

    - Delay relative to the counterfactual delay from charging later in the
      shift at the highest remaining power level.
    - Deviation from the target SoC (quadratic penalty).
    - Constraint violations (infeasible trip, battery overflow, infeasible
      shift) via ``-inf`` masks.

    Additional Args (beyond :class:`AbstractChargingChoiceStrategy`):
        soc_buffer_low: Vehicle-table column: minimum SoC buffer for trip/shift
            energy calculations (fraction of ``batt_cap``).
        soc_buffer_high: Vehicle-table column: target SoC (fraction of
            ``batt_cap``).
        min_soc_charge: Vehicle-table column: minimum fraction of ``batt_cap``
            to charge per session (avoids tiny plug-ins).
        plug_in_and_out_delay_hrs: Vehicle-table column: fixed delay penalty
            for plug-in/out at zero-duration optional stops.
        consumed_kwh_next: Dwell-table column: energy needed for the next trip.
        consumed_kwh_shift: Dwell-table column: total remaining shift energy.
        power_kw_shift_max_remaining: Dwell-table column: highest charging
            power available at any future dwell in this shift.
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
        dwl: np.recarray,
        veh: np.recarray,
        modes: np.recarray,
    ) -> tuple[float, float, int]:
        """Select the utility-maximising charging energy and mode for one dwell.

        Implements the forward-looking utility maximisation described in the
        paper (Model Module 4).  The algorithm proceeds as follows:

        1. Decode the ``modes_avail`` bitmask to a boolean availability vector.
        2. Exit immediately with no charging if no power is available.
        3. Build a ``(N_CHG_OPTS × n_modes)`` energy matrix ``e`` for the six
           options described in the class docstring.
        4. Zero out energy on unavailable modes.
        5. Compute outcomes: final energy, trip success, battery-bounds
           compliance, shift feasibility (all as 0 / ``-inf`` masks).
        6. Compute charging time and resulting delay at this dwell.
        7. Compute counterfactual delay if charging were deferred to the
           highest-power future dwell.
        8. Compute SoC-targeting quadratic utility (penalises deviation from
           ``soc_buffer_high``).
        9. Sum all utility components and find the argmax.
        10. Return the selected energy, net delay change, and mode index.

        Args:
            cur_energy: Current energy (kWh) after subtracting trip consumption.
            dwl: Dwell recarray row; must contain ``modes_avail``, ``dwell_hrs``,
                ``consumed_kwh_next``, ``consumed_kwh_shift``, and
                ``power_kw_shift_max_remaining``.
            veh: Vehicle recarray row; must contain ``batt_cap``,
                ``soc_buffer_low``, ``soc_buffer_high``, ``min_soc_charge``,
                and ``plug_in_and_out_delay_hrs``.
            modes: Mode recarray; must contain ``avail_kw``.

        Returns:
            Three-tuple ``(charge_kwh, delay_hrs, mode_id)`` where
            ``delay_hrs`` is the net change in accumulated delay (positive =
            new delay incurred, negative = delay recovered relative to the
            counterfactual deferral).
        """

        n_modes = modes["avail_kw"].shape[0]
        modes_avail = bits_to_bool_vec(dwl["modes_avail"], n_modes=n_modes)
        powers_flat = modes["avail_kw"] * modes_avail

        # If no charging power is available, then quickly exit
        # Consider adding np.isclose() check if bad effects continue
        if np.max(powers_flat) <= 0.0:
            chg, dly, mode = (0.0, 0.0, int(np.argmin(powers_flat)))
            return (chg, dly, mode)

        # Set some weighting constants
        EXTREME_DELAY_HRS = 10000.0  # More than a year of delay
        BETA_SOC = 0.5

        # Set the shapes of the evaluation arrays
        N_CHG_OPTS = 6
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
        avail_hrs = dwl["dwell_hrs"]
        min_e_chg = veh["min_soc_charge"] * veh["batt_cap"]
        e[1, :] = np.maximum(powers_flat * avail_hrs, min_e_chg)

        # Option 3: Charge for next trip
        buff = veh["soc_buffer_low"] * veh["batt_cap"]
        e[2, :] = np.maximum(dwl["consumed_kwh_next"] + buff - cur_energy, min_e_chg)

        # Option 4: Charge for full shift
        nrg_needed_shift = np.maximum(
            dwl["consumed_kwh_shift"] + buff - cur_energy, min_e_chg
        )
        e[3, :] = nrg_needed_shift

        # Option 5: Charge to optimal SoC, with a minimum level charged to avoid tiny
        #   charging sessions.
        e[4, :] = np.maximum(
            veh["soc_buffer_high"] * veh["batt_cap"] - cur_energy, min_e_chg
        )

        # Option 6: Charge to fill battery
        e[5, :] = np.maximum(veh["batt_cap"] - cur_energy, min_e_chg)

        # Zero out charging on unavailable modes
        e = e * (caster * modes_avail)

        ## Calculate outcomes
        # Energy at end of charging (i.e., "final" energy)
        e_fin = cur_energy + e

        # Trip successfully completed
        # TODO: Implement a soft lower buffer, to deter insufficient charging
        e_next = e_fin - dwl["consumed_kwh_next"]
        trip_succeeds = np.where(e_next > 0, 0, -np.inf)

        # Battery charged within bounds
        batt_respected = np.where(e_fin <= veh["batt_cap"], 0, -np.inf)

        # Shift is feasible, because energy at end of shift is greater than zero
        #   OR there is some opportunity to charge later.
        best_alt_power = dwl["power_kw_shift_max_remaining"]
        e_shift = e_fin - dwl["consumed_kwh_shift"]
        shift_feasible = np.where((e_shift > 0) | (best_alt_power > 0), 0, -np.inf)

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
        if best_alt_power == 0:
            delay_shift = np.ones_like(delay) * EXTREME_DELAY_HRS
        else:
            e_rem = (
                nrg_needed_shift - cur_energy - np.maximum(nrg_needed_shift - e_fin, 0)
            )
            delay_shift = e_rem / best_alt_power

        soc_targeting = (
            -BETA_SOC * (veh["soc_buffer_high"] - e_fin / veh["batt_cap"]) ** 2
        )

        ## Calculate indirect utility and maximize
        v = (
            soc_targeting
            - (delay - delay_shift)
            + trip_succeeds
            + batt_respected
            + shift_feasible
        )
        flat_best_idx = np.argmax(v, axis=None)
        best_idx = (flat_best_idx // n_modes, flat_best_idx % n_modes)
        chg = float(e[best_idx])
        dly = float(time_delta[best_idx])
        mode = int(best_idx[1])
        return (chg, dly, mode)
