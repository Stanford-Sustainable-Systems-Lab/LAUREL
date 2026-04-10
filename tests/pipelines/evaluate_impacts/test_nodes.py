"""Unit tests for evaluate_impacts pipeline nodes.

Covers the memory-efficient accumulator pattern introduced in
``sample_profiles_node``:

- ``_accumulate`` closure: column renaming, region-name restoration, and
  accumulator dict structure.
- ``compress_bootstrap_profiles``: both the accumulator dict path and the
  legacy DataFrame path.
"""

import pandas as pd
import pytest

from laurel.pipelines.evaluate_impacts.nodes import compress_bootstrap_profiles

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_accumulate(
    prof_renamer: dict,
    reg_col: str,
    reg_col_compact: str,
    reg_name_restorer: pd.Series,
    summ_cols: list[str],
    value_cols: list[str],
    boot_profs_accum: dict,
    boot_summs: dict,
    save_debug: bool = False,
    debug_partition: dict | None = None,
):
    """Replicate the ``_accumulate`` closure for unit testing.

    Creates and returns a function with the same logic as the private
    ``_accumulate`` closure defined inside ``sample_profiles_node``.  All
    closure variables are passed explicitly so tests can inspect the mutated
    accumulator dicts after calling the returned function.
    """

    def _accumulate(prof: pd.DataFrame, summ: pd.DataFrame, boot_id: int) -> None:
        prof = prof.rename(columns=prof_renamer)
        prof[reg_col] = prof[reg_col_compact].map(reg_name_restorer)
        prof = prof.drop(columns=[reg_col_compact])
        if save_debug and debug_partition is not None:
            debug_partition[str(boot_id).zfill(4)] = prof.reset_index(drop=True)
        rows_dict = prof.set_index(summ_cols)[value_cols].to_dict("index")
        for group_key, col_vals in rows_dict.items():
            if group_key not in boot_profs_accum:
                boot_profs_accum[group_key] = [[] for _ in value_cols]
            for col_idx, col in enumerate(value_cols):
                boot_profs_accum[group_key][col_idx].append(col_vals[col])
        boot_summs[boot_id] = summ

    return _accumulate


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

REG_COL = "region"
REG_COL_COMPACT = "_region_compact"
TIME_COL = "time_bin"
T0 = pd.Timestamp("1970-01-01 10:00:00")
T1 = pd.Timestamp("1970-01-01 11:00:00")

PROF_RENAMER = {"power_kw_diff": "power_kw"}
REG_NAME_RESTORER = pd.Series({0: "sub_001", 1: "sub_002"})
SUMM_COLS = [REG_COL, TIME_COL]
VALUE_COLS = ["power_kw"]


def _make_prof(
    region_compact: int = 0,
    time_bin: pd.Timestamp = T0,
    power_kw_diff: float = 100.0,
) -> pd.DataFrame:
    """Return a minimal raw profile DataFrame as produced by ``sample_profiles``."""
    return pd.DataFrame(
        {
            REG_COL_COMPACT: [region_compact],
            TIME_COL: [time_bin],
            "power_kw_diff": [power_kw_diff],
        }
    )


def _make_summ(region_compact: int = 0) -> pd.DataFrame:
    """Return a minimal raw summary DataFrame."""
    return pd.DataFrame({REG_COL_COMPACT: [region_compact], "n_dwells": [5]})


# ---------------------------------------------------------------------------
# TestAccumulate
# ---------------------------------------------------------------------------


class TestAccumulate:
    """Tests for the ``_accumulate`` closure logic."""

    @pytest.fixture
    def accum_state(self):
        """Fresh mutable accumulator dicts."""
        return {"boot_profs_accum": {}, "boot_summs": {}}

    @pytest.fixture
    def accumulate(self, accum_state):
        """``_accumulate`` closure wired to a fresh accumulator state."""
        return make_accumulate(
            prof_renamer=PROF_RENAMER,
            reg_col=REG_COL,
            reg_col_compact=REG_COL_COMPACT,
            reg_name_restorer=REG_NAME_RESTORER,
            summ_cols=SUMM_COLS,
            value_cols=VALUE_COLS,
            boot_profs_accum=accum_state["boot_profs_accum"],
            boot_summs=accum_state["boot_summs"],
        )

    def test_column_renaming(self, accumulate, accum_state):
        """Diff columns are renamed to final profile column names in the key."""
        accumulate(_make_prof(power_kw_diff=123.4), _make_summ(), boot_id=0)
        key = (REG_NAME_RESTORER[0], T0)
        assert key in accum_state["boot_profs_accum"]
        assert accum_state["boot_profs_accum"][key][0] == [123.4]

    def test_region_name_restored(self, accumulate, accum_state):
        """Compact int region codes are restored to string region names as group keys."""
        accumulate(
            _make_prof(region_compact=1), _make_summ(region_compact=1), boot_id=0
        )
        key = (REG_NAME_RESTORER[1], T0)
        assert key in accum_state["boot_profs_accum"]
        # Compact-code key must not appear
        assert (1, T0) not in accum_state["boot_profs_accum"]

    def test_accumulator_structure_single_boot(self, accumulate, accum_state):
        """After one call there is one value per value_col under the group key."""
        accumulate(_make_prof(power_kw_diff=50.0), _make_summ(), boot_id=0)
        key = (REG_NAME_RESTORER[0], T0)
        lists = accum_state["boot_profs_accum"][key]
        assert len(lists) == len(VALUE_COLS)
        assert lists[0] == [50.0]

    def test_accumulator_structure_multiple_boots(self, accumulate, accum_state):
        """After N calls the same group key accumulates N values."""
        n = 4
        for i in range(n):
            accumulate(_make_prof(power_kw_diff=float(i * 10)), _make_summ(), boot_id=i)
        key = (REG_NAME_RESTORER[0], T0)
        assert len(accum_state["boot_profs_accum"][key][0]) == n

    def test_multiple_groups_create_distinct_entries(self, accumulate, accum_state):
        """Two distinct (region, time_bin) pairs create separate accumulator entries."""
        accumulate(_make_prof(region_compact=0, time_bin=T0), _make_summ(0), boot_id=0)
        accumulate(_make_prof(region_compact=1, time_bin=T1), _make_summ(1), boot_id=1)
        assert len(accum_state["boot_profs_accum"]) == 2
        assert (REG_NAME_RESTORER[0], T0) in accum_state["boot_profs_accum"]
        assert (REG_NAME_RESTORER[1], T1) in accum_state["boot_profs_accum"]

    def test_value_correctness(self, accumulate, accum_state):
        """The accumulated float values match the source DataFrame values exactly."""
        values = [37.5, 82.0, 150.25]
        for boot_id, v in enumerate(values):
            accumulate(_make_prof(power_kw_diff=v), _make_summ(), boot_id=boot_id)
        key = (REG_NAME_RESTORER[0], T0)
        assert accum_state["boot_profs_accum"][key][0] == values

    def test_boot_summs_stored(self, accumulate, accum_state):
        """``boot_summs[boot_id]`` stores the raw summary DataFrame unchanged."""
        summ = _make_summ()
        accumulate(_make_prof(), summ, boot_id=7)
        pd.testing.assert_frame_equal(accum_state["boot_summs"][7], summ)

    def test_debug_disabled_by_default(self, accum_state):
        """When ``save_debug=False``, the debug partition is not populated."""
        debug_partition = {}
        fn = make_accumulate(
            prof_renamer=PROF_RENAMER,
            reg_col=REG_COL,
            reg_col_compact=REG_COL_COMPACT,
            reg_name_restorer=REG_NAME_RESTORER,
            summ_cols=SUMM_COLS,
            value_cols=VALUE_COLS,
            boot_profs_accum=accum_state["boot_profs_accum"],
            boot_summs=accum_state["boot_summs"],
            save_debug=False,
            debug_partition=debug_partition,
        )
        fn(_make_prof(), _make_summ(), boot_id=0)
        assert len(debug_partition) == 0

    def test_debug_enabled_populates_partition(self, accum_state):
        """When ``save_debug=True``, the renamed profile is stored in the debug partition."""
        debug_partition = {}
        fn = make_accumulate(
            prof_renamer=PROF_RENAMER,
            reg_col=REG_COL,
            reg_col_compact=REG_COL_COMPACT,
            reg_name_restorer=REG_NAME_RESTORER,
            summ_cols=SUMM_COLS,
            value_cols=VALUE_COLS,
            boot_profs_accum=accum_state["boot_profs_accum"],
            boot_summs=accum_state["boot_summs"],
            save_debug=True,
            debug_partition=debug_partition,
        )
        fn(_make_prof(power_kw_diff=99.9), _make_summ(), boot_id=3)
        assert "0003" in debug_partition
        df = debug_partition["0003"]
        assert "power_kw" in df.columns
        assert REG_COL_COMPACT not in df.columns
        assert df["power_kw"].iloc[0] == 99.9

    def test_multi_value_cols_accumulated_independently(self, accum_state):
        """Multiple value columns each get their own inner list in the accumulator."""
        value_cols = ["power_kw", "energy_kwh"]
        prof_renamer = {"power_kw_diff": "power_kw", "energy_kwh_diff": "energy_kwh"}
        fn = make_accumulate(
            prof_renamer=prof_renamer,
            reg_col=REG_COL,
            reg_col_compact=REG_COL_COMPACT,
            reg_name_restorer=REG_NAME_RESTORER,
            summ_cols=SUMM_COLS,
            value_cols=value_cols,
            boot_profs_accum=accum_state["boot_profs_accum"],
            boot_summs=accum_state["boot_summs"],
        )
        prof = pd.DataFrame(
            {
                REG_COL_COMPACT: [0],
                TIME_COL: [T0],
                "power_kw_diff": [100.0],
                "energy_kwh_diff": [200.0],
            }
        )
        fn(prof, _make_summ(), boot_id=0)
        key = (REG_NAME_RESTORER[0], T0)
        lists = accum_state["boot_profs_accum"][key]
        assert len(lists) == 2
        assert lists[0] == [100.0]
        assert lists[1] == [200.0]


# ---------------------------------------------------------------------------
# TestCompressBootstrapProfiles
# ---------------------------------------------------------------------------


class TestCompressBootstrapProfiles:
    """Tests for ``compress_bootstrap_profiles`` — both accumulator and DataFrame paths."""

    @pytest.fixture
    def base_params(self):
        """Minimal params dict compatible with both paths."""
        return {
            "time_col": "time_bin",
            "discrete_freq": "1h",
            "slice_freq": "1D",
            "bootstrap_id_col": "boot_id",
            "n_bootstraps": 5,
            "quantiles": [0.1, 0.5, 0.9],
        }

    @pytest.fixture
    def base_pcols(self):
        """Minimal pcols dict compatible with both paths."""
        return {
            "group_cols": ["region"],
            "timezone_col": "tz",
            "profile_cols": {"power": "power_kw"},
            "diff_cols": {"power": "power_kw_diff"},
        }

    # --- Accumulator path ---

    def _make_accum(self, regions, timestamps, value_lists, n_boots):
        """Build an accumulator dict for ``compress_bootstrap_profiles``."""
        accum = {}
        for reg, ts, vals in zip(regions, timestamps, value_lists):
            accum[(reg, ts)] = [vals]
        return accum

    def test_accumulator_path_returns_dataframe(self, base_params, base_pcols):
        """Accumulator dict input returns a DataFrame."""
        accum = {("sub_001", T0): [[100.0, 200.0, 150.0, 180.0, 120.0]]}
        result = compress_bootstrap_profiles(accum, base_params, base_pcols)
        assert isinstance(result, pd.DataFrame)

    def test_accumulator_path_output_columns(self, base_params, base_pcols):
        """Output columns follow the ``{value_col}_{quantile}`` naming convention."""
        accum = {("sub_001", T0): [[100.0, 200.0, 150.0, 180.0, 120.0]]}
        result = compress_bootstrap_profiles(accum, base_params, base_pcols)
        expected_cols = [f"power_kw_{q}" for q in base_params["quantiles"]]
        assert list(result.columns) == expected_cols

    def test_accumulator_path_index_names(self, base_params, base_pcols):
        """Output index names match ``group_cols + [time_col]``."""
        accum = {("sub_001", T0): [[100.0, 200.0, 150.0, 180.0, 120.0]]}
        result = compress_bootstrap_profiles(accum, base_params, base_pcols)
        assert list(result.index.names) == ["region", "time_bin"]

    def test_accumulator_path_zero_padding(self, base_params, base_pcols):
        """A group present in only 1 of 5 bootstraps has its median quantile at zero.

        ``n_bootstraps=5`` and 1 observed value means 4 zero-padded slots.
        The 50th percentile of [0, 0, 0, 0, x] is 0.
        """
        accum = {("sub_001", T0): [[500.0]]}  # observed in 1 of 5 bootstraps
        result = compress_bootstrap_profiles(accum, base_params, base_pcols)
        assert result.loc[("sub_001", T0), "power_kw_0.5"] == 0.0

    def test_accumulator_path_all_observed_positive(self, base_params, base_pcols):
        """A group observed in every bootstrap has a positive median quantile."""
        accum = {("sub_001", T0): [[100.0, 200.0, 150.0, 180.0, 120.0]]}
        result = compress_bootstrap_profiles(accum, base_params, base_pcols)
        assert result.loc[("sub_001", T0), "power_kw_0.5"] > 0.0

    def test_accumulator_path_multi_value_cols(self, base_params, base_pcols):
        """Two value columns produce 2 × len(quantiles) output columns."""
        pcols = {
            **base_pcols,
            "profile_cols": {"power": "power_kw", "energy": "energy_kwh"},
        }
        # inner lists: one per value_col in profile_cols order
        accum = {
            ("sub_001", T0): [
                [100.0, 200.0, 150.0, 180.0, 120.0],
                [10.0, 20.0, 15.0, 18.0, 12.0],
            ]
        }
        result = compress_bootstrap_profiles(accum, base_params, pcols)
        n_q = len(base_params["quantiles"])
        assert len(result.columns) == 2 * n_q

    def test_accumulator_path_multiple_groups(self, base_params, base_pcols):
        """Multiple (region, time_bin) groups each appear as a separate index row."""
        accum = {
            ("sub_001", T0): [[100.0, 120.0, 110.0, 130.0, 140.0]],
            ("sub_002", T1): [[50.0, 60.0, 55.0, 65.0, 70.0]],
        }
        result = compress_bootstrap_profiles(accum, base_params, base_pcols)
        assert len(result) == 2
        assert ("sub_001", T0) in result.index
        assert ("sub_002", T1) in result.index

    def test_accumulator_path_quantile_ordering(self, base_params, base_pcols):
        """Lower quantile values are <= higher quantile values for the same group."""
        accum = {("sub_001", T0): [[10.0, 50.0, 30.0, 80.0, 60.0]]}
        result = compress_bootstrap_profiles(accum, base_params, base_pcols)
        row = result.loc[("sub_001", T0)]
        q_vals = [row[f"power_kw_{q}"] for q in sorted(base_params["quantiles"])]
        assert q_vals == sorted(q_vals)

    # --- DataFrame path (legacy) ---

    def _make_profs_df(self, n_boots: int = 3) -> pd.DataFrame:
        """Minimal profile DataFrame for the legacy DataFrame path.

        Uses ``slice_time_relative`` as the time column to match the real
        parameter configuration, and ``no_time_zone`` as the timezone value
        to match ``possible_tzs=["no_time_zone"]`` used inside the function.
        Timestamps are within the 1-day window anchored at ``pd.Timestamp(0)``.
        """
        ts = pd.Timestamp("1970-01-01 10:00:00")  # within [T(0), T(0)+1D]
        return pd.DataFrame(
            {
                "slice_time_relative": [ts] * n_boots,
                "region": ["sub_001"] * n_boots,
                "boot_id": list(range(n_boots)),
                "power_kw": [100.0 + i * 10 for i in range(n_boots)],
                "tz": ["no_time_zone"] * n_boots,
            }
        )

    def _make_df_params(self, n_boots: int = 3) -> dict:
        return {
            "time_col": "slice_time_relative",
            "discrete_freq": "1h",
            "slice_freq": "1D",
            "bootstrap_id_col": "boot_id",
            "n_bootstraps": n_boots,
            "quantiles": [0.1, 0.5, 0.9],
        }

    def _make_df_pcols(self) -> dict:
        return {
            "group_cols": ["region"],
            "timezone_col": "tz",
            "profile_cols": {"power": "power_kw"},
            "diff_cols": {"power": "power_kw_diff"},
        }

    def test_dataframe_path_returns_dataframe(self):
        """DataFrame input returns a DataFrame (legacy path is active)."""
        result = compress_bootstrap_profiles(
            self._make_profs_df(), self._make_df_params(), self._make_df_pcols()
        )
        assert isinstance(result, pd.DataFrame)

    def test_dataframe_path_output_columns(self):
        """DataFrame path produces columns with the same ``{col}_{q}`` schema."""
        params = self._make_df_params()
        result = compress_bootstrap_profiles(
            self._make_profs_df(), params, self._make_df_pcols()
        )
        expected_cols = [f"power_kw_{q}" for q in params["quantiles"]]
        assert list(result.columns) == expected_cols

    # --- Schema consistency between paths ---

    def test_both_paths_same_schema(self):
        """Accumulator and DataFrame paths produce identical column and index schemas."""
        n_boots = 3
        ts = pd.Timestamp("1970-01-01 10:00:00")
        params_df = self._make_df_params(n_boots)
        pcols_df = self._make_df_pcols()

        # Accumulator path
        accum_params = {
            "time_col": "slice_time_relative",
            "n_bootstraps": n_boots,
            "quantiles": params_df["quantiles"],
        }
        accum_pcols = {
            "group_cols": ["region"],
            "profile_cols": {"power": "power_kw"},
        }
        accum = {("sub_001", ts): [[100.0, 110.0, 120.0]]}
        result_accum = compress_bootstrap_profiles(accum, accum_params, accum_pcols)

        # DataFrame path
        result_df = compress_bootstrap_profiles(
            self._make_profs_df(n_boots), params_df, pcols_df
        )

        assert list(result_accum.columns) == list(result_df.columns)
        assert result_accum.index.names == result_df.index.names
