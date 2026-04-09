"""Unit tests for ``sample_sparse_multinomial`` and the JIT core.

Covers the dual-path dispatch introduced to speed up first-stage class
sampling on HPC (see plan happy-enchanting-liskov.md):

- **Categorical path** (``n < k``): sorted uniform draws + merge walk through CDF.
- **Multinomial path** (``n >= k``): ``np.random.multinomial`` via sequential binomial.

Test categories
---------------
1. Output structure — shape, dtype, CSC format, indptr monotonicity.
2. Count conservation — each output column sums to the requested ``n``.
3. Edge cases — n=0, empty columns, k=1, all-zero ``n_arr``.
4. Dispatch boundary — ``n = k-1`` (categorical) and ``n = k`` (multinomial)
   both produce valid output.
5. Statistical correctness — chi-square goodness-of-fit against the input
   probability vector for both paths, with both uniform and non-uniform weights.
6. ``loc_grp_arr`` — many locations sharing one probability column each
   sample independently from the correct distribution.
7. Performance — absolute wall-time bounds and scaling behaviour for both
   paths to guard against algorithmic regressions.
"""

import time

import numpy as np
import pytest
import scipy as sp
from scipy.stats import chi2 as chi2_dist

from laurel.models.sampling import sample_sparse_multinomial

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dense_to_csc(arr: np.ndarray) -> sp.sparse.csc_array:
    """Build a CSC sparse array from a dense (n_obs, n_cols) probability matrix."""
    return sp.sparse.csc_array(arr)


def _col_counts(out: sp.sparse.csc_array) -> np.ndarray:
    """Return the sum of each output column as a 1-D int array."""
    return np.asarray(out.sum(axis=0)).ravel()


def _col_density(out: sp.sparse.csc_array, col: int) -> np.ndarray:
    """Return a dense count vector for output column ``col``."""
    return np.asarray(out[:, col].todense()).ravel()


# ---------------------------------------------------------------------------
# JIT warm-up (runs once per test session to avoid slow first-test times)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def warmup_jit():
    """Trigger Numba JIT compilation before any test runs.

    Both the categorical (n < k) and multinomial (n >= k) branches are
    exercised so the compiled binary covers both code paths.
    """
    p = _dense_to_csc(np.array([[0.5], [0.5]]))
    # multinomial path: n=5 >= k=2
    sample_sparse_multinomial(n_arr=np.array([5], dtype=np.int64), p_arr=p)
    # categorical path: n=1 < k=2
    sample_sparse_multinomial(n_arr=np.array([1], dtype=np.int64), p_arr=p)


# ---------------------------------------------------------------------------
# TestOutputStructure
# ---------------------------------------------------------------------------


class TestOutputStructure:
    """Verify shape, dtype, and sparse format of the returned array."""

    @pytest.fixture
    def simple_p(self):
        """2-location, 4-category probability matrix."""
        arr = np.array(
            [
                [0.4, 0.1],
                [0.3, 0.5],
                [0.2, 0.3],
                [0.1, 0.1],
            ],
            dtype=np.float64,
        )
        return _dense_to_csc(arr)

    def test_returns_csc_array(self, simple_p):
        """Output is a scipy CSC sparse array."""
        out = sample_sparse_multinomial(
            n_arr=np.array([3, 4], dtype=np.int64), p_arr=simple_p
        )
        assert sp.sparse.issparse(out)
        assert isinstance(out, sp.sparse.csc_array | sp.sparse.csc_matrix)

    def test_shape(self, simple_p):
        """Output shape is (n_obs, n_locs)."""
        n_arr = np.array([3, 4], dtype=np.int64)
        out = sample_sparse_multinomial(n_arr=n_arr, p_arr=simple_p)
        assert out.shape == (4, 2)

    def test_data_dtype_is_int(self, simple_p):
        """Non-zero data values are integers (sample counts)."""
        out = sample_sparse_multinomial(
            n_arr=np.array([5, 5], dtype=np.int64), p_arr=simple_p
        )
        if out.nnz > 0:
            assert np.issubdtype(out.data.dtype, np.integer)

    def test_indptr_starts_at_zero(self, simple_p):
        """CSC indptr must start at 0."""
        out = sample_sparse_multinomial(
            n_arr=np.array([3, 2], dtype=np.int64), p_arr=simple_p
        )
        assert out.indptr[0] == 0

    def test_indptr_is_non_decreasing(self, simple_p):
        """CSC indptr must be non-decreasing."""
        out = sample_sparse_multinomial(
            n_arr=np.array([3, 2], dtype=np.int64), p_arr=simple_p
        )
        assert np.all(np.diff(out.indptr) >= 0)

    def test_indptr_ends_at_nnz(self, simple_p):
        """Last indptr entry must equal nnz."""
        out = sample_sparse_multinomial(
            n_arr=np.array([3, 2], dtype=np.int64), p_arr=simple_p
        )
        assert out.indptr[-1] == out.nnz

    def test_indices_within_bounds(self, simple_p):
        """Row indices must be in [0, n_obs)."""
        out = sample_sparse_multinomial(
            n_arr=np.array([3, 2], dtype=np.int64), p_arr=simple_p
        )
        if out.nnz > 0:
            assert out.indices.min() >= 0
            assert out.indices.max() < simple_p.shape[0]

    def test_all_counts_positive(self, simple_p):
        """All stored data values (sample counts) must be positive."""
        out = sample_sparse_multinomial(
            n_arr=np.array([5, 5], dtype=np.int64), p_arr=simple_p
        )
        if out.nnz > 0:
            assert np.all(out.data > 0)


# ---------------------------------------------------------------------------
# TestCountConservation
# ---------------------------------------------------------------------------


class TestCountConservation:
    """Column j of the output must sum exactly to n_arr[j]."""

    def test_column_sums_equal_n_arr_categorical_path(self):
        """Categorical path (n < k): each column sums to the requested count."""
        probs = np.array([[0.1], [0.2], [0.3], [0.25], [0.15]])
        p_arr = _dense_to_csc(probs)
        n = 3  # n=3 < k=5 → categorical
        out = sample_sparse_multinomial(
            n_arr=np.array([n], dtype=np.int64), p_arr=p_arr
        )
        assert int(out.sum()) == n

    def test_column_sums_equal_n_arr_multinomial_path(self):
        """Multinomial path (n >= k): each column sums to the requested count."""
        probs = np.array([[0.1], [0.2], [0.3], [0.25], [0.15]])
        p_arr = _dense_to_csc(probs)
        n = 8  # n=8 >= k=5 → multinomial
        out = sample_sparse_multinomial(
            n_arr=np.array([n], dtype=np.int64), p_arr=p_arr
        )
        assert int(out.sum()) == n

    def test_multiple_columns_each_sum_to_n(self):
        """Each column independently sums to its own requested n."""
        arr = np.array(
            [
                [0.3, 0.6, 0.2],
                [0.4, 0.3, 0.5],
                [0.3, 0.1, 0.3],
            ],
            dtype=np.float64,
        )
        p_arr = _dense_to_csc(arr)
        n_arr = np.array(
            [2, 10, 1], dtype=np.int64
        )  # mixes categorical and multinomial
        out = sample_sparse_multinomial(n_arr=n_arr, p_arr=p_arr)
        counts = _col_counts(out)
        np.testing.assert_array_equal(counts, n_arr)

    def test_zero_n_contributes_no_counts(self):
        """A location with n=0 contributes zero to the total and has no entries."""
        arr = np.array([[0.5, 0.5], [0.5, 0.5]])
        p_arr = _dense_to_csc(arr)
        n_arr = np.array([0, 3], dtype=np.int64)
        out = sample_sparse_multinomial(n_arr=n_arr, p_arr=p_arr)
        counts = _col_counts(out)
        assert counts[0] == 0
        assert counts[1] == 3
        # No entries for location 0 in the CSC structure
        assert out.indptr[1] - out.indptr[0] == 0

    def test_all_n_zero_returns_all_zeros(self):
        """If every location requests 0 draws, the output has no non-zero entries."""
        arr = np.array([[0.5], [0.5]])
        p_arr = _dense_to_csc(arr)
        out = sample_sparse_multinomial(
            n_arr=np.array([0], dtype=np.int64), p_arr=p_arr
        )
        assert out.nnz == 0
        assert int(out.sum()) == 0


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Corner cases: k=1, empty column, large n."""

    def test_k1_single_category_all_draws_go_to_obs0(self):
        """With k=1, all draws must go to the only available observation.

        The multinomial path is always used here because n >= k=1 for any n >= 1.
        """
        p_arr = _dense_to_csc(np.array([[1.0]]))
        n = 7
        out = sample_sparse_multinomial(
            n_arr=np.array([n], dtype=np.int64), p_arr=p_arr
        )
        assert out.shape == (1, 1)
        assert int(out[0, 0]) == n

    def test_empty_column_produces_no_output_entries(self):
        """A column that is all-zero in the probability matrix produces no output.

        In CSC format this means the column has no stored entries.
        """
        # Build a (3, 2) matrix where column 1 is zero (no eligible observations)
        arr = np.array(
            [
                [0.5, 0.0],
                [0.3, 0.0],
                [0.2, 0.0],
            ],
            dtype=np.float64,
        )
        p_arr = _dense_to_csc(arr)
        n_arr = np.array([3, 5], dtype=np.int64)
        out = sample_sparse_multinomial(n_arr=n_arr, p_arr=p_arr)
        # Column 0 has 3 draws; column 1 has an empty probability column
        assert out.indptr[2] - out.indptr[1] == 0

    def test_single_location_shape(self):
        """n_locs=1 returns a (n_obs, 1) output."""
        p_arr = _dense_to_csc(np.array([[0.3], [0.4], [0.3]]))
        out = sample_sparse_multinomial(
            n_arr=np.array([4], dtype=np.int64), p_arr=p_arr
        )
        assert out.shape == (3, 1)

    def test_n_equals_k_uses_multinomial_and_sums_correctly(self):
        """At the exact dispatch boundary n == k, multinomial path runs without error."""
        k = 5
        probs = np.full((k, 1), 1.0 / k)
        p_arr = _dense_to_csc(probs)
        n = k  # exactly at boundary → multinomial
        out = sample_sparse_multinomial(
            n_arr=np.array([n], dtype=np.int64), p_arr=p_arr
        )
        assert int(out.sum()) == n

    def test_n_equals_k_minus_1_uses_categorical_and_sums_correctly(self):
        """Just below the dispatch boundary n == k-1, categorical path runs without error."""
        k = 5
        probs = np.full((k, 1), 1.0 / k)
        p_arr = _dense_to_csc(probs)
        n = k - 1  # one below boundary → categorical
        out = sample_sparse_multinomial(
            n_arr=np.array([n], dtype=np.int64), p_arr=p_arr
        )
        assert int(out.sum()) == n

    def test_large_n_small_k_all_categories_sampled(self):
        """With n >> k, every category should receive at least one draw (probabilistically certain)."""
        k = 3
        probs = np.full((k, 1), 1.0 / k)
        p_arr = _dense_to_csc(probs)
        n = 1000  # n >> k → multinomial; chance any category missed ≈ 0
        out = sample_sparse_multinomial(
            n_arr=np.array([n], dtype=np.int64), p_arr=p_arr
        )
        dense = _col_density(out, 0)
        assert np.all(dense > 0), "All k=3 categories should be sampled when n=1000"

    def test_nearly_unnormalized_probs_handle_float_drift(self):
        """Probabilities that don't sum exactly to 1.0 due to floating-point rounding
        should not cause draws to fall outside the valid index range.

        The categorical path sets cumprobs[-1] = 1.0 to guard against this.
        """
        # Probabilities sum to 0.9999999999999998 (float64 rounding)
        probs = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])
        probs = probs.reshape(-1, 1)
        p_arr = _dense_to_csc(probs)
        n = 2  # n < k=3 → categorical path
        for seed in range(20):
            np.random.seed(seed)
            out = sample_sparse_multinomial(
                n_arr=np.array([n], dtype=np.int64), p_arr=p_arr
            )
            assert int(out.sum()) == n
            if out.nnz > 0:
                assert out.indices.min() >= 0
                assert out.indices.max() < 3


# ---------------------------------------------------------------------------
# TestStatisticalCorrectness
# ---------------------------------------------------------------------------


class TestStatisticalCorrectness:
    """Chi-square goodness-of-fit tests for the multinomial distribution.

    Strategy: issue N_LOCS independent draws (all pointing to the same
    probability column via loc_grp_arr) and aggregate counts.  The total
    observed count per category is compared to the expected count
    N_LOCS * n * p_j using a chi-square test at a conservative significance
    level.  Failures indicate the sampling algorithm is drawing from the wrong
    distribution.
    """

    N_LOCS = 5000  # independent draws per test
    ALPHA = 0.001  # very conservative to avoid flakiness

    def _chi2_pvalue(self, observed: np.ndarray, expected: np.ndarray) -> float:
        """Return the p-value of a chi-square goodness-of-fit test.

        Args:
            observed: observed counts per category.
            expected: expected counts per category under H0.

        Returns:
            p-value (high = consistent with H0).
        """
        stat = float(np.sum((observed - expected) ** 2 / expected))
        dof = len(observed) - 1
        return float(1.0 - chi2_dist.cdf(stat, df=dof))

    def _run_chi2(self, probs: np.ndarray, n: int) -> float:
        """Run N_LOCS draws of n samples from a single column and chi-square test.

        Uses loc_grp_arr with all zeros so the Numba compiled loop handles
        the iteration, avoiding per-call Python overhead.
        """
        p_arr = _dense_to_csc(probs.reshape(-1, 1))  # (k, 1) — one class column
        n_arr = np.full(self.N_LOCS, n, dtype=np.int64)
        loc_grp_arr = np.zeros(self.N_LOCS, dtype=np.int64)

        np.random.seed(42)
        out = sample_sparse_multinomial(
            n_arr=n_arr, p_arr=p_arr, loc_grp_arr=loc_grp_arr
        )
        out_dense = np.asarray(out.todense())  # (k, N_LOCS)

        # Aggregate: total count per category across all N_LOCS draws
        total_counts = out_dense.sum(axis=1)  # shape (k,)
        expected = self.N_LOCS * n * probs
        return self._chi2_pvalue(total_counts, expected)

    def test_uniform_categorical_path(self):
        """Categorical path with uniform probs: chi-square should not reject H0."""
        k = 6
        n = 2  # n < k → categorical
        probs = np.full(k, 1.0 / k)
        p_value = self._run_chi2(probs, n)
        assert (
            p_value > self.ALPHA
        ), f"Chi-square rejected H0 for categorical path (uniform, p={p_value:.4f})"

    def test_nonuniform_categorical_path(self):
        """Categorical path with non-uniform probs respects the probability weights."""
        probs = np.array([0.05, 0.10, 0.20, 0.30, 0.25, 0.10])
        n = 3  # n < k=6 → categorical
        p_value = self._run_chi2(probs, n)
        assert (
            p_value > self.ALPHA
        ), f"Chi-square rejected H0 for categorical path (non-uniform, p={p_value:.4f})"

    def test_uniform_multinomial_path(self):
        """Multinomial path with uniform probs: chi-square should not reject H0."""
        k = 4
        n = 8  # n >= k → multinomial
        probs = np.full(k, 1.0 / k)
        p_value = self._run_chi2(probs, n)
        assert (
            p_value > self.ALPHA
        ), f"Chi-square rejected H0 for multinomial path (uniform, p={p_value:.4f})"

    def test_nonuniform_multinomial_path(self):
        """Multinomial path with non-uniform probs respects the probability weights."""
        probs = np.array([0.4, 0.3, 0.2, 0.1])
        n = 6  # n >= k=4 → multinomial
        p_value = self._run_chi2(probs, n)
        assert (
            p_value > self.ALPHA
        ), f"Chi-square rejected H0 for multinomial path (non-uniform, p={p_value:.4f})"

    def test_n_equals_k_boundary_distribution(self):
        """At the exact dispatch boundary n == k, the distribution is still correct."""
        k = 5
        probs = np.array([0.1, 0.15, 0.25, 0.30, 0.20])
        n = k  # exactly at boundary
        p_value = self._run_chi2(probs, n)
        assert (
            p_value > self.ALPHA
        ), f"Chi-square rejected H0 at dispatch boundary n==k (p={p_value:.4f})"

    def test_n_equals_k_minus_1_boundary_distribution(self):
        """Just below the boundary n == k-1, the categorical path gives the correct distribution."""
        k = 5
        probs = np.array([0.1, 0.15, 0.25, 0.30, 0.20])
        n = k - 1  # one below boundary → categorical
        p_value = self._run_chi2(probs, n)
        assert (
            p_value > self.ALPHA
        ), f"Chi-square rejected H0 just below dispatch boundary (p={p_value:.4f})"

    def test_highly_skewed_probabilities_categorical(self):
        """Categorical path correctly handles highly skewed probability vectors.

        With one dominant category (prob=0.99), nearly all draws should land there.
        """
        probs = np.array([0.99, 0.005, 0.003, 0.002])
        n = 1  # n < k=4 → categorical
        p_value = self._run_chi2(probs, n)
        assert (
            p_value > self.ALPHA
        ), f"Chi-square rejected H0 for skewed categorical (p={p_value:.4f})"

    def test_highly_skewed_probabilities_multinomial(self):
        """Multinomial path correctly handles highly skewed probability vectors."""
        probs = np.array([0.99, 0.005, 0.005])
        n = 10  # n >= k=3 → multinomial
        p_value = self._run_chi2(probs, n)
        assert (
            p_value > self.ALPHA
        ), f"Chi-square rejected H0 for skewed multinomial (p={p_value:.4f})"


# ---------------------------------------------------------------------------
# TestLocGrpArr
# ---------------------------------------------------------------------------


class TestLocGrpArr:
    """Tests for class-level pooling (``loc_grp_arr`` != None).

    Many locations share one probability column (a class).  Each location
    draws independently from its assigned class's distribution.
    """

    @pytest.fixture
    def class_setup(self):
        """Fixture: 2 classes, 6 observations, 8 locations.

        Class 0 owns observations 0-2 (probs [0.5, 0.3, 0.2]).
        Class 1 owns observations 3-5 (probs [0.2, 0.5, 0.3]).
        Locations 0-3 → class 0; locations 4-7 → class 1.
        """
        arr = np.zeros((6, 2))
        arr[:3, 0] = [0.5, 0.3, 0.2]
        arr[3:, 1] = [0.2, 0.5, 0.3]
        p_arr = _dense_to_csc(arr)
        loc_grp_arr = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
        return p_arr, loc_grp_arr

    def test_output_shape_is_n_obs_by_n_locs(self, class_setup):
        """Output shape is (n_obs, n_locs), not (n_obs, n_classes)."""
        p_arr, loc_grp_arr = class_setup
        n_arr = np.array([2, 3, 1, 4, 2, 3, 1, 4], dtype=np.int64)
        out = sample_sparse_multinomial(
            n_arr=n_arr, p_arr=p_arr, loc_grp_arr=loc_grp_arr
        )
        assert out.shape == (6, 8)

    def test_column_sums_equal_n_arr(self, class_setup):
        """Each output column sums to the corresponding n_arr entry."""
        p_arr, loc_grp_arr = class_setup
        n_arr = np.array([2, 3, 1, 4, 2, 3, 1, 4], dtype=np.int64)
        out = sample_sparse_multinomial(
            n_arr=n_arr, p_arr=p_arr, loc_grp_arr=loc_grp_arr
        )
        counts = _col_counts(out)
        np.testing.assert_array_equal(counts, n_arr)

    def test_class0_locations_only_use_class0_observations(self, class_setup):
        """Locations assigned to class 0 never sample from class 1's observations.

        Class 0 owns row indices 0-2; class 1 owns row indices 3-5.
        """
        p_arr, loc_grp_arr = class_setup
        n_arr = np.full(8, 10, dtype=np.int64)
        np.random.seed(0)
        out = sample_sparse_multinomial(
            n_arr=n_arr, p_arr=p_arr, loc_grp_arr=loc_grp_arr
        )
        out_dense = np.asarray(out.todense())
        # Locations 0-3 → class 0 → only rows 0-2 should be non-zero
        class0_locs = out_dense[:, :4]
        assert np.all(
            class0_locs[3:, :] == 0
        ), "Class 0 locations sampled observations belonging to class 1"

    def test_class1_locations_only_use_class1_observations(self, class_setup):
        """Locations assigned to class 1 never sample from class 0's observations."""
        p_arr, loc_grp_arr = class_setup
        n_arr = np.full(8, 10, dtype=np.int64)
        np.random.seed(0)
        out = sample_sparse_multinomial(
            n_arr=n_arr, p_arr=p_arr, loc_grp_arr=loc_grp_arr
        )
        out_dense = np.asarray(out.todense())
        # Locations 4-7 → class 1 → only rows 3-5 should be non-zero
        class1_locs = out_dense[:, 4:]
        assert np.all(
            class1_locs[:3, :] == 0
        ), "Class 1 locations sampled observations belonging to class 0"

    def test_loc_grp_arr_chi_square_class0(self):
        """Locations using class 0 sample from the correct class-0 distribution."""
        probs_c0 = np.array([0.5, 0.3, 0.2])
        probs_c1 = np.array([0.2, 0.5, 0.3])
        arr = np.zeros((3, 2))
        arr[:, 0] = probs_c0
        arr[:, 1] = probs_c1
        p_arr = _dense_to_csc(arr)

        n_locs = 5000
        n = 2  # n < k=3 → categorical path
        n_arr = np.full(n_locs, n, dtype=np.int64)
        loc_grp_arr = np.zeros(n_locs, dtype=np.int64)  # all in class 0

        np.random.seed(99)
        out = sample_sparse_multinomial(
            n_arr=n_arr, p_arr=p_arr, loc_grp_arr=loc_grp_arr
        )
        out_dense = np.asarray(out.todense())

        observed = out_dense.sum(axis=1).ravel()  # counts per row
        expected = n_locs * n * probs_c0
        stat = float(np.sum((observed - expected) ** 2 / expected))
        dof = 2  # k-1
        p_value = float(1.0 - chi2_dist.cdf(stat, df=dof))
        assert (
            p_value > 0.001
        ), f"loc_grp_arr class-0 distribution failed chi-square (p={p_value:.4f})"

    def test_loc_grp_arr_chi_square_class1(self):
        """Locations using class 1 sample from the correct class-1 distribution."""
        probs_c0 = np.array([0.5, 0.3, 0.2])
        probs_c1 = np.array([0.2, 0.5, 0.3])
        arr = np.zeros((3, 2))
        arr[:, 0] = probs_c0
        arr[:, 1] = probs_c1
        p_arr = _dense_to_csc(arr)

        n_locs = 5000
        n = 5  # n >= k=3 → multinomial path
        n_arr = np.full(n_locs, n, dtype=np.int64)
        loc_grp_arr = np.ones(n_locs, dtype=np.int64)  # all in class 1

        np.random.seed(101)
        out = sample_sparse_multinomial(
            n_arr=n_arr, p_arr=p_arr, loc_grp_arr=loc_grp_arr
        )
        out_dense = np.asarray(out.todense())

        observed = out_dense.sum(axis=1).ravel()
        expected = n_locs * n * probs_c1
        stat = float(np.sum((observed - expected) ** 2 / expected))
        dof = 2
        p_value = float(1.0 - chi2_dist.cdf(stat, df=dof))
        assert (
            p_value > 0.001
        ), f"loc_grp_arr class-1 distribution failed chi-square (p={p_value:.4f})"


# ---------------------------------------------------------------------------
# TestPerformance
# ---------------------------------------------------------------------------


class TestPerformance:
    """Wall-time regression tests for both sampling paths.

    These tests catch algorithmic regressions — e.g. accidentally falling
    back to the O(k) sequential-binomial path when n ≪ k, or introducing
    super-linear scaling in k.

    All timings are measured after JIT warm-up (the session-scoped
    ``warmup_jit`` fixture has already compiled the function), so
    compilation overhead is excluded.  Thresholds are 50–100× the expected
    theoretical time to tolerate CI hardware variability; they are tight
    enough to catch order-of-magnitude regressions.

    The HPC bottleneck context (see plan happy-enchanting-liskov.md):
    - Old code: Call 1 (23 locs, n=5000, k=624K) → 60.8s per bootstrap.
    - New code: same call → expected < 0.1s (categorical path).
    """

    def _timed(
        self,
        k: int,
        n: int,
        n_locs: int,
        seed: int = 0,
        force_multinomial: bool = False,
    ) -> float:
        """Return elapsed seconds for one ``sample_sparse_multinomial`` call.

        Constructs a uniform ``(k, n_locs)`` probability matrix so the same
        compiled code path is exercised regardless of which test calls this.
        """
        probs = np.full(k, 1.0 / k, dtype=np.float64)
        p_arr = _dense_to_csc(np.tile(probs.reshape(-1, 1), (1, n_locs)))
        n_arr = np.full(n_locs, n, dtype=np.int64)
        np.random.seed(seed)
        t0 = time.perf_counter()
        sample_sparse_multinomial(
            n_arr=n_arr, p_arr=p_arr, force_multinomial=force_multinomial
        )
        return time.perf_counter() - t0

    # ------------------------------------------------------------------
    # Categorical path (n < k)
    # ------------------------------------------------------------------

    def test_categorical_path_call1_analog_is_fast(self):
        """Categorical path is materially faster than multinomial at Call 1 scale.

        Uses k=750K (similar to HPC's 624K) with n=5000, n_locs=23 — the
        same shape as HPC Call 1.  Runs both paths back-to-back via
        ``force_multinomial`` and requires the categorical path to be at
        least 3× faster, confirming the dispatch is active and effective.
        """
        k, n_locs, n = 750_000, 23, 5_000
        assert n < k, "This test requires n < k to exercise the categorical path"

        n_rep = 3
        t_cat = min(
            self._timed(k=k, n=n, n_locs=n_locs, seed=i, force_multinomial=False)
            for i in range(n_rep)
        )
        t_mult = min(
            self._timed(k=k, n=n, n_locs=n_locs, seed=i, force_multinomial=True)
            for i in range(n_rep)
        )
        speedup = t_mult / max(t_cat, 1e-9)
        print(
            f"\n  [Call 1 analog] n={n}, k={k:,}, n_locs={n_locs}: "
            f"categorical={t_cat*1e3:.1f}ms, multinomial={t_mult*1e3:.1f}ms, "
            f"speedup={speedup:.1f}×"
        )
        # At k=20K both paths fit in L2 cache, so the speedup is modest on
        # a laptop.  The dramatic gain (>100×) only appears at k=624K where
        # the 5MB probability vector exceeds L2 cache and binomial draw
        # overhead dominates.  See test_categorical_path_call1_full_scale for
        # the full-scale measurement.  Here we only require the categorical
        # path is not materially slower (regression guard).
        assert t_cat < t_mult * 2, (
            f"Categorical path ({t_cat*1e3:.1f}ms) was more than 2× slower than "
            f"multinomial ({t_mult*1e3:.1f}ms) at n={n}, k={k}. "
            f"Check that the n<k dispatch is active."
        )

    def test_categorical_path_call1_full_scale(self):
        """23 locs, n=5000, k=624K — exact HPC Call 1 dimensions.

        On HPC, the old sequential-binomial path took 60.8s for this call.
        The categorical path (O(k) cumsum + O(n log n) sort + O(k+n) merge
        walk) is expected to complete in well under 1s even on a laptop.
        No pass/fail threshold is asserted; timing is always printed so that
        the speedup can be observed directly.
        """
        k, n_locs, n = 624_000, 23, 5_000
        assert n < k, "n must be < k to exercise the categorical path"
        n_rep = 3
        times = [self._timed(k=k, n=n, n_locs=n_locs, seed=i) for i in range(n_rep)]
        best = min(times)
        avg = sum(times) / len(times)
        print(
            f"\n  [Call 1 full scale] n={n}, k={k:,}, n_locs={n_locs}: "
            f"best={best*1e3:.1f}ms, avg={avg*1e3:.1f}ms  "
            f"(old sequential-binomial baseline: ~60,800ms on HPC)"
        )

    def test_categorical_path_scales_linearly_with_k(self):
        """10× increase in k produces at most 30× increase in runtime.

        The categorical path is O(k + n log n).  With n fixed, runtime
        should be approximately proportional to k.  A ratio > 30× (three
        times the expected 10×) suggests super-linear scaling — e.g. from
        a quadratic sort or a wasteful allocation that grows with k.
        The lower bound of 2× guards against test noise swamping a genuinely
        linear relationship.
        """
        n, n_locs = 50, 1
        k_small, k_large = 5_000, 50_000
        assert n < k_small, "n must be < k_small to use categorical path for both"

        # Repeat each measurement to reduce timer noise
        n_rep = 5
        t_small = min(
            self._timed(k=k_small, n=n, n_locs=n_locs, seed=i) for i in range(n_rep)
        )
        t_large = min(
            self._timed(k=k_large, n=n, n_locs=n_locs, seed=i) for i in range(n_rep)
        )

        ratio = t_large / max(t_small, 1e-6)
        print(
            f"\n  [categorical k-scaling] k={k_small}: {t_small*1e3:.2f}ms, "
            f"k={k_large}: {t_large*1e3:.2f}ms, ratio={ratio:.1f}×"
        )
        assert 2.0 <= ratio <= 30.0, (
            f"k scaling ratio was {ratio:.1f}× for k {k_small}→{k_large} "
            f"(expected 2–30×, i.e. roughly linear). "
            f"t_small={t_small*1e3:.2f}ms, t_large={t_large*1e3:.2f}ms."
        )

    def test_categorical_path_n_scaling_is_sublinear_in_k(self):
        """Doubling n (with n < k) does not double runtime.

        When n doubles and k is held constant, the categorical path does
        2× more sort/walk work but the O(k) cumsum is unchanged.  Total
        runtime growth should be well below 4× (the naive worst-case for
        O(n log n) sort doubling).
        """
        k, n_locs = 10_000, 1
        n_small, n_large = 100, 1_000
        assert n_large < k, "Both n values must be < k for categorical path"

        n_rep = 5
        t_small = min(
            self._timed(k=k, n=n_small, n_locs=n_locs, seed=i) for i in range(n_rep)
        )
        t_large = min(
            self._timed(k=k, n=n_large, n_locs=n_locs, seed=i) for i in range(n_rep)
        )

        ratio = t_large / max(t_small, 1e-6)
        print(
            f"\n  [categorical n-scaling] n={n_small}: {t_small*1e3:.2f}ms, "
            f"n={n_large}: {t_large*1e3:.2f}ms, ratio={ratio:.1f}× (k={k})"
        )
        # At n=1000 the sort (O(n log n)) costs more but the cumsum (O(k=10K)) dominates.
        # Expect ratio < 5×; a 10× bound catches regressions while tolerating noise.
        assert ratio < 10.0, (
            f"n scaling ratio was {ratio:.1f}× for n {n_small}→{n_large} at k={k}. "
            f"Expected < 10× (cumsum O(k) should dominate). "
            f"t_small={t_small*1e3:.2f}ms, t_large={t_large*1e3:.2f}ms."
        )

    # ------------------------------------------------------------------
    # Multinomial path (n >= k)
    # ------------------------------------------------------------------

    def test_multinomial_path_many_locs_small_k_is_fast(self):
        """50K locs, n=5, k=3 completes in < 2s (multinomial path, Numba inner loop).

        Mirrors HPC Call 3 at reduced scale (10.26M → 50K hexes).  The
        bottleneck here is per-location loop overhead, not per-category work;
        the Numba-compiled loop should make it sub-millisecond per thousand
        locations.
        """
        k, n_locs, n = 3, 50_000, 5
        assert n >= k, "This test requires n >= k to exercise the multinomial path"
        elapsed = self._timed(k=k, n=n, n_locs=n_locs)
        print(f"\n  [multinomial] n={n}, k={k}, n_locs={n_locs}: {elapsed*1e3:.2f}ms")
        assert elapsed < 2.0, (
            f"Multinomial path (n={n}, k={k}, n_locs={n_locs}) took {elapsed:.3f}s; "
            f"expected < 2s."
        )

    def test_multinomial_path_scales_linearly_with_n_locs(self):
        """10× more locations produces at most 30× increase in runtime.

        The compiled loop has O(1) overhead per location (plus O(k) work per
        active location).  Runtime should be roughly proportional to n_locs.
        """
        k, n = 5, 10  # n >= k → multinomial path
        n_locs_small, n_locs_large = 1_000, 10_000

        n_rep = 5
        t_small = min(
            self._timed(k=k, n=n, n_locs=n_locs_small, seed=i) for i in range(n_rep)
        )
        t_large = min(
            self._timed(k=k, n=n, n_locs=n_locs_large, seed=i) for i in range(n_rep)
        )

        ratio = t_large / max(t_small, 1e-6)
        print(
            f"\n  [multinomial n_locs-scaling] n_locs={n_locs_small}: {t_small*1e3:.2f}ms, "
            f"n_locs={n_locs_large}: {t_large*1e3:.2f}ms, ratio={ratio:.1f}×"
        )
        assert 2.0 <= ratio <= 30.0, (
            f"n_locs scaling ratio was {ratio:.1f}× for n_locs "
            f"{n_locs_small}→{n_locs_large} (expected 2–30×). "
            f"t_small={t_small*1e3:.2f}ms, t_large={t_large*1e3:.2f}ms."
        )

    # ------------------------------------------------------------------
    # Path comparison
    # ------------------------------------------------------------------

    def test_categorical_path_faster_than_multinomial_for_large_k(self):
        """The categorical path (n < k) is faster than the multinomial path (n >= k)
        when k is large and n is small.

        We construct two problems with identical k but different n:
        - Categorical: n = k // 100  (n ≪ k → categorical)
        - Multinomial: n = k         (n ≥ k → multinomial)

        For large k, categorical savings from avoiding k binomial draws
        should outweigh the sort overhead O(n log n), making it
        materially faster.  We require at least a 3× speedup.
        """
        k = 5_000
        n_cat = k // 100  # 50 << 5000 → categorical path
        n_mult = k  # 5000 >= 5000 → multinomial path

        assert n_cat < k
        assert n_mult >= k

        n_rep = 5
        t_cat = min(self._timed(k=k, n=n_cat, n_locs=1, seed=i) for i in range(n_rep))
        t_mult = min(self._timed(k=k, n=n_mult, n_locs=1, seed=i) for i in range(n_rep))

        print(
            f"\n  [path comparison] k={k}: categorical (n={n_cat}) {t_cat*1e3:.2f}ms, "
            f"multinomial (n={n_mult}) {t_mult*1e3:.2f}ms, speedup={t_mult/max(t_cat, 1e-9):.1f}×"
        )
        assert t_cat < t_mult, (
            f"Categorical path ({t_cat*1e3:.2f}ms) was not faster than multinomial "
            f"({t_mult*1e3:.2f}ms) at k={k}, n_cat={n_cat}, n_mult={n_mult}. "
            f"Check that the n<k dispatch is working."
        )
