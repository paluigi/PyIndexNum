"""
Tests for bilateral price index functions.

Regression tests use performance_price_data.csv with expected values from
the R PriceIndices library (gold standard). Unit tests use small synthetic
datasets with hand-computed expected values.

Since 0.4.0 all bilateral functions consume the canonical panel schema
emitted by aggregate_time (period / product_id / aggregated_price /
aggregated_quantity); custom and legacy column names are supported via
the *_col keyword arguments, and unbalanced two-period comparisons via
balanced=False.
"""

import warnings
from datetime import date

import pytest
import polars as pl
import numpy as np

from pyindexnum.bilateral import (
    jevons,
    carli,
    dutot,
    laspeyres,
    paasche,
    fisher,
    tornqvist,
    walsh,
)
from pyindexnum.utils import standardize_columns, aggregate_time, remove_unbalanced


# ---------------------------------------------------------------------------
# R PriceIndices gold-standard values for performance_price_data.csv
# Dataset: 2024-01 -> 2024-02, 1200 matched products
# ---------------------------------------------------------------------------
R_EXPECTED = {
    "jevons": 1.2007598,
    "carli": 269.7122261,
    "dutot": 1.1356881,
    "laspeyres": 1.0894350,
    "paasche": 1.1380544,
    "fisher": 1.1134794,
    "tornqvist": 1.1828549,
    "walsh": 1.1163318,
}

TOLERANCE_R = 1e-5


# (function, requires quantity) pairs used by the parametrized sweeps
BILATERAL_FUNCS = [
    pytest.param(jevons, False, id="jevons"),
    pytest.param(carli, False, id="carli"),
    pytest.param(dutot, False, id="dutot"),
    pytest.param(laspeyres, True, id="laspeyres"),
    pytest.param(paasche, True, id="paasche"),
    pytest.param(fisher, True, id="fisher"),
    pytest.param(tornqvist, True, id="tornqvist"),
    pytest.param(walsh, True, id="walsh"),
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def perf_panel(perf_raw):
    """aggregate_time output for 2024-01 -> 2024-02, balanced.

    The aggregation output feeds every index function directly — no renames.
    """
    df = standardize_columns(
        perf_raw,
        date_col="time",
        price_col="prices",
        id_col="prodID",
        quantity_col="quantities",
    )
    agg = aggregate_time(
        df,
        date_col="date",
        price_col="price",
        quantity_col="quantity",
        id_col="product_id",
        agg_type="weighted_arithmetic",
        freq="1mo",
    )
    balanced = remove_unbalanced(agg)
    return balanced.filter(
        pl.col("period").is_in([date(2024, 1, 1), date(2024, 2, 1)])
    )


# ---------------------------------------------------------------------------
# Regression tests: aggregate_time output feeds every bilateral function
# directly (the unified-schema bug fix) and matches the gold-standard values
# ---------------------------------------------------------------------------


class TestRegressionR:
    """Validate all bilateral indices on aggregate_time output against gold-standard values."""

    @pytest.mark.parametrize("func,weighted", BILATERAL_FUNCS)
    def test_direct_feed_matches_reference(self, perf_panel, func, weighted):
        assert abs(func(perf_panel) - R_EXPECTED[func.__name__]) < TOLERANCE_R


# ---------------------------------------------------------------------------
# Hand-computed unit tests (small synthetic dataset)
#
# Data:
#   Product A: p0=100, q0=10, pt=110, qt=15
#   Product B: p0=200, q0=20, pt=190, qt=25
# ---------------------------------------------------------------------------

SYNTH_UNWEIGHTED = pl.DataFrame(
    {
        "period": [
            "2023-01-01",
            "2023-01-01",
            "2023-02-01",
            "2023-02-01",
        ],
        "product_id": ["A", "B", "A", "B"],
        "aggregated_price": [100.0, 200.0, 110.0, 190.0],
    }
)

SYNTH_WEIGHTED = pl.DataFrame(
    {
        "period": [
            "2023-01-01",
            "2023-01-01",
            "2023-02-01",
            "2023-02-01",
        ],
        "product_id": ["A", "B", "A", "B"],
        "aggregated_price": [100.0, 200.0, 110.0, 190.0],
        "aggregated_quantity": [10, 20, 15, 25],
    }
)

TOL = 1e-10


class TestJevonsUnit:
    """Jevons = geometric mean of price relatives."""

    def test_synth(self):
        # relatives: 110/100 = 1.1, 190/200 = 0.95
        # gm = (1.1 * 0.95)^(1/2) = sqrt(1.045) = 1.022252...
        expected = np.sqrt(1.1 * 0.95)
        result = jevons(SYNTH_UNWEIGHTED)
        assert abs(result - expected) < TOL

    def test_single_product(self):
        df = pl.DataFrame(
            {
                "period": ["2023-01-01", "2023-02-01"],
                "product_id": ["A", "A"],
                "aggregated_price": [100.0, 110.0],
            }
        )
        assert abs(jevons(df) - 1.1) < TOL


class TestCarliUnit:
    """Carli = arithmetic mean of price relatives."""

    def test_synth(self):
        # relatives: 1.1, 0.95
        # mean = (1.1 + 0.95) / 2 = 1.025
        expected = (1.1 + 0.95) / 2
        result = carli(SYNTH_UNWEIGHTED)
        assert abs(result - expected) < TOL

    def test_single_product(self):
        df = pl.DataFrame(
            {
                "period": ["2023-01-01", "2023-02-01"],
                "product_id": ["A", "A"],
                "aggregated_price": [100.0, 110.0],
            }
        )
        assert abs(carli(df) - 1.1) < TOL


class TestDutotUnit:
    """Dutot = ratio of arithmetic means of prices."""

    def test_synth(self):
        # mean_base = (100 + 200) / 2 = 150
        # mean_curr = (110 + 190) / 2 = 150
        # index = 150 / 150 = 1.0
        expected = 150.0 / 150.0
        result = dutot(SYNTH_UNWEIGHTED)
        assert abs(result - expected) < TOL

    def test_single_product(self):
        df = pl.DataFrame(
            {
                "period": ["2023-01-01", "2023-02-01"],
                "product_id": ["A", "A"],
                "aggregated_price": [100.0, 110.0],
            }
        )
        assert abs(dutot(df) - 1.1) < TOL


class TestLaspeyresUnit:
    """Laspeyres = sum(p_t * q_0) / sum(p_0 * q_0)."""

    def test_synth(self):
        # num = 110*10 + 190*20 = 1100 + 3800 = 4900
        # den = 100*10 + 200*20 = 1000 + 4000 = 5000
        expected = 4900.0 / 5000.0
        result = laspeyres(SYNTH_WEIGHTED)
        assert abs(result - expected) < TOL


class TestPaascheUnit:
    """Paasche = sum(p_t * q_t) / sum(p_0 * q_t)."""

    def test_synth(self):
        # num = 110*15 + 190*25 = 1650 + 4750 = 6400
        # den = 100*15 + 200*25 = 1500 + 5000 = 6500
        expected = 6400.0 / 6500.0
        result = paasche(SYNTH_WEIGHTED)
        assert abs(result - expected) < TOL


class TestFisherUnit:
    """Fisher = sqrt(Laspeyres * Paasche)."""

    def test_synth(self):
        L = 4900.0 / 5000.0
        P = 6400.0 / 6500.0
        expected = np.sqrt(L * P)
        result = fisher(SYNTH_WEIGHTED)
        assert abs(result - expected) < TOL

    def test_time_reversal(self):
        """Fisher(t,0) * Fisher(0,t) = 1."""
        df_rev = SYNTH_WEIGHTED.with_columns(
            pl.when(pl.col("period") == "2023-01-01")
            .then(pl.lit("2023-02-01"))
            .when(pl.col("period") == "2023-02-01")
            .then(pl.lit("2023-01-01"))
            .otherwise(pl.col("period"))
            .alias("period")
        )
        fwd = fisher(SYNTH_WEIGHTED)
        rev = fisher(df_rev)
        assert abs(fwd * rev - 1.0) < 1e-10


class TestTornqvistUnit:
    """Tornqvist = exp(sum(0.5*(s_0+s_t) * ln(p_t/p_0)))."""

    def test_synth(self):
        # Expenditures base: A=100*10=1000, B=200*20=4000, total=5000
        # Expenditures curr: A=110*15=1650, B=190*25=4750, total=6400
        # Shares base: A=0.2, B=0.8
        # Shares curr: A=1650/6400=0.2578125, B=4750/6400=0.7421875
        # Avg shares: A=0.5*(0.2+0.2578125)=0.22890625
        #             B=0.5*(0.8+0.7421875)=0.77109375
        # Log relatives: A=ln(110/100)=0.0953102..., B=ln(190/200)=-0.0512932...
        # Sum = 0.22890625*0.09531018 + 0.77109375*(-0.05129329)
        #     = 0.02181734 - 0.03955297 = -0.01773563
        # exp(-0.01773563) = 0.98242116...
        expected = np.exp(-0.01773563)
        result = tornqvist(SYNTH_WEIGHTED)
        assert abs(result - expected) < 1e-6

    def test_synth_step_by_step(self):
        """Verify each intermediate step of the Tornqvist calculation."""
        p0 = np.array([100.0, 200.0])
        pt = np.array([110.0, 190.0])
        q0 = np.array([10.0, 20.0])
        qt = np.array([15.0, 25.0])

        exp0 = p0 * q0  # [1000, 4000]
        expt = pt * qt  # [1650, 4750]
        assert np.allclose(exp0, [1000, 4000])
        assert np.allclose(expt, [1650, 4750])

        s0 = exp0 / exp0.sum()  # [0.2, 0.8]
        st = expt / expt.sum()
        assert abs(s0.sum() - 1.0) < TOL
        assert abs(st.sum() - 1.0) < TOL
        assert np.allclose(s0, [0.2, 0.8])

        avg_s = 0.5 * (s0 + st)
        assert abs(avg_s.sum() - 1.0) < TOL

        log_rel = np.log(pt / p0)
        weighted = avg_s * log_rel
        expected = np.exp(weighted.sum())

        result = tornqvist(SYNTH_WEIGHTED)
        assert abs(result - expected) < TOL


class TestWalshUnit:
    """Walsh = sum(p_t * sqrt(q_0 * q_t)) / sum(p_0 * sqrt(q_0 * q_t))."""

    def test_synth(self):
        # basket: A=sqrt(10*15)=sqrt(150)=12.24744871...
        #         B=sqrt(20*25)=sqrt(500)=22.36067977...
        # num = 110*12.24744871 + 190*22.36067977 = 1347.21936 + 4248.52916 = 5595.74852
        # den = 100*12.24744871 + 200*22.36067977 = 1224.74487 + 4472.13595 = 5696.88083
        basket = np.sqrt(np.array([10.0, 20.0]) * np.array([15.0, 25.0]))
        p0 = np.array([100.0, 200.0])
        pt = np.array([110.0, 190.0])
        expected = np.sum(pt * basket) / np.sum(p0 * basket)
        result = walsh(SYNTH_WEIGHTED)
        assert abs(result - expected) < 1e-10

    def test_synth_step_by_step(self):
        """Verify basket, numerator, and denominator separately."""
        q0 = np.array([10.0, 20.0])
        qt = np.array([15.0, 25.0])
        basket = np.sqrt(q0 * qt)

        assert abs(basket[0] - np.sqrt(150)) < TOL
        assert abs(basket[1] - np.sqrt(500)) < TOL

        p0 = np.array([100.0, 200.0])
        pt = np.array([110.0, 190.0])
        num = np.sum(pt * basket)
        den = np.sum(p0 * basket)

        result = walsh(SYNTH_WEIGHTED)
        assert abs(result - num / den) < TOL


# ---------------------------------------------------------------------------
# Custom column names
# ---------------------------------------------------------------------------


CUSTOM_RENAME = {
    "period": "month",
    "aggregated_price": "avg_price",
    "product_id": "sku",
    "aggregated_quantity": "total_qty",
}


class TestCustomColumns:
    """User-defined column names produce identical results to canonical names."""

    @pytest.mark.parametrize("func,weighted", BILATERAL_FUNCS)
    def test_custom_names_match_canonical(self, func, weighted):
        canonical = SYNTH_WEIGHTED if weighted else SYNTH_UNWEIGHTED
        if weighted:
            custom = canonical.rename(CUSTOM_RENAME)
        else:
            custom = canonical.rename(
                {k: v for k, v in CUSTOM_RENAME.items() if k != "aggregated_quantity"}
            )
        kwargs = {"period_col": "month", "price_col": "avg_price", "id_col": "sku"}
        if weighted:
            kwargs["quantity_col"] = "total_qty"
        assert func(custom, **kwargs) == pytest.approx(func(canonical), rel=1e-12)

    def test_missing_custom_column(self):
        custom = SYNTH_UNWEIGHTED.rename({"period": "month"})
        with pytest.raises(ValueError, match="Missing required columns"):
            jevons(custom, period_col="nonexistent")

    def test_rename_collision(self):
        custom = SYNTH_UNWEIGHTED.rename({"period": "month"})
        custom = custom.with_columns(pl.lit(0).alias("period"))  # junk column
        with pytest.raises(ValueError, match="Cannot rename column"):
            jevons(custom, period_col="month")


class TestLegacySchema:
    """Pre-0.4.0 date/price/quantity column names keep working via kwargs."""

    LEGACY_RENAME = {
        "period": "date",
        "aggregated_price": "price",
        "aggregated_quantity": "quantity",
    }

    @pytest.mark.parametrize("func,weighted", BILATERAL_FUNCS)
    def test_legacy_columns_via_kwargs(self, func, weighted):
        canonical = SYNTH_WEIGHTED if weighted else SYNTH_UNWEIGHTED
        if weighted:
            legacy = canonical.rename(self.LEGACY_RENAME)
        else:
            legacy = canonical.rename(
                {k: v for k, v in self.LEGACY_RENAME.items() if k != "aggregated_quantity"}
            )
        kwargs = {"period_col": "date", "price_col": "price", "id_col": "product_id"}
        if weighted:
            kwargs["quantity_col"] = "quantity"
        assert func(legacy, **kwargs) == pytest.approx(func(canonical), rel=1e-12)


# ---------------------------------------------------------------------------
# Unbalanced panels (balanced=False)
#
# Data: base period has products {A, B, C}, current period has {A, B, D}.
# The matched set is {A, B} with the same values as SYNTH_WEIGHTED.
# ---------------------------------------------------------------------------

UNBAL_UNWEIGHTED = pl.DataFrame(
    {
        "period": [
            "2023-01-01",
            "2023-01-01",
            "2023-01-01",
            "2023-02-01",
            "2023-02-01",
            "2023-02-01",
        ],
        "product_id": ["A", "B", "C", "A", "B", "D"],
        "aggregated_price": [100.0, 200.0, 300.0, 110.0, 190.0, 400.0],
    }
)

UNBAL_WEIGHTED = UNBAL_UNWEIGHTED.with_columns(
    pl.Series("aggregated_quantity", [10.0, 20.0, 5.0, 15.0, 25.0, 2.0])
)

MATCHED_SUBSET = UNBAL_WEIGHTED.filter(pl.col("product_id").is_in(["A", "B"]))


class TestUnbalancedPanels:
    """balanced=False computes each index on the matched product set."""

    @pytest.mark.parametrize("func,weighted", BILATERAL_FUNCS)
    def test_equals_manually_matched_subset(self, func, weighted):
        df = UNBAL_WEIGHTED if weighted else UNBAL_UNWEIGHTED
        assert func(df, balanced=False) == pytest.approx(func(MATCHED_SUBSET), rel=1e-12)

    @pytest.mark.parametrize("func,weighted", BILATERAL_FUNCS)
    def test_strict_default_still_raises(self, func, weighted):
        df = UNBAL_WEIGHTED if weighted else UNBAL_UNWEIGHTED
        with pytest.raises(ValueError, match="Products must be identical"):
            func(df)

    @pytest.mark.parametrize("func,weighted", BILATERAL_FUNCS)
    def test_panel_warnings_emitted(self, func, weighted):
        df = UNBAL_WEIGHTED if weighted else UNBAL_UNWEIGHTED
        with pytest.warns(UserWarning, match="Unbalanced comparison"):
            func(df, balanced=False, panel_warnings=True)

    @pytest.mark.parametrize("func,weighted", BILATERAL_FUNCS)
    def test_silent_by_default(self, func, weighted):
        df = UNBAL_WEIGHTED if weighted else UNBAL_UNWEIGHTED
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            func(df, balanced=False)

    def test_no_common_products(self):
        df = pl.DataFrame(
            {
                "period": ["2023-01-01", "2023-02-01"],
                "product_id": ["A", "B"],
                "aggregated_price": [100.0, 200.0],
            }
        )
        with pytest.raises(ValueError, match="No common products"):
            jevons(df, balanced=False)

    def test_dutot_matched_means(self):
        # Unmatched Dutot compares means over the matched products {A, B}:
        # (110 + 190) / 2 over (100 + 200) / 2 = 150 / 150 = 1.0
        assert dutot(UNBAL_UNWEIGHTED, balanced=False) == pytest.approx(1.0, rel=1e-12)

    def test_tornqvist_shares_renormalised_over_matched_set(self):
        # Matched products A and B carry the same values as SYNTH_WEIGHTED,
        # so the matched-sample Tornqvist equals the balanced case computed
        # on {A, B} only: shares of C (base) and D (current) are excluded.
        expected = np.exp(-0.01773563)
        assert tornqvist(UNBAL_WEIGHTED, balanced=False) == pytest.approx(expected, abs=1e-6)

    def test_fisher_threads_flags(self):
        expected = np.sqrt(
            laspeyres(MATCHED_SUBSET) * paasche(MATCHED_SUBSET)
        )
        assert fisher(UNBAL_WEIGHTED, balanced=False) == pytest.approx(expected, rel=1e-12)


# ---------------------------------------------------------------------------
# Matched-sample regression on the performance dataset with product churn:
# products with sorted rank r are dropped from the second month when
# r % 10 == 1 and from the first month when r % 10 == 6
# ---------------------------------------------------------------------------

MATCHED_EXPECTED = {
    "jevons": 1.09010035769845,
    "carli": 248.013541588354,
    "dutot": 1.15171184436301,
    "laspeyres": 1.08941767299547,
    "paasche": 1.13327862877501,
    "fisher": 1.11113175034087,
    "tornqvist": 1.09100612362932,
    "walsh": 1.11343141764876,
}

TOLERANCE_MATCHED = 1e-6


@pytest.fixture
def perf_churned(perf_panel):
    """Balanced two-month panel with deterministic product churn applied."""
    prods = sorted(perf_panel["product_id"].unique().to_list())
    months = sorted(perf_panel["period"].unique().to_list())
    r = pl.Series([prods.index(p) + 1 for p in perf_panel["product_id"].to_list()])
    k = pl.Series([months.index(m) + 1 for m in perf_panel["period"].to_list()])
    keep = ~((k == 2) & (r % 10 == 1)) & ~((k == 1) & (r % 10 == 6))
    return perf_panel.filter(keep)


class TestMatchedSampleRegression:
    """All bilateral indices on the churned panel against expected values."""

    @pytest.mark.parametrize("func,weighted", BILATERAL_FUNCS)
    def test_matches_expected(self, perf_churned, func, weighted):
        assert abs(func(perf_churned, balanced=False) - MATCHED_EXPECTED[func.__name__]) < TOLERANCE_MATCHED


# ---------------------------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------------------------


class TestValidation:
    def test_wrong_number_of_periods(self):
        df = pl.DataFrame(
            {
                "period": ["2023-01-01", "2023-01-01"],
                "product_id": ["A", "B"],
                "aggregated_price": [100.0, 200.0],
            }
        )
        with pytest.raises(ValueError, match="exactly 2 unique periods"):
            jevons(df)

    def test_three_periods(self):
        df = pl.DataFrame(
            {
                "period": ["2023-01-01", "2023-02-01", "2023-03-01"],
                "product_id": ["A", "A", "A"],
                "aggregated_price": [100.0, 110.0, 120.0],
            }
        )
        with pytest.raises(ValueError, match="exactly 2 unique periods"):
            jevons(df)

    def test_missing_columns(self):
        df = pl.DataFrame(
            {
                "period": ["2023-01-01", "2023-02-01"],
                "aggregated_price": [100.0, 110.0],
            }
        )
        with pytest.raises(ValueError, match="Missing required columns"):
            jevons(df)

    def test_zero_price(self):
        df = pl.DataFrame(
            {
                "period": ["2023-01-01", "2023-01-01", "2023-02-01", "2023-02-01"],
                "product_id": ["A", "B", "A", "B"],
                "aggregated_price": [0.0, 200.0, 110.0, 190.0],
            }
        )
        with pytest.raises(ValueError, match="All prices must be positive"):
            jevons(df)

    def test_negative_price(self):
        df = pl.DataFrame(
            {
                "period": ["2023-01-01", "2023-01-01", "2023-02-01", "2023-02-01"],
                "product_id": ["A", "B", "A", "B"],
                "aggregated_price": [-100.0, 200.0, 110.0, 190.0],
            }
        )
        with pytest.raises(ValueError, match="All prices must be positive"):
            jevons(df)

    def test_products_differ_between_periods(self):
        df = pl.DataFrame(
            {
                "period": [
                    "2023-01-01",
                    "2023-01-01",
                    "2023-02-01",
                    "2023-02-01",
                ],
                "product_id": ["A", "B", "A", "C"],
                "aggregated_price": [100.0, 200.0, 110.0, 190.0],
            }
        )
        with pytest.raises(ValueError, match="Products must be identical"):
            jevons(df)

    def test_missing_quantity_column(self):
        with pytest.raises(ValueError, match="Missing required column"):
            laspeyres(SYNTH_UNWEIGHTED)

    def test_zero_quantity(self):
        df = pl.DataFrame(
            {
                "period": ["2023-01-01", "2023-02-01"],
                "product_id": ["A", "A"],
                "aggregated_price": [100.0, 110.0],
                "aggregated_quantity": [0, 10],
            }
        )
        with pytest.raises(ValueError, match="All quantities must be positive"):
            laspeyres(df)

    def test_negative_quantity(self):
        df = pl.DataFrame(
            {
                "period": ["2023-01-01", "2023-02-01"],
                "product_id": ["A", "A"],
                "aggregated_price": [100.0, 110.0],
                "aggregated_quantity": [-10, 10],
            }
        )
        with pytest.raises(ValueError, match="All quantities must be positive"):
            laspeyres(df)

    def test_multiple_prices_per_product_period(self):
        df = pl.DataFrame(
            {
                "period": [
                    "2023-01-01",
                    "2023-01-01",
                    "2023-01-01",
                    "2023-02-01",
                    "2023-02-01",
                    "2023-02-01",
                ],
                "product_id": ["A", "A", "B", "A", "A", "B"],
                "aggregated_price": [100.0, 105.0, 200.0, 110.0, 115.0, 190.0],
            }
        )
        with pytest.raises(ValueError, match="exactly one price per period"):
            jevons(df)
