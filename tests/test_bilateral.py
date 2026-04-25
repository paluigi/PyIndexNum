"""
Tests for bilateral price index functions.

Regression tests use performance_price_data.csv with expected values from
the R PriceIndices library (gold standard). Unit tests use small synthetic
datasets with hand-computed expected values.
"""

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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def perf_data():
    """Load and preprocess performance_price_data.csv for 2024-01 -> 2024-02."""
    df = pl.read_csv("tests/performance_price_data.csv")
    df = standardize_columns(
        df,
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
    agg = agg.rename({"aggregated_price": "price", "aggregated_quantity": "quantity"})
    balanced = remove_unbalanced(agg)
    balanced = balanced.rename({"period": "date"})

    start_dt = pl.lit("2024-01-01").str.strptime(pl.Date, "%Y-%m-%d")
    end_dt = pl.lit("2024-02-01").str.strptime(pl.Date, "%Y-%m-%d")
    balanced = balanced.filter(
        (pl.col("date") == start_dt) | (pl.col("date") == end_dt)
    )
    return balanced


@pytest.fixture
def perf_unweighted(perf_data):
    """Preprocessed data without quantity column (for unweighted indices)."""
    return perf_data.select("date", "price", "product_id")


# ---------------------------------------------------------------------------
# Regression tests against R PriceIndices
# ---------------------------------------------------------------------------


class TestRegressionR:
    """Validate all bilateral indices against R PriceIndices gold standard."""

    def test_jevons(self, perf_unweighted):
        result = jevons(perf_unweighted)
        assert abs(result - R_EXPECTED["jevons"]) < TOLERANCE_R

    def test_carli(self, perf_unweighted):
        result = carli(perf_unweighted)
        assert abs(result - R_EXPECTED["carli"]) < TOLERANCE_R

    def test_dutot(self, perf_unweighted):
        result = dutot(perf_unweighted)
        assert abs(result - R_EXPECTED["dutot"]) < TOLERANCE_R

    def test_laspeyres(self, perf_data):
        result = laspeyres(perf_data)
        assert abs(result - R_EXPECTED["laspeyres"]) < TOLERANCE_R

    def test_paasche(self, perf_data):
        result = paasche(perf_data)
        assert abs(result - R_EXPECTED["paasche"]) < TOLERANCE_R

    def test_fisher(self, perf_data):
        result = fisher(perf_data)
        assert abs(result - R_EXPECTED["fisher"]) < TOLERANCE_R

    def test_tornqvist(self, perf_data):
        result = tornqvist(perf_data)
        assert abs(result - R_EXPECTED["tornqvist"]) < TOLERANCE_R

    def test_walsh(self, perf_data):
        result = walsh(perf_data)
        assert abs(result - R_EXPECTED["walsh"]) < TOLERANCE_R


# ---------------------------------------------------------------------------
# Hand-computed unit tests (small synthetic dataset)
#
# Data:
#   Product A: p0=100, q0=10, pt=110, qt=15
#   Product B: p0=200, q0=20, pt=190, qt=25
# ---------------------------------------------------------------------------

SYNTH_UNWEIGHTED = pl.DataFrame(
    {
        "date": [
            "2023-01-01",
            "2023-01-01",
            "2023-02-01",
            "2023-02-01",
        ],
        "product_id": ["A", "B", "A", "B"],
        "price": [100.0, 200.0, 110.0, 190.0],
    }
)

SYNTH_WEIGHTED = pl.DataFrame(
    {
        "date": [
            "2023-01-01",
            "2023-01-01",
            "2023-02-01",
            "2023-02-01",
        ],
        "product_id": ["A", "B", "A", "B"],
        "price": [100.0, 200.0, 110.0, 190.0],
        "quantity": [10, 20, 15, 25],
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
                "date": ["2023-01-01", "2023-02-01"],
                "product_id": ["A", "A"],
                "price": [100.0, 110.0],
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
                "date": ["2023-01-01", "2023-02-01"],
                "product_id": ["A", "A"],
                "price": [100.0, 110.0],
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
                "date": ["2023-01-01", "2023-02-01"],
                "product_id": ["A", "A"],
                "price": [100.0, 110.0],
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
            pl.when(pl.col("date") == "2023-01-01")
            .then(pl.lit("2023-02-01"))
            .when(pl.col("date") == "2023-02-01")
            .then(pl.lit("2023-01-01"))
            .otherwise(pl.col("date"))
            .alias("date")
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
# Input validation tests
# ---------------------------------------------------------------------------


class TestValidation:
    def test_wrong_number_of_dates(self):
        df = pl.DataFrame(
            {
                "date": ["2023-01-01", "2023-01-01"],
                "product_id": ["A", "B"],
                "price": [100.0, 200.0],
            }
        )
        with pytest.raises(ValueError, match="exactly 2 unique dates"):
            jevons(df)

    def test_three_dates(self):
        df = pl.DataFrame(
            {
                "date": ["2023-01-01", "2023-02-01", "2023-03-01"],
                "product_id": ["A", "A", "A"],
                "price": [100.0, 110.0, 120.0],
            }
        )
        with pytest.raises(ValueError, match="exactly 2 unique dates"):
            jevons(df)

    def test_missing_columns(self):
        df = pl.DataFrame(
            {
                "date": ["2023-01-01", "2023-02-01"],
                "price": [100.0, 110.0],
            }
        )
        with pytest.raises(ValueError, match="Missing required columns"):
            jevons(df)

    def test_zero_price(self):
        df = pl.DataFrame(
            {
                "date": ["2023-01-01", "2023-01-01", "2023-02-01", "2023-02-01"],
                "product_id": ["A", "B", "A", "B"],
                "price": [0.0, 200.0, 110.0, 190.0],
            }
        )
        with pytest.raises(ValueError, match="All prices must be positive"):
            jevons(df)

    def test_negative_price(self):
        df = pl.DataFrame(
            {
                "date": ["2023-01-01", "2023-01-01", "2023-02-01", "2023-02-01"],
                "product_id": ["A", "B", "A", "B"],
                "price": [-100.0, 200.0, 110.0, 190.0],
            }
        )
        with pytest.raises(ValueError, match="All prices must be positive"):
            jevons(df)

    def test_products_differ_between_periods(self):
        df = pl.DataFrame(
            {
                "date": [
                    "2023-01-01",
                    "2023-01-01",
                    "2023-02-01",
                    "2023-02-01",
                ],
                "product_id": ["A", "B", "A", "C"],
                "price": [100.0, 200.0, 110.0, 190.0],
            }
        )
        with pytest.raises(ValueError, match="Products must be identical"):
            jevons(df)

    def test_missing_quantity_column(self):
        df = pl.DataFrame(
            {
                "date": ["2023-01-01", "2023-02-01"],
                "product_id": ["A", "A"],
                "price": [100.0, 110.0],
            }
        )
        with pytest.raises(ValueError, match="Missing required column: quantity"):
            laspeyres(df)

    def test_zero_quantity(self):
        df = pl.DataFrame(
            {
                "date": ["2023-01-01", "2023-02-01"],
                "product_id": ["A", "A"],
                "price": [100.0, 110.0],
                "quantity": [0, 10],
            }
        )
        with pytest.raises(ValueError, match="All quantities must be positive"):
            laspeyres(df)

    def test_negative_quantity(self):
        df = pl.DataFrame(
            {
                "date": ["2023-01-01", "2023-02-01"],
                "product_id": ["A", "A"],
                "price": [100.0, 110.0],
                "quantity": [-10, 10],
            }
        )
        with pytest.raises(ValueError, match="All quantities must be positive"):
            laspeyres(df)

    def test_multiple_prices_per_product_period(self):
        df = pl.DataFrame(
            {
                "date": [
                    "2023-01-01",
                    "2023-01-01",
                    "2023-01-01",
                    "2023-02-01",
                    "2023-02-01",
                    "2023-02-01",
                ],
                "product_id": ["A", "A", "B", "A", "A", "B"],
                "price": [100.0, 105.0, 200.0, 110.0, 115.0, 190.0],
            }
        )
        with pytest.raises(ValueError, match="exactly one price per period"):
            jevons(df)
