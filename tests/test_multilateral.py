"""
Tests for multilateral price index functions.

Covers the canonical panel schema, custom column names, and unbalanced-panel
support (balanced=False) for the GEKS family and Geary-Khamis, plus the
naturally unbalanced time-product-dummy estimator.
"""

import warnings
from collections import defaultdict
from datetime import date

import pytest
import polars as pl
import numpy as np

from pyindexnum import (
    geks_fisher,
    geks_tornqvist,
    geks_jevons,
    geary_khamis,
    time_product_dummy,
)
from pyindexnum.bilateral import fisher
from pyindexnum.utils import standardize_columns, aggregate_time, remove_unbalanced

TOL = 1e-10

MULTILATERAL_FUNCS = [
    pytest.param(geks_fisher, id="geks_fisher"),
    pytest.param(geks_tornqvist, id="geks_tornqvist"),
    pytest.param(geks_jevons, id="geks_jevons"),
    pytest.param(geary_khamis, id="geary_khamis"),
    pytest.param(time_product_dummy, id="time_product_dummy"),
]


class TestGEKSIndices:
    """Test GEKS multilateral indices."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return pl.DataFrame({
            "product_id": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
            "period": [
                date(2023, 1, 1), date(2023, 2, 1), date(2023, 3, 1),
                date(2023, 1, 1), date(2023, 2, 1), date(2023, 3, 1),
                date(2023, 1, 1), date(2023, 2, 1), date(2023, 3, 1)
            ],
            "aggregated_price": [100, 105, 110, 200, 210, 220, 50, 52, 54],
            "aggregated_quantity": [10, 10, 10, 20, 20, 20, 5, 5, 5]
        })

    def test_geks_fisher_basic(self, sample_data):
        """Test basic GEKS-Fisher functionality."""
        result = geks_fisher(sample_data)

        # Check structure
        assert isinstance(result, pl.DataFrame)
        assert result.columns == ["period", "index_value"]
        assert result.height == 3  # 3 periods

        # Check base period is 1.0
        assert result.filter(pl.col("period") == date(2023, 1, 1)).select("index_value").item() == pytest.approx(1.0)

        # For GEKS-Fisher with constant quantities across all periods,
        # the multilateral index should be transitive and match bilateral Fisher.
        # Since quantities are constant (10, 20, 5), Fisher matches Laspeyres/Paasche.
        # Relative price change from period 1 to 2 for A, B, C: 1.05, 1.05, 1.04
        # Fisher(1,2) = (105*10 + 210*20 + 52*5) / (100*10 + 200*20 + 50*5)
        # = (1050 + 4200 + 260) / (1000 + 4000 + 250) = 5510 / 5250 ≈ 1.0495238
        expected_2 = 5510 / 5250
        assert result.filter(pl.col("period") == date(2023, 2, 1)).select("index_value").item() == pytest.approx(expected_2, abs=1e-6)

    def test_geks_tornqvist_basic(self, sample_data):
        """Test basic GEKS-Törnqvist functionality."""
        result = geks_tornqvist(sample_data)

        # Check structure
        assert isinstance(result, pl.DataFrame)
        assert result.columns == ["period", "index_value"]
        assert result.height == 3

        # Check base period is 1.0
        assert result.filter(pl.col("period") == date(2023, 1, 1)).select("index_value").item() == pytest.approx(1.0)

        # With constant quantities, GEKS-Tornqvist reduces to the bilateral Tornqvist.
        # GEKS(2) = [T(1,2)/T(1,1) * T(2,2)/T(2,1) * T(3,2)/T(3,1)]^(1/3)
        # = [T12 * T12 * T13/T23]^(1/3)
        # T12 = 1.0495238, T13 = 1.0990476, T23 = 1.0471869
        # GEKS(2) ≈ 1.0495238
        expected_2 = 1.0495238009
        assert result.filter(pl.col("period") == date(2023, 2, 1)).select("index_value").item() == pytest.approx(expected_2, abs=1e-6)

    def test_geks_insufficient_periods(self):
        """Test GEKS with insufficient periods."""
        df = pl.DataFrame({
            "product_id": ["A", "B"],
            "period": [date(2023, 1, 1), date(2023, 1, 1)],
            "aggregated_price": [100, 200],
            "aggregated_quantity": [10, 20]
        })

        with pytest.raises(ValueError, match="require at least 2 periods"):
            geks_fisher(df)

    def test_geks_insufficient_products(self):
        """Test GEKS with insufficient products."""
        df = pl.DataFrame({
            "product_id": ["A", "A"],
            "period": [date(2023, 1, 1), date(2023, 2, 1)],
            "aggregated_price": [100, 105],
            "aggregated_quantity": [10, 10]
        })

        with pytest.raises(ValueError, match="require at least 2 products"):
            geks_fisher(df)


class TestGEKSJevons:
    """Test GEKS-Jevons multilateral index."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data with quantity for testing (3 products, 3 periods)."""
        return pl.DataFrame({
            "product_id": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
            "period": [
                date(2023, 1, 1), date(2023, 2, 1), date(2023, 3, 1),
                date(2023, 1, 1), date(2023, 2, 1), date(2023, 3, 1),
                date(2023, 1, 1), date(2023, 2, 1), date(2023, 3, 1)
            ],
            "aggregated_price": [100, 105, 110, 200, 210, 220, 50, 52, 54],
            "aggregated_quantity": [10, 10, 10, 20, 20, 20, 5, 5, 5]
        })

    @pytest.fixture
    def sample_data_no_quantity(self):
        """Create sample data without quantity column (3 products, 3 periods)."""
        return pl.DataFrame({
            "product_id": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
            "period": [
                date(2023, 1, 1), date(2023, 2, 1), date(2023, 3, 1),
                date(2023, 1, 1), date(2023, 2, 1), date(2023, 3, 1),
                date(2023, 1, 1), date(2023, 2, 1), date(2023, 3, 1)
            ],
            "aggregated_price": [100, 105, 110, 200, 210, 220, 50, 52, 54]
        })

    def test_geks_jevons_basic_structure(self, sample_data):
        """Test basic GEKS-Jevons output structure."""
        result = geks_jevons(sample_data)

        # Check structure
        assert isinstance(result, pl.DataFrame)
        assert result.columns == ["period", "index_value"]
        assert result.height == 3  # 3 periods

        # Check base period is 1.0
        assert result.filter(pl.col("period") == date(2023, 1, 1)).select("index_value").item() == pytest.approx(1.0)

    def test_geks_jevons_values(self, sample_data):
        """Test GEKS-Jevons with hand-computed expected values.

        With 3 products (A, B, C) and 3 periods, the bilateral Jevons indices are:
        J(0,1) = (105/100 * 210/200 * 52/50)^(1/3) = (1.05 * 1.05 * 1.04)^(1/3)
        J(0,2) = (110/100 * 220/200 * 54/50)^(1/3)

        Since Jevons is transitive (satisfies the circular test) with complete data,
        GEKS-Jevons reduces to the bilateral Jevons index.
        """
        result = geks_jevons(sample_data)

        # Hand-computed bilateral Jevons values
        expected_2 = (1.05 * 1.05 * 1.04) ** (1 / 3)  # J(0,1) ≈ 1.046656
        expected_3 = (1.10 * 1.10 * 1.08) ** (1 / 3)  # J(0,2) ≈ 1.093293

        assert result.filter(pl.col("period") == date(2023, 2, 1)).select("index_value").item() == pytest.approx(expected_2, abs=1e-6)
        assert result.filter(pl.col("period") == date(2023, 3, 1)).select("index_value").item() == pytest.approx(expected_3, abs=1e-6)

    def test_geks_jevons_equals_bilateral_jevons_constant_quantities(self, sample_data):
        """Test that GEKS-Jevons matches bilateral Jevons with complete balanced data.

        Since Jevons is transitive and data is complete and balanced,
        the GEKS combination should produce the same result as the bilateral Jevons.
        This is because P_J is transitive: P_J(0,t) = P_J(0,k) * P_J(k,t) for all k,
        so the GEKS geometric average collapses to P_J(0,t).
        """
        from pyindexnum import jevons

        result = geks_jevons(sample_data)

        # Compute bilateral Jevons for period 2 vs period 1
        bilateral_df = pl.DataFrame({
            "product_id": ["A", "A", "B", "B", "C", "C"],
            "period": [
                date(2023, 1, 1), date(2023, 2, 1),
                date(2023, 1, 1), date(2023, 2, 1),
                date(2023, 1, 1), date(2023, 2, 1)
            ],
            "aggregated_price": [100, 105, 200, 210, 50, 52]
        })
        bilateral_val = jevons(bilateral_df)
        geks_val = result.filter(pl.col("period") == date(2023, 2, 1)).select("index_value").item()
        assert geks_val == pytest.approx(bilateral_val, abs=1e-10)

    def test_geks_jevons_increasing_prices(self):
        """Test GEKS-Jevons with monotonically increasing prices."""
        df = pl.DataFrame({
            "product_id": ["A", "A", "A", "B", "B", "B"],
            "period": [
                date(2023, 1, 1), date(2023, 2, 1), date(2023, 3, 1),
                date(2023, 1, 1), date(2023, 2, 1), date(2023, 3, 1)
            ],
            "aggregated_price": [100, 110, 120, 200, 220, 240],
        })
        result = geks_jevons(df)

        # All index values should be > 1 for periods after base
        for i in range(1, result.height):
            val = result[i, "index_value"]
            assert val > 1.0

        # Index values should be monotonically increasing
        values = result.sort("period").select("index_value").to_series().to_list()
        for i in range(1, len(values)):
            assert values[i] >= values[i - 1]

    def test_geks_jevons_quantity_invariant(self):
        """Test that GEKS-Jevons is invariant to quantity changes.

        Since Jevons is an unweighted index, changing quantities should not
        affect the result.
        """
        df_fixed_qty = pl.DataFrame({
            "product_id": ["A", "A", "B", "B"],
            "period": [date(2023, 1, 1), date(2023, 2, 1),
                       date(2023, 1, 1), date(2023, 2, 1)],
            "aggregated_price": [100, 110, 200, 210],
            "aggregated_quantity": [10, 10, 20, 20]
        })
        df_var_qty = pl.DataFrame({
            "product_id": ["A", "A", "B", "B"],
            "period": [date(2023, 1, 1), date(2023, 2, 1),
                       date(2023, 1, 1), date(2023, 2, 1)],
            "aggregated_price": [100, 110, 200, 210],
            "aggregated_quantity": [10, 50, 20, 100]  # Very different quantities
        })

        result_fixed = geks_jevons(df_fixed_qty)
        result_var = geks_jevons(df_var_qty)

        val_fixed = result_fixed.filter(pl.col("period") == date(2023, 2, 1)).select("index_value").item()
        val_var = result_var.filter(pl.col("period") == date(2023, 2, 1)).select("index_value").item()

        assert val_fixed == pytest.approx(val_var, abs=1e-10)

    def test_geks_jevons_no_quantity_column(self, sample_data_no_quantity):
        """Test that GEKS-Jevons works without a quantity column.

        Since Jevons is an unweighted index, the quantity column should be
        entirely optional.
        """
        result = geks_jevons(sample_data_no_quantity)

        assert isinstance(result, pl.DataFrame)
        assert result.columns == ["period", "index_value"]
        assert result.height == 3
        assert result.filter(pl.col("period") == date(2023, 1, 1)).select("index_value").item() == pytest.approx(1.0)

        # Should produce the same values as with quantity column
        df_with_qty = sample_data_no_quantity.with_columns(
            pl.lit(10).alias("aggregated_quantity")
        )
        result_with_qty = geks_jevons(df_with_qty)

        vals_no = result.sort("period").select("index_value").to_series().to_list()
        vals_with = result_with_qty.sort("period").select("index_value").to_series().to_list()
        assert vals_no == pytest.approx(vals_with, abs=1e-10)

    def test_geks_jevons_insufficient_periods(self):
        """Test GEKS-Jevons with insufficient periods."""
        df = pl.DataFrame({
            "product_id": ["A", "B"],
            "period": [date(2023, 1, 1), date(2023, 1, 1)],
            "aggregated_price": [100, 200],
        })

        with pytest.raises(ValueError, match="require at least 2 periods"):
            geks_jevons(df)

    def test_geks_jevons_insufficient_products(self):
        """Test GEKS-Jevons with insufficient products."""
        df = pl.DataFrame({
            "product_id": ["A", "A"],
            "period": [date(2023, 1, 1), date(2023, 2, 1)],
            "aggregated_price": [100, 105],
        })

        with pytest.raises(ValueError, match="require at least 2 products"):
            geks_jevons(df)


class TestGearyKhamis:
    """Test Geary-Khamis multilateral index."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return pl.DataFrame({
            "product_id": ["A", "A", "A", "B", "B", "B"],
            "period": [
                date(2023, 1, 1), date(2023, 2, 1), date(2023, 3, 1),
                date(2023, 1, 1), date(2023, 2, 1), date(2023, 3, 1)
            ],
            "aggregated_price": [100, 105, 110, 200, 210, 220],
            "aggregated_quantity": [10, 10, 10, 20, 20, 20]
        })

    def test_geary_khamis_basic(self, sample_data):
        """Test basic Geary-Khamis functionality."""
        result = geary_khamis(sample_data)

        # Check structure
        assert isinstance(result, pl.DataFrame)
        assert result.columns == ["period", "index_value"]
        assert result.height == 3

        # Check base period is 1.0
        assert result.filter(pl.col("period") == date(2023, 1, 1)).select("index_value").item() == pytest.approx(1.0, abs=1e-6)

        # With constant quantities, price levels should follow the average price change
        # Period 2 vs Period 1: (105*10 + 210*20) / (100*10 + 200*20) = 5250 / 5000 = 1.05
        assert result.filter(pl.col("period") == date(2023, 2, 1)).select("index_value").item() == pytest.approx(1.05, abs=1e-6)

    def test_geary_khamis_convergence(self, sample_data):
        """Test Geary-Khamis convergence."""
        # Should converge quickly for this simple case
        result = geary_khamis(sample_data, max_iter=10)
        assert result.height == 3


class TestTimeProductDummy:
    """Test Time Product Dummy multilateral index."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return pl.DataFrame({
            "product_id": ["A", "A", "A", "B", "B", "B"],
            "period": [
                date(2023, 1, 1), date(2023, 2, 1), date(2023, 3, 1),
                date(2023, 1, 1), date(2023, 2, 1), date(2023, 3, 1)
            ],
            "aggregated_price": [100, 105, 110, 200, 210, 220],
            "aggregated_quantity": [10, 10, 10, 20, 20, 20]
        })

    @pytest.fixture
    def sample_data_no_quantity(self):
        """Create sample data without quantity column."""
        return pl.DataFrame({
            "product_id": ["A", "A", "A", "B", "B", "B"],
            "period": [
                date(2023, 1, 1), date(2023, 2, 1), date(2023, 3, 1),
                date(2023, 1, 1), date(2023, 2, 1), date(2023, 3, 1)
            ],
            "aggregated_price": [100, 105, 110, 200, 210, 220]
        })

    def test_time_product_dummy_weighted(self, sample_data):
        """Test weighted Time Product Dummy."""
        result = time_product_dummy(sample_data, weighted=True)

        assert isinstance(result, pl.DataFrame)
        assert result.columns == ["period", "index_value"]
        assert result.height == 3
        assert result.filter(pl.col("period") == date(2023, 1, 1)).select("index_value").item() == pytest.approx(1.0)
        assert result.filter(pl.col("index_value") <= 0).height == 0

    def test_time_product_dummy_weighted_expenditure_shares(self):
        """Test that weighted TPD uses expenditure shares, not raw quantities."""
        df = pl.DataFrame({
            "product_id": ["A", "A", "B", "B"],
            "period": [date(2023, 1, 1), date(2023, 2, 1),
                       date(2023, 1, 1), date(2023, 2, 1)],
            "aggregated_price": [10.0, 12.0, 100.0, 105.0],
            "aggregated_quantity": [100, 100, 1, 1],
        })
        result_weighted = time_product_dummy(df, weighted=True)
        result_unweighted = time_product_dummy(df, weighted=False)

        assert result_weighted.height == 2
        assert result_unweighted.height == 2

        py_w = result_weighted.filter(pl.col("period") == date(2023, 2, 1)).select("index_value").item()
        py_u = result_unweighted.filter(pl.col("period") == date(2023, 2, 1)).select("index_value").item()
        assert py_w != pytest.approx(py_u, abs=1e-3)

    def test_time_product_dummy_unweighted_ignores_quantity(self, sample_data):
        """Test that unweighted TPD gives same results with or without quantity column."""
        result_with_qty = time_product_dummy(sample_data, weighted=False)
        sample_data_no_q = sample_data.drop("aggregated_quantity")
        result_no_qty = time_product_dummy(sample_data_no_q, weighted=False)

        vals_with = result_with_qty.sort("period").select("index_value").to_series().to_list()
        vals_no = result_no_qty.sort("period").select("index_value").to_series().to_list()
        assert vals_with == pytest.approx(vals_no, abs=1e-10)

    def test_time_product_dummy_no_quantity_weighted(self, sample_data_no_quantity):
        """Test that weighted TPD fails without quantity column."""
        with pytest.raises(ValueError, match="Missing required columns"):
            time_product_dummy(sample_data_no_quantity, weighted=True)

    def test_time_product_dummy_no_quantity_unweighted(self, sample_data_no_quantity):
        """Test that unweighted TPD works without quantity column."""
        result = time_product_dummy(sample_data_no_quantity, weighted=False)
        assert result.height == 3


# ---------------------------------------------------------------------------
# Direct feed: aggregate_time output runs through every multilateral function
# with zero renaming
# ---------------------------------------------------------------------------


class TestDirectFeedFromAggregation:
    """aggregate_time output feeds every multilateral function directly."""

    @pytest.fixture
    def agg_panel(self, test_data_raw):
        df = standardize_columns(
            test_data_raw,
            date_col="date",
            price_col="price",
            id_col="product_id",
            quantity_col="quantity",
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
        return remove_unbalanced(agg)

    @pytest.mark.parametrize("func", MULTILATERAL_FUNCS)
    def test_direct_feed(self, agg_panel, func):
        result = func(agg_panel)

        assert isinstance(result, pl.DataFrame)
        assert result.columns == ["period", "index_value"]
        n_periods = agg_panel.select("period").unique().height
        assert result.height == n_periods
        values = result.select("index_value").to_series().to_list()
        assert all(np.isfinite(v) and v > 0 for v in values)
        assert values[0] == pytest.approx(1.0)


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

    @pytest.fixture
    def canonical(self):
        return pl.DataFrame({
            "product_id": ["A", "A", "A", "B", "B", "B"],
            "period": [
                date(2023, 1, 1), date(2023, 2, 1), date(2023, 3, 1),
                date(2023, 1, 1), date(2023, 2, 1), date(2023, 3, 1)
            ],
            "aggregated_price": [100, 105, 110, 200, 210, 220],
            "aggregated_quantity": [10, 12, 14, 20, 22, 24]
        })

    @pytest.mark.parametrize("func", MULTILATERAL_FUNCS)
    def test_custom_names_match_canonical(self, canonical, func):
        custom = canonical.rename(CUSTOM_RENAME)
        kwargs = {"period_col": "month", "price_col": "avg_price", "id_col": "sku"}
        if func is not geks_jevons:  # unweighted: no quantity_col parameter
            kwargs["quantity_col"] = "total_qty"
        expected = func(canonical)
        result = func(custom, **kwargs)
        assert result.select("index_value").to_series().to_list() == pytest.approx(
            expected.select("index_value").to_series().to_list(), rel=1e-10
        )

    def test_missing_custom_column(self, canonical):
        custom = canonical.rename({"period": "month"})
        with pytest.raises(ValueError, match="Missing required columns"):
            geks_fisher(custom, period_col="nonexistent")


# ---------------------------------------------------------------------------
# Unbalanced panels: GEKS
#
# Churn design (3 periods): C exists in periods 0-1 only, D in periods 1-2
# only, so pair (0,1) matches on {A,B,C}, (1,2) on {A,B,D}, (0,2) on {A,B}.
# ---------------------------------------------------------------------------


def unbalanced_churn_panel():
    return pl.DataFrame({
        "product_id": ["A", "A", "A", "B", "B", "B", "C", "C", "D", "D"],
        "period": [
            date(2023, 1, 1), date(2023, 2, 1), date(2023, 3, 1),
            date(2023, 1, 1), date(2023, 2, 1), date(2023, 3, 1),
            date(2023, 1, 1), date(2023, 2, 1),
            date(2023, 2, 1), date(2023, 3, 1),
        ],
        "aggregated_price": [100, 105, 110, 200, 210, 220, 50, 52, 400, 380],
        "aggregated_quantity": [10, 11, 12, 20, 21, 22, 5, 6, 2, 3],
    })


class TestUnbalancedGEKS:
    """balanced=False computes each GEKS link on the matched product set."""

    def test_hand_computed_geks_jevons_with_churn(self):
        df = unbalanced_churn_panel().drop("aggregated_quantity")
        result = geks_jevons(df, balanced=False)

        # Pairwise matched-sample Jevons links:
        # J(0,1) on {A,B,C}: (1.05 * 1.05 * 1.04)^(1/3)
        # J(1,2) on {A,B,D}: ((110/105) * (220/210) * (380/400))^(1/3)
        # J(0,2) on {A,B}:   (1.10 * 1.10)^(1/2)
        J01 = (1.05 * 1.05 * 1.04) ** (1 / 3)
        J12 = ((110 / 105) * (220 / 210) * (380 / 400)) ** (1 / 3)
        J02 = (1.10 * 1.10) ** (1 / 2)

        # GEKS(1) = exp((1/3) * [2*ln J(0,1) + ln(J(0,2)/J(1,2))])
        expected_1 = np.exp((2 * np.log(J01) + np.log(J02 / J12)) / 3)
        # GEKS(2) = exp((1/3) * [2*ln J(0,2) + ln(J(1,2)*J(0,1))])
        expected_2 = np.exp((2 * np.log(J02) + np.log(J12 * J01)) / 3)

        assert result.filter(pl.col("period") == date(2023, 1, 1)).select("index_value").item() == pytest.approx(1.0, abs=1e-12)
        assert result.filter(pl.col("period") == date(2023, 2, 1)).select("index_value").item() == pytest.approx(expected_1, abs=1e-12)
        assert result.filter(pl.col("period") == date(2023, 3, 1)).select("index_value").item() == pytest.approx(expected_2, abs=1e-12)

    def test_two_periods_collapse_to_matched_bilateral(self):
        df = unbalanced_churn_panel().filter(
            pl.col("period").is_in([date(2023, 1, 1), date(2023, 2, 1)])
        )
        matched = df.filter(pl.col("product_id").is_in(["A", "B", "C"]))

        result = geks_fisher(df, balanced=False)
        val = result.filter(pl.col("period") == date(2023, 2, 1)).select("index_value").item()
        assert val == pytest.approx(fisher(matched), rel=1e-12)

    def test_complete_panel_balanced_flag_irrelevant(self):
        df = pl.DataFrame({
            "product_id": ["A", "A", "A", "B", "B", "B"],
            "period": [
                date(2023, 1, 1), date(2023, 2, 1), date(2023, 3, 1),
                date(2023, 1, 1), date(2023, 2, 1), date(2023, 3, 1)
            ],
            "aggregated_price": [100, 105, 110, 200, 210, 220],
            "aggregated_quantity": [10, 12, 14, 20, 22, 24],
        })
        strict = geks_fisher(df)
        relaxed = geks_fisher(df, balanced=False)
        assert relaxed.select("index_value").to_series().to_list() == pytest.approx(
            strict.select("index_value").to_series().to_list(), rel=1e-12
        )

    def test_default_raises_on_unbalanced(self):
        with pytest.raises(ValueError, match="Panel is not balanced"):
            geks_fisher(unbalanced_churn_panel())

    def test_zero_overlap_pair_skipped(self):
        # Periods 0 and 2 share no products; both share products with period 1.
        # J(0,1) = J(1,2) = 1.1, so the index is [1.0, 1.1, 1.21] with the
        # (0,2) link skipped and period 2 linked through period 1 only.
        df = pl.DataFrame({
            "product_id": ["A", "A", "B", "B", "C", "C", "D", "D"],
            "period": [
                date(2023, 1, 1), date(2023, 2, 1),
                date(2023, 1, 1), date(2023, 2, 1),
                date(2023, 2, 1), date(2023, 3, 1),
                date(2023, 2, 1), date(2023, 3, 1),
            ],
            "aggregated_price": [100, 110, 200, 220, 50, 55, 100, 110],
        })
        result = geks_jevons(df, balanced=False)

        values = result.sort("period").select("index_value").to_series().to_list()
        assert values[0] == pytest.approx(1.0, abs=1e-12)
        assert values[1] == pytest.approx(1.1, abs=1e-12)
        assert values[2] == pytest.approx(1.1 * 1.1, abs=1e-12)

    def test_zero_overlap_warning(self):
        df = pl.DataFrame({
            "product_id": ["A", "A", "B", "B", "C", "C", "D", "D"],
            "period": [
                date(2023, 1, 1), date(2023, 2, 1),
                date(2023, 1, 1), date(2023, 2, 1),
                date(2023, 2, 1), date(2023, 3, 1),
                date(2023, 2, 1), date(2023, 3, 1),
            ],
            "aggregated_price": [100, 110, 200, 220, 50, 55, 100, 110],
        })
        with pytest.warns(UserWarning, match="no common products were skipped"):
            geks_jevons(df, balanced=False, panel_warnings=True)

    def test_silent_by_default(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            geks_jevons(unbalanced_churn_panel().drop("aggregated_quantity"), balanced=False)

    def test_disconnected_period_raises(self):
        # Period 3 shares no products with any other period.
        df = pl.DataFrame({
            "product_id": ["A", "A", "A", "B", "B", "B", "X"],
            "period": [
                date(2023, 1, 1), date(2023, 2, 1), date(2023, 3, 1),
                date(2023, 1, 1), date(2023, 2, 1), date(2023, 3, 1),
                date(2023, 4, 1),
            ],
            "aggregated_price": [100, 105, 110, 200, 210, 220, 50],
        })
        with pytest.raises(ValueError, match="No common products between period"):
            geks_jevons(df, balanced=False)


# ---------------------------------------------------------------------------
# Unbalanced panels: Geary-Khamis
# ---------------------------------------------------------------------------


def gk_reference(rows, periods, tol=1e-12, max_iter=100000):
    """Independent plain-Python Geary-Khamis fixed point over observed cells.

    rows: iterable of (product_id, period_index, price, quantity).
    """
    cells = {(n, t): (p, q) for n, t, p, q in rows}
    products = sorted({n for n, _ in cells})
    T = len(periods)
    levels = [1.0] * T

    for _ in range(max_iter):
        alpha = {}
        for n in products:
            num = sum(q * p / levels[t] for (m, t), (p, q) in cells.items() if m == n)
            den = sum(q for (m, t), (p, q) in cells.items() if m == n)
            alpha[n] = num / den

        new_levels = []
        for t in range(T):
            num = sum(p * q for (m, s), (p, q) in cells.items() if s == t)
            den = sum(alpha[m] * q for (m, s), (p, q) in cells.items() if s == t)
            new_levels.append(num / den)
        new_levels = [v / new_levels[0] for v in new_levels]

        if max(abs(a - b) for a, b in zip(new_levels, levels)) < tol:
            levels = new_levels
            break
        levels = new_levels

    return dict(zip(periods, levels))


class TestGearyKhamisUnbalanced:
    """balanced=False runs the iteration over observed cells only."""

    def test_matches_independent_reimplementation(self):
        periods = [date(2023, 1, 1), date(2023, 2, 1), date(2023, 3, 1)]
        df = unbalanced_churn_panel()
        result = geary_khamis(df, balanced=False)

        rows = list(
            zip(
                df.select("product_id").to_series().to_list(),
                df.select("period").to_series().to_list(),
                df.select("aggregated_price").to_series().to_list(),
                df.select("aggregated_quantity").to_series().to_list(),
            )
        )
        rows = [(n, periods.index(t), p, q) for n, t, p, q in rows]
        expected = gk_reference(rows, periods)

        for period in periods:
            val = result.filter(pl.col("period") == period).select("index_value").item()
            assert val == pytest.approx(expected[period], abs=1e-7)

    def test_complete_panel_balanced_flag_irrelevant(self):
        df = pl.DataFrame({
            "product_id": ["A", "A", "A", "B", "B", "B"],
            "period": [
                date(2023, 1, 1), date(2023, 2, 1), date(2023, 3, 1),
                date(2023, 1, 1), date(2023, 2, 1), date(2023, 3, 1)
            ],
            "aggregated_price": [100, 105, 110, 200, 210, 220],
            "aggregated_quantity": [10, 12, 14, 20, 22, 24],
        })
        strict = geary_khamis(df)
        relaxed = geary_khamis(df, balanced=False)
        assert relaxed.select("index_value").to_series().to_list() == pytest.approx(
            strict.select("index_value").to_series().to_list(), rel=1e-10
        )

    def test_default_raises_clear_error_not_convergence(self):
        with pytest.raises(ValueError, match="Panel is not balanced"):
            geary_khamis(unbalanced_churn_panel())

    def test_panel_warning(self):
        with pytest.warns(UserWarning, match="Unbalanced panel"):
            geary_khamis(unbalanced_churn_panel(), balanced=False, panel_warnings=True)

    def test_silent_by_default(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            geary_khamis(unbalanced_churn_panel(), balanced=False)


# ---------------------------------------------------------------------------
# Unbalanced panels: Time Product Dummy (natively supported)
# ---------------------------------------------------------------------------


def tpd_reference(rows, periods, weighted=True):
    """Independent NumPy time-product-dummy estimator on observed rows.

    rows: iterable of (product_id, period, price, quantity).
    """
    products = sorted({n for n, _, _, _ in rows})
    rows = list(rows)

    X, y = [], []
    for n, t, p, _q in rows:
        X.append(
            [1.0]
            + [1.0 if t == s else 0.0 for s in periods[1:]]
            + [1.0 if n == m else 0.0 for m in products[1:]]
        )
        y.append(np.log(p))
    X = np.array(X)
    y = np.array(y)

    if weighted:
        exp_by_period = defaultdict(float)
        for n, t, p, q in rows:
            exp_by_period[t] += p * q
        w = np.array([p * q / exp_by_period[t] for n, t, p, q in rows])
        sw = np.sqrt(w)
        beta, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    else:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)

    out = {periods[0]: 1.0}
    for i, t in enumerate(periods[1:], start=1):
        out[t] = np.exp(beta[i])
    return out


class TestTimeProductDummyUnbalanced:
    """TPD handles unbalanced panels without any flag."""

    @pytest.mark.parametrize("weighted", [True, False])
    def test_matches_independent_reimplementation(self, weighted):
        df = unbalanced_churn_panel()
        result = time_product_dummy(df, weighted=weighted)

        periods = sorted(df.select("period").unique().to_series().to_list())
        rows = list(
            zip(
                df.select("product_id").to_series().to_list(),
                df.select("period").to_series().to_list(),
                df.select("aggregated_price").to_series().to_list(),
                df.select("aggregated_quantity").to_series().to_list(),
            )
        )
        expected = tpd_reference(rows, periods, weighted=weighted)

        for period in periods:
            val = result.filter(pl.col("period") == period).select("index_value").item()
            assert val == pytest.approx(expected[period], rel=1e-10)


# ---------------------------------------------------------------------------
# Expected values on the test dataset (monthly weighted-arithmetic
# aggregation). Two panels:
#   - balanced: all 5 products in all 15 periods (spot checks)
#   - churned:  drop product with sorted rank r in month with sorted rank k
#               whenever (r + k) % 4 == 0 (full series)
# ---------------------------------------------------------------------------

EXPECTED_BALANCED_SPOT = {
    # method: (mid_period, mid_value, last_period, last_value)
    "geks_fisher": ("2023-08-01", 1.22982132863481, "2024-03-01", 1.49511048480412),
    "geks_tornqvist": ("2023-08-01", 1.23034861859828, "2024-03-01", 1.49619569200226),
    "geks_jevons": ("2023-08-01", 1.05109954733425, "2024-03-01", 1.21871215869637),
    "geary_khamis": ("2023-09-01", 1.28250809348126, "2024-03-01", 1.50486848971153),
    "time_product_dummy": ("2023-09-01", 1.2679308147982, "2024-03-01", 1.49802178690056),
}

EXPECTED_CHURNED_SERIES = {
    "geks_fisher": {
        "2023-01-01": 1.0,
        "2023-02-01": 1.24479815171281,
        "2023-03-01": 1.11648086164102,
        "2023-04-01": 1.14940362515075,
        "2023-05-01": 1.17368053473769,
        "2023-06-01": 1.2905550128231,
        "2023-07-01": 1.24830535712109,
        "2023-08-01": 1.26562592721059,
        "2023-09-01": 1.31246600251257,
        "2023-10-01": 1.24050878153592,
        "2023-11-01": 1.38933160028401,
        "2023-12-01": 1.40386516413222,
        "2024-01-01": 1.477966053078,
        "2024-02-01": 1.280591529228,
        "2024-03-01": 1.54506751345135,
    },
    "geks_tornqvist": {
        "2023-01-01": 1.0,
        "2023-02-01": 1.24621391522804,
        "2023-03-01": 1.11770176666773,
        "2023-04-01": 1.1506254288527,
        "2023-05-01": 1.17462977750055,
        "2023-06-01": 1.29199621912076,
        "2023-07-01": 1.24983395706965,
        "2023-08-01": 1.26745974694748,
        "2023-09-01": 1.31437472965381,
        "2023-10-01": 1.24200250612919,
        "2023-11-01": 1.39118481148114,
        "2023-12-01": 1.40629801767003,
        "2024-01-01": 1.47996866300987,
        "2024-02-01": 1.28198159477284,
        "2024-03-01": 1.54760784138611,
    },
    "geks_jevons": {
        "2023-01-01": 1.0,
        "2023-02-01": 1.12120581106693,
        "2023-03-01": 1.02261144306399,
        "2023-04-01": 1.09976792455112,
        "2023-05-01": 1.10245775658895,
        "2023-06-01": 1.14159240670195,
        "2023-07-01": 1.10463646580715,
        "2023-08-01": 1.08124692212774,
        "2023-09-01": 1.09324470015528,
        "2023-10-01": 1.06137028195058,
        "2023-11-01": 1.19597017890127,
        "2023-12-01": 1.15205741395831,
        "2024-01-01": 1.20855362643572,
        "2024-02-01": 1.15499505701703,
        "2024-03-01": 1.30527835872509,
    },
    "geary_khamis": {
        "2023-02-01": 1.30165364951495,
        "2023-03-01": 1.15779527172799,
        "2023-04-01": 1.19626824417371,
        "2023-05-01": 1.22313027932229,
        "2023-06-01": 1.35009224084543,
        "2023-07-01": 1.32033580775677,
        "2023-08-01": 1.34737673511709,
        "2023-09-01": 1.41615412223827,
        "2023-10-01": 1.29295415545492,
        "2023-11-01": 1.48462587762115,
        "2023-12-01": 1.49902281533839,
        "2024-01-01": 1.62180762171154,
        "2024-02-01": 1.32346177629277,
        "2024-03-01": 1.66547484838053,
    },
    "time_product_dummy": {
        "2023-02-01": 1.32293894871495,
        "2023-03-01": 1.16263992818967,
        "2023-04-01": 1.19398532455792,
        "2023-05-01": 1.21516876376516,
        "2023-06-01": 1.37318465004219,
        "2023-07-01": 1.3179434827645,
        "2023-08-01": 1.34290947710922,
        "2023-09-01": 1.40938727126452,
        "2023-10-01": 1.31614307221276,
        "2023-11-01": 1.4854132658548,
        "2023-12-01": 1.50168252669973,
        "2024-01-01": 1.61820751612393,
        "2024-02-01": 1.34532480362881,
        "2024-03-01": 1.67172001536247,
    },
}

TOLERANCE_EXPECTED = 1e-6


@pytest.fixture
def balanced_panel(test_data_raw):
    df = standardize_columns(
        test_data_raw,
        date_col="date",
        price_col="price",
        id_col="product_id",
        quantity_col="quantity",
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
    return remove_unbalanced(agg)


@pytest.fixture
def churned_panel(balanced_panel):
    prods = sorted(balanced_panel["product_id"].unique().to_list())
    months = sorted(balanced_panel["period"].unique().to_list())
    r = pl.Series([prods.index(p) + 1 for p in balanced_panel["product_id"].to_list()])
    k = pl.Series([months.index(m) + 1 for m in balanced_panel["period"].to_list()])
    return balanced_panel.filter(((r + k) % 4) != 0)


class TestExpectedValues:
    """Multilateral indices against expected values, balanced and churned."""

    FUNCS = {
        "geks_fisher": lambda df, balanced: geks_fisher(df, balanced=balanced),
        "geks_tornqvist": lambda df, balanced: geks_tornqvist(df, balanced=balanced),
        "geks_jevons": lambda df, balanced: geks_jevons(df, balanced=balanced),
        "geary_khamis": lambda df, balanced: geary_khamis(df, balanced=balanced),
        "time_product_dummy": lambda df, balanced: time_product_dummy(df, weighted=True),
    }

    @pytest.mark.parametrize("method", list(EXPECTED_BALANCED_SPOT))
    def test_balanced_spot_values(self, balanced_panel, method):
        mid_p, mid_v, last_p, last_v = EXPECTED_BALANCED_SPOT[method]
        result = self.FUNCS[method](balanced_panel, balanced=True)
        by_period = {
            p.strftime("%Y-%m-%d"): v
            for p, v in zip(result["period"].to_list(), result["index_value"].to_list())
        }
        assert abs(by_period[mid_p] - mid_v) < TOLERANCE_EXPECTED
        assert abs(by_period[last_p] - last_v) < TOLERANCE_EXPECTED

    @pytest.mark.parametrize("method", list(EXPECTED_CHURNED_SERIES))
    def test_churned_series(self, churned_panel, method):
        result = self.FUNCS[method](churned_panel, balanced=False)
        by_period = {
            p.strftime("%Y-%m-%d"): v
            for p, v in zip(result["period"].to_list(), result["index_value"].to_list())
        }
        for period, expected in EXPECTED_CHURNED_SERIES[method].items():
            assert abs(by_period[period] - expected) < TOLERANCE_EXPECTED


# ---------------------------------------------------------------------------
# Robustness: seeded random unbalanced panels through every method
# ---------------------------------------------------------------------------


class TestRandomPanelRobustness:
    """Random unbalanced panels produce finite, positive, base-1.0 series."""

    @staticmethod
    def _random_panel(seed, n_products=12, n_periods=8, keep_prob=0.8):
        rng = np.random.default_rng(seed)
        periods = [date(2023 + i // 12, i % 12 + 1, 1) for i in range(n_periods)]
        rows = {"product_id": [], "period": [],
                "aggregated_price": [], "aggregated_quantity": []}
        for n in range(n_products):
            for t, period in enumerate(periods):
                if rng.random() > keep_prob:
                    continue
                rows["product_id"].append(f"p{n}")
                rows["period"].append(period)
                rows["aggregated_price"].append(float(np.exp(rng.normal(2, 0.3))))
                rows["aggregated_quantity"].append(float(rng.integers(1, 50)))
        return pl.DataFrame(rows), periods

    @pytest.mark.parametrize("seed", [1, 2, 3])
    def test_geks_methods(self, seed):
        df, periods = self._random_panel(seed)
        for func, kwargs in [
            (geks_fisher, {"quantity_col": "aggregated_quantity"}),
            (geks_tornqvist, {"quantity_col": "aggregated_quantity"}),
            (geks_jevons, {}),
        ]:
            result = func(df, balanced=False, **kwargs)
            assert result.height == len(periods)
            values = result.sort("period").select("index_value").to_series().to_list()
            assert all(np.isfinite(v) and v > 0 for v in values)
            assert values[0] == pytest.approx(1.0)

    @pytest.mark.parametrize("seed", [1, 2, 3])
    def test_tpd(self, seed):
        df, periods = self._random_panel(seed)
        result = time_product_dummy(df, weighted=True)
        values = result.sort("period").select("index_value").to_series().to_list()
        assert all(np.isfinite(v) and v > 0 for v in values)
        assert values[0] == pytest.approx(1.0)

    @pytest.mark.parametrize("seed", [1, 2, 3])
    def test_geary_khamis(self, seed):
        df, periods = self._random_panel(seed)
        try:
            result = geary_khamis(df, balanced=False)
        except ValueError as e:
            # Convergence is not guaranteed on arbitrary unbalanced panels
            assert "did not converge" in str(e)
            return
        values = result.sort("period").select("index_value").to_series().to_list()
        assert all(np.isfinite(v) and v > 0 for v in values)
        assert values[0] == pytest.approx(1.0)

    @pytest.mark.parametrize("seed", [1, 2, 3])
    def test_bilateral_on_random_two_periods(self, seed):
        from pyindexnum.bilateral import (
            jevons, carli, dutot, laspeyres, paasche, fisher, tornqvist, walsh,
        )
        df, periods = self._random_panel(seed)
        two = df.filter(pl.col("period").is_in(periods[:2]))
        for func in [jevons, carli, dutot, laspeyres, paasche, fisher, tornqvist, walsh]:
            value = func(two, balanced=False)
            assert np.isfinite(value) and value > 0


# ---------------------------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------------------------


class TestValidation:
    """Test input validation for multilateral functions."""

    def test_missing_columns(self):
        """Test validation with missing columns."""
        df = pl.DataFrame({
            "product_id": ["A", "B"],
            "period": [date(2023, 1, 1), date(2023, 1, 1)],
            "price": [100, 200]  # Wrong column name
        })

        with pytest.raises(ValueError, match="Missing required columns"):
            geks_fisher(df)

    def test_zero_price(self):
        """Test validation with zero prices."""
        df = pl.DataFrame({
            "product_id": ["A", "B"],
            "period": [date(2023, 1, 1), date(2023, 1, 1)],
            "aggregated_price": [0, 200],
            "aggregated_quantity": [10, 20]
        })

        with pytest.raises(ValueError, match="must be positive"):
            geks_fisher(df)

    def test_duplicate_observations(self):
        """Test validation with duplicate product-period observations."""
        df = pl.DataFrame({
            "product_id": ["A", "A", "B"],
            "period": [date(2023, 1, 1), date(2023, 1, 1), date(2023, 1, 1)],
            "aggregated_price": [100, 105, 200],
            "aggregated_quantity": [10, 10, 20]
        })

        with pytest.raises(ValueError, match="exactly one observation per period"):
            geks_fisher(df)
