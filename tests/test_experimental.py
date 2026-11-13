"""
Tests for the experimental spatial multilateral index functions.

Covers the bridge-form GEKS-Paasche and GEKS-Laspeyres indices: exact
hand-computed values on an asymmetric-quantity fixture, structural and
axiomatic properties, matched-sample behaviour on unbalanced panels,
input validation, and the generic coverage (custom column names, the
rename bridge from the temporal pipeline, random-panel robustness) that
the standard MULTILATERAL_FUNCS sweep would otherwise provide.
"""

import math
from datetime import date, timedelta

import pytest
import polars as pl
import numpy as np

from pyindexnum import geks_paasche, geks_laspeyres, geks_fisher
from pyindexnum.bilateral import fisher

TOL = 1e-10


@pytest.fixture
def asym_data():
    """3 regions x 2 products with asymmetric quantities (P != L != F).

    Region A: p = (1, 1), q = (10, 10)
    Region B: p = (2, 1), q = (5, 20)
    Region C: p = (1, 3), q = (8, 4)
    """
    return pl.DataFrame({
        "region": ["A", "B", "C", "A", "B", "C"],
        "product_id": ["X", "X", "X", "Y", "Y", "Y"],
        "aggregated_price": [1.0, 2.0, 1.0, 1.0, 1.0, 3.0],
        "aggregated_quantity": [10.0, 5.0, 8.0, 10.0, 20.0, 4.0],
    })


@pytest.fixture
def const_quantity_data():
    """3 regions x 3 products with constant quantities: P = L = Fisher."""
    return pl.DataFrame({
        "region": ["R1", "R2", "R3", "R1", "R2", "R3", "R1", "R2", "R3"],
        "product_id": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
        "aggregated_price": [100, 105, 110, 200, 210, 220, 50, 52, 54],
        "aggregated_quantity": [10, 10, 10, 20, 20, 20, 5, 5, 5],
    })


@pytest.fixture
def churn_data():
    """Regions with partially overlapping product sets (no empty pairs).

    A and B share {X, Y}; A and C share {X}; B and C share {X}; C also
    carries Z, which no other region has.
    """
    return pl.DataFrame({
        "region": ["A", "A", "B", "B", "C", "C"],
        "product_id": ["X", "Y", "X", "Y", "X", "Z"],
        "aggregated_price": [1.0, 1.0, 2.0, 1.0, 1.0, 2.0],
        "aggregated_quantity": [10.0, 10.0, 5.0, 20.0, 8.0, 2.0],
    })


def spatial_reference(data, formula, base="A"):
    """Plain-Python bridge-form reference for the spatial GEKS indices.

    index(k) = exp( mean_m [ ln B(m, k) - ln B(m, base) ] ) over bridges m
    whose both legs have matched products, where B is the directed
    matched-sample Paasche/Laspeyres bilateral index.
    """
    prices = {}
    quantities = {}
    for r, i, p, q in zip(
        data["region"], data["product_id"],
        data["aggregated_price"], data["aggregated_quantity"],
    ):
        prices[(r, i)] = p
        quantities[(r, i)] = q

    regions = sorted(set(r for r, _ in prices))
    products = sorted(set(i for _, i in prices))

    def bilateral(m, k):
        common = [i for i in products if (m, i) in prices and (k, i) in prices]
        if not common:
            return None
        if formula == "paasche":
            num = sum(prices[(k, i)] * quantities[(k, i)] for i in common)
            den = sum(prices[(m, i)] * quantities[(k, i)] for i in common)
        else:
            num = sum(prices[(k, i)] * quantities[(m, i)] for i in common)
            den = sum(prices[(m, i)] * quantities[(m, i)] for i in common)
        return num / den

    out = {}
    for k in regions:
        log_sum, n_links = 0.0, 0
        for m in regions:
            b_mk = 1.0 if m == k else bilateral(m, k)
            b_mb = 1.0 if m == base else bilateral(m, base)
            if b_mk is None or b_mb is None:
                continue
            log_sum += math.log(b_mk) - math.log(b_mb)
            n_links += 1
        if n_links == 0:
            out[k] = None
        else:
            out[k] = math.exp(log_sum / n_links)
    return out


class TestSpatialGEKSPaasche:
    """Test the spatial GEKS-Paasche index."""

    def test_structure(self, asym_data):
        """Test output schema, height and base entity normalisation."""
        result = geks_paasche(asym_data)

        assert isinstance(result, pl.DataFrame)
        assert result.columns == ["region", "geks_paasche"]
        assert result.height == 3
        base_value = (
            result.filter(pl.col("region") == "A").select("geks_paasche").item()
        )
        assert base_value == pytest.approx(1.0)

    def test_hand_computed_values(self, asym_data):
        """Test exact hand-computed values on the asymmetric fixture.

        geks_paasche(B) = [P_P(C,B) * 1 * P_P(A,B) / (P_P(B,A) * 1 * P_P(C,A))]^(1/3)
                        = [(6/5 * 1 * 6/13) / (2/3 * 1 * 1/2)]^(1/3)
                        = (108/65)^(1/3)
        geks_paasche(C) = [(5/3 * 1 * 1) / (2/3 * 1 * 1/2)]^(1/3) = 5^(1/3)

        These exact constants pin the corrected naming: with the draft's
        swapped orientation, geks_paasche(B) would instead equal 3^(1/3).
        """
        result = geks_paasche(asym_data)

        expected = {"A": 1.0, "B": (108 / 65) ** (1 / 3), "C": 5 ** (1 / 3)}
        for region, value in zip(result["region"], result["geks_paasche"]):
            assert value == pytest.approx(expected[region], rel=TOL)

    def test_matches_reference_implementation(self, asym_data, churn_data):
        """Test agreement with the independent plain-Python reference."""
        for data in (asym_data, churn_data):
            result = geks_paasche(data)
            reference = spatial_reference(data, "paasche")
            for region, value in zip(result["region"], result["geks_paasche"]):
                assert value == pytest.approx(reference[region], rel=TOL)


class TestSpatialGEKSLaspeyres:
    """Test the spatial GEKS-Laspeyres index."""

    def test_structure(self, asym_data):
        """Test output schema, height and base entity normalisation."""
        result = geks_laspeyres(asym_data)

        assert isinstance(result, pl.DataFrame)
        assert result.columns == ["region", "geks_laspeyres"]
        assert result.height == 3
        base_value = (
            result.filter(pl.col("region") == "A").select("geks_laspeyres").item()
        )
        assert base_value == pytest.approx(1.0)

    def test_hand_computed_values(self, asym_data):
        """Test exact hand-computed values on the asymmetric fixture.

        geks_laspeyres(B) = [P_P(A,B) * P_P(A,C) / (P_P(B,A) * P_P(B,C))]^(1/3)
                          = [(6/5 * 5/3) / (2/3 * 1)]^(1/3) = 3^(1/3)
        geks_laspeyres(C) = [(6/5 * 5/3) / (1/2 * 6/13)]^(1/3) = (26/3)^(1/3)

        This is Bialek's (2022) GEKS-L applied over regions: every bridge
        comparison uses the bridge's basket.
        """
        result = geks_laspeyres(asym_data)

        expected = {"A": 1.0, "B": 3 ** (1 / 3), "C": (26 / 3) ** (1 / 3)}
        for region, value in zip(result["region"], result["geks_laspeyres"]):
            assert value == pytest.approx(expected[region], rel=TOL)

    def test_matches_reference_implementation(self, asym_data, churn_data):
        """Test agreement with the independent plain-Python reference."""
        for data in (asym_data, churn_data):
            result = geks_laspeyres(data)
            reference = spatial_reference(data, "laspeyres")
            for region, value in zip(result["region"], result["geks_laspeyres"]):
                assert value == pytest.approx(reference[region], rel=TOL)


class TestSpatialProperties:
    """Axiomatic properties of the spatial bridge-form indices."""

    def test_two_entities_reduce_to_bilateral_fisher(self, asym_data):
        """With M = 2 both indices collapse to the bilateral Fisher index.

        geks_X(B) = [B(A,B) / B(B,A)]^(1/2) = sqrt(P_P(A,B) * P_L(A,B))
        """
        ab = asym_data.filter(pl.col("region").is_in(["A", "B"]))

        paasche_b = (
            geks_paasche(ab)
            .filter(pl.col("region") == "B")
            .select("geks_paasche")
            .item()
        )
        laspeyres_b = (
            geks_laspeyres(ab)
            .filter(pl.col("region") == "B")
            .select("geks_laspeyres")
            .item()
        )
        fisher_b = fisher(ab, period_col="region")

        assert paasche_b == pytest.approx(math.sqrt(1.8), rel=TOL)
        assert laspeyres_b == pytest.approx(math.sqrt(1.8), rel=TOL)
        assert paasche_b == pytest.approx(fisher_b, rel=TOL)
        assert laspeyres_b == pytest.approx(fisher_b, rel=TOL)

    def test_constant_quantities_match_geks_fisher_and_bilateral(
        self, const_quantity_data
    ):
        """With constant quantities P = L = Fisher, so all methods coincide.

        R2 = (105*10 + 210*20 + 52*5) / (100*10 + 200*20 + 50*5)
           = 5510 / 5250
        """
        paasche = geks_paasche(const_quantity_data)
        laspeyres = geks_laspeyres(const_quantity_data)
        fisher_result = geks_fisher(const_quantity_data, period_col="region")

        expected = {"R1": 1.0, "R2": 5510 / 5250, "R3": 5770 / 5250}
        for region, p_val, l_val in zip(
            paasche["region"], paasche["geks_paasche"], laspeyres["geks_laspeyres"]
        ):
            assert p_val == pytest.approx(expected[region], abs=1e-6)
            assert l_val == pytest.approx(expected[region], abs=1e-6)

        fisher_values = dict(zip(fisher_result["period"], fisher_result["index_value"]))
        for region, p_val in zip(paasche["region"], paasche["geks_paasche"]):
            assert p_val == pytest.approx(fisher_values[region], rel=1e-10)

    def test_base_invariance(self, asym_data):
        """Changing the base entity rescales all values by a constant."""
        base_a = geks_paasche(asym_data, base_entity="A")
        base_b = geks_paasche(asym_data, base_entity="B")

        base_b_value = base_b.filter(
            pl.col("region") == "B"
        ).select("geks_paasche").item()
        assert base_b_value == pytest.approx(1.0)

        # index_A(k) / index_B(k) = level(B) / level(A) must be constant in k
        ratios = [
            a / b
            for a, b in zip(base_a["geks_paasche"], base_b["geks_paasche"])
            if b != 1.0
        ]
        assert ratios[0] == pytest.approx(
            base_a.filter(pl.col("region") == "B").select("geks_paasche").item(),
            rel=1e-12,
        )
        for r in ratios[1:]:
            assert r == pytest.approx(ratios[0], rel=1e-12)

    def test_parities_independent_of_base(self, asym_data):
        """Transitivity: index(C)/index(B) is the same whatever the base."""
        base_a = geks_laspeyres(asym_data, base_entity="A")
        base_b = geks_laspeyres(asym_data, base_entity="B")

        vals_a = dict(zip(base_a["region"], base_a["geks_laspeyres"]))
        vals_b = dict(zip(base_b["region"], base_b["geks_laspeyres"]))

        assert vals_a["C"] / vals_a["B"] == pytest.approx(vals_b["C"], rel=1e-12)

    def test_distinct_from_geks_fisher_asymmetric(self, asym_data):
        """With asymmetric weights the two indices differ from GEKS-Fisher.

        On this fixture: geks_laspeyres(B) > geks_fisher(B) > geks_paasche(B).
        """
        paasche_b = (
            geks_paasche(asym_data)
            .filter(pl.col("region") == "B")
            .select("geks_paasche")
            .item()
        )
        laspeyres_b = (
            geks_laspeyres(asym_data)
            .filter(pl.col("region") == "B")
            .select("geks_laspeyres")
            .item()
        )
        fisher_b = (
            geks_fisher(asym_data, period_col="region")
            .filter(pl.col("period") == "B")
            .select("index_value")
            .item()
        )

        assert laspeyres_b > fisher_b > paasche_b
        assert abs(laspeyres_b - fisher_b) > 1e-3
        assert abs(paasche_b - fisher_b) > 1e-3


class TestUnbalancedSpatial:
    """Matched-sample behaviour on unbalanced panels."""

    def test_matched_sample_hand_computed(self, churn_data):
        """Each link runs on its matched products; exact closed forms.

        geks_paasche(B)  = (1.2 * 1.5 * 2.0)^(1/3) = 3.6^(1/3)
        geks_paasche(C)  = (1.0 * 0.75 * 1.0)^(1/3) = 0.75^(1/3)
        geks_laspeyres(B) = (1.5 * 1.2 * 2.0)^(1/3) = 3.6^(1/3)
        geks_laspeyres(C) = (1.0 * 0.6 * 1.0)^(1/3) = 0.6^(1/3)
        """
        paasche = dict(zip(
            geks_paasche(churn_data)["region"],
            geks_paasche(churn_data)["geks_paasche"],
        ))
        laspeyres = dict(zip(
            geks_laspeyres(churn_data)["region"],
            geks_laspeyres(churn_data)["geks_laspeyres"],
        ))

        assert paasche["B"] == pytest.approx(3.6 ** (1 / 3), rel=TOL)
        assert paasche["C"] == pytest.approx(0.75 ** (1 / 3), rel=TOL)
        assert laspeyres["B"] == pytest.approx(3.6 ** (1 / 3), rel=TOL)
        assert laspeyres["C"] == pytest.approx(0.6 ** (1 / 3), rel=TOL)

    def test_unmatched_products_warning(self, churn_data):
        """panel_warnings=True summarises unmatched products."""
        with pytest.warns(UserWarning, match="unmatched"):
            geks_paasche(churn_data, panel_warnings=True)

        with pytest.warns(UserWarning, match="unmatched"):
            geks_laspeyres(churn_data, panel_warnings=True)

    def test_skipped_pairs_warning(self, churn_data):
        """Entity pairs with no common products are skipped with a warning.

        Region D only shares product Z with C: A-D and B-D have no common
        products and are skipped. Because every bridge into an entity needs
        links to the base entity as well, and D has no link to the base A,
        the D bridge never enters any average: A, B and C keep their values
        unchanged while D still receives an index through its C link.
        """
        extra = pl.DataFrame({
            "region": ["D"],
            "product_id": ["Z"],
            "aggregated_price": [3.0],
            "aggregated_quantity": [6.0],
        })
        data = pl.concat([churn_data, extra])

        base = geks_laspeyres(churn_data)
        with pytest.warns(UserWarning, match="no common products"):
            result = geks_laspeyres(data, panel_warnings=True)

        assert result.height == 4
        base_values = dict(zip(base["region"], base["geks_laspeyres"]))
        result_values = dict(zip(result["region"], result["geks_laspeyres"]))
        for region in ("A", "B", "C"):
            assert result_values[region] == pytest.approx(
                base_values[region], rel=1e-12
            )
        assert np.isfinite(result_values["D"]) and result_values["D"] > 0

    def test_isolated_entity_raises(self, churn_data):
        """An entity sharing no products with any other raises ValueError."""
        isolated = pl.DataFrame({
            "region": ["E", "E"],
            "product_id": ["W", "V"],
            "aggregated_price": [1.0, 2.0],
            "aggregated_quantity": [3.0, 4.0],
        })
        data = pl.concat([churn_data, isolated])

        with pytest.raises(ValueError, match="No common products between entity E"):
            geks_paasche(data)

        with pytest.raises(ValueError, match="No common products between entity E"):
            geks_laspeyres(data)


class TestValidation:
    """Input validation for the spatial indices."""

    def test_missing_columns(self, asym_data):
        with pytest.raises(ValueError, match="Missing required columns"):
            geks_paasche(asym_data.drop("aggregated_quantity"))

    def test_requires_at_least_two_entities(self):
        df = pl.DataFrame({
            "region": ["A", "A"],
            "product_id": ["X", "Y"],
            "aggregated_price": [1.0, 2.0],
            "aggregated_quantity": [1.0, 1.0],
        })
        with pytest.raises(ValueError, match="at least 2 entities"):
            geks_paasche(df)

    def test_requires_at_least_two_products(self):
        df = pl.DataFrame({
            "region": ["A", "B"],
            "product_id": ["X", "X"],
            "aggregated_price": [1.0, 2.0],
            "aggregated_quantity": [1.0, 1.0],
        })
        with pytest.raises(ValueError, match="at least 2 products"):
            geks_laspeyres(df)

    def test_duplicate_observation_per_entity(self):
        df = pl.DataFrame({
            "region": ["A", "A", "B", "B"],
            "product_id": ["X", "X", "X", "Y"],
            "aggregated_price": [1.0, 1.5, 2.0, 1.0],
            "aggregated_quantity": [1.0, 1.0, 1.0, 1.0],
        })
        with pytest.raises(ValueError, match="exactly one observation per entity"):
            geks_paasche(df)

    def test_non_positive_price(self, asym_data):
        bad = asym_data.with_columns(
            pl.when(pl.col("region") == "C")
            .then(0.0)
            .otherwise(pl.col("aggregated_price"))
            .alias("aggregated_price")
        )
        with pytest.raises(
            ValueError, match="aggregated_price values must be positive"
        ):
            geks_paasche(bad)

    def test_non_positive_quantity(self, asym_data):
        bad = asym_data.with_columns(
            pl.when(pl.col("region") == "B")
            .then(-1.0)
            .otherwise(pl.col("aggregated_quantity"))
            .alias("aggregated_quantity")
        )
        with pytest.raises(
            ValueError, match="aggregated_quantity values must be positive"
        ):
            geks_laspeyres(bad)

    def test_non_numeric_price(self, asym_data):
        bad = asym_data.with_columns(pl.col("aggregated_price").cast(pl.String))
        with pytest.raises(ValueError, match="aggregated_price must be numeric"):
            geks_paasche(bad)

    def test_unknown_base_entity(self, asym_data):
        with pytest.raises(ValueError, match="not found"):
            geks_paasche(asym_data, base_entity="Z")


class TestGenericCoverage:
    """Generic coverage replicating the MULTILATERAL_FUNCS sweep."""

    def test_custom_column_names(self, asym_data):
        """Custom column names produce identical results to canonical names."""
        renamed = asym_data.rename({
            "region": "territory",
            "product_id": "sku",
            "aggregated_price": "price",
            "aggregated_quantity": "qty",
        })

        for func, value_name in (
            (geks_paasche, "geks_paasche"),
            (geks_laspeyres, "geks_laspeyres"),
        ):
            result = func(
                renamed,
                entity_col="territory",
                id_col="sku",
                price_col="price",
                quantity_col="qty",
            )
            canonical = func(asym_data)

            assert result.columns == ["territory", value_name]
            assert result[value_name].to_list() == canonical[value_name].to_list()

    def test_rename_bridge_from_temporal_pipeline(self, asym_data):
        """The documented rename step bridges the temporal pipeline.

        standardize_columns -> aggregate_time emits a "period" column;
        renaming it to "region" makes the spatial defaults apply. The
        result equals calling with entity_col="period" directly.
        """
        temporal = asym_data.rename({"region": "period"})

        via_rename = geks_paasche(temporal.rename({"period": "region"}))
        direct = geks_paasche(temporal, entity_col="period")

        assert via_rename.columns == ["region", "geks_paasche"]
        assert direct.columns == ["period", "geks_paasche"]
        assert via_rename["geks_paasche"].to_list() == direct["geks_paasche"].to_list()

    @pytest.mark.parametrize("seed", [1, 2, 3])
    def test_random_panel_robustness(self, seed):
        """Seeded random balanced panels: finite, positive, base = 1.0."""
        rng = np.random.default_rng(seed)
        regions = ["R1", "R2", "R3", "R4", "R5"]
        products = ["P1", "P2", "P3", "P4", "P5", "P6"]
        df = pl.DataFrame({
            "region": [r for r in regions for _ in products],
            "product_id": products * len(regions),
            "aggregated_price": rng.uniform(
                0.5, 3.0, len(regions) * len(products)
            ).round(4),
            "aggregated_quantity": rng.uniform(
                1.0, 20.0, len(regions) * len(products)
            ).round(4),
        })

        for func, value_name in (
            (geks_paasche, "geks_paasche"),
            (geks_laspeyres, "geks_laspeyres"),
        ):
            result = func(df)

            assert result.columns == ["region", value_name]
            assert result.height == len(regions)
            values = result[value_name].to_list()
            assert all(np.isfinite(v) and v > 0 for v in values)
            assert values[0] == pytest.approx(1.0)
            # Asymmetric random weights keep the two methods distinct
            assert values != func(df, base_entity="R3")[value_name].to_list()

    def test_date_entities_supported(self, asym_data):
        """Entity labels may be dates (any sortable dtype works)."""
        offsets = [0, 31, 62]
        dates = [date(2023, 1, 1) + timedelta(days=d) for d in offsets] * 2
        dated = asym_data.with_columns(pl.Series("region", dates, dtype=pl.Date))

        result = geks_laspeyres(dated)
        string_result = geks_laspeyres(asym_data)

        assert result.columns == ["region", "geks_laspeyres"]
        assert result.height == 3
        assert result["geks_laspeyres"][0] == pytest.approx(1.0)
        ours = result["geks_laspeyres"].to_list()
        theirs = string_result["geks_laspeyres"].to_list()
        assert ours == theirs
