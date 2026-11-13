# Experimental: Spatial Price Indices

This page documents `geks_paasche` and `geks_laspeyres`, multilateral price
indices designed for **spatial** (cross-region) comparisons: measuring price
levels of several regions against a common base region in a single
calculation, the way purchasing power parities compare countries.

```{note}
These functions live in the `pyindexnum.experimental` module and their API is
experimental. It intentionally deviates from the package's temporal
conventions: an `entity_col` defaulting to `"region"` instead of `period_col`,
an explicit `base_entity` argument, and value columns named after the method
(`geks_paasche`/`geks_laspeyres`) instead of `index_value`. The API may change
in future releases.
```

## Why spatial comparisons need a different construction

The classic GEKS combination of bilateral indices presupposes a bilateral
formula that satisfies time (country) reversal, `P(j, k) * P(k, j) = 1`, as
Fisher and Törnqvist do. Paasche and Laspeyres indices use asymmetric
expenditure weights and fail this test, so applying the traditional GEKS
recipe to them is not conceptually sound. In particular, the symmetric
least-squares regression formulation

```text
ln P(j, k) = pi_k - pi_j + u        (Rao & Timmer 2003; Diewert 1999)
```

reproduces the **GEKS-Fisher parities exactly** when fed Paasche or Laspeyres
bilateral indices (van IJzeren 1987, pp. 62–65, reported in Diewert 1999,
fn. 28) — a faithful regression implementation would simply duplicate
`geks_fisher`.

## The bridge formulation

Both functions instead use the one-sided **bridge form**: each term
`ln[P(m, k) / P(m, b)]` is the two-hop parity estimate of region `k` against
the base region `b` through bridge region `m`, and the index is the average
of these estimates over all bridges:

```text
geks_laspeyres(k) = prod_m [ P_L(m, k) / P_L(m, b) ]^(1/M)
geks_paasche(k)   = prod_m [ P_P(m, k) / P_P(m, b) ]^(1/M)
```

where `P(m, k)` is the bilateral index with `m` as base and `k` as current,
computed on the products matched between the two regions, and `M` is the
number of regions.

- `geks_laspeyres` is the GEKS-L index of Białek (2022) applied over regions:
  every bridge comparison uses the bridge region's basket.
- `geks_paasche` is its Paasche mirror image: every comparison into a region
  uses that region's own basket.

Properties (both indices):

- **Transitive** and **base invariant** — no country reversal required,
  because the result always has the form `level(k) / level(b)`.
- **Distinct from GEKS-Fisher** whenever regional weights differ.
- **Reduce to the bilateral Fisher index** when only two regions are
  compared, and coincide with `geks_fisher` (and with each other) when
  quantities are identical across regions.

## Usage

Spatial data with a region column works out of the box with the defaults:

```python
import polars as pl
import pyindexnum as pin

df_region = pl.DataFrame({
    "region": ["North", "North", "Centre", "Centre", "South", "South"],
    "product_id": ["bread", "fuel", "bread", "fuel", "bread", "fuel"],
    "aggregated_price": [1.0, 1.0, 2.0, 1.0, 1.0, 3.0],
    "aggregated_quantity": [10.0, 10.0, 5.0, 20.0, 8.0, 4.0],
})

paasche = pin.geks_paasche(df_region, base_entity="North")
laspeyres = pin.geks_laspeyres(df_region, base_entity="North")
```

```text
shape: (3, 2)                shape: (3, 2)
┌────────┬──────────────┐   ┌────────┬────────────────┐
│ region ┆ geks_paasche │   │ region ┆ geks_laspeyres │
╞════════╪══════════════╡   ╞════════╪════════════════╡
│ Centre ┆ 1.184414     │   │ Centre ┆ 1.44225        │
│ North  ┆ 1.0          │   │ North  ┆ 1.0            │
│ South  ┆ 1.709976     │   │ South  ┆ 2.05408        │
└────────┴──────────────┘   └────────┴────────────────┘
```

If `base_entity` is not given, the first sorted region (alphabetically first
for string labels) is used as the base.

### Bridging from the temporal pipeline

The standard pipeline (`standardize_columns` → `aggregate_time`) emits a
`period` column. To feed the same aggregated panel to the spatial functions,
rename that column to `region` first:

```python
df_agg = pin.aggregate_time(pin.standardize_columns(df_raw))

df_region = df_agg.rename({"period": "region"})
paasche = pin.geks_paasche(df_region)
```

(This is exactly the `entity_col` argument in action — any entity column name
can be passed instead of renaming.)

## Unbalanced panels

Spatial panels are often unbalanced: regions carry different product
assortments. The spatial indices always compute on matched product sets —
there is no `balanced` flag:

- each bilateral link uses the products matched between its two regions;
- each region's bridge average runs over the bridges available for it;
- region pairs sharing no products are skipped, and a region connected to no
  other region raises an error;
- pass `panel_warnings=True` for a summary of unmatched products and skipped
  pairs.

## Output schema

The result is a DataFrame with columns `[entity_col, "geks_paasche"]` (or
`"geks_laspeyres"`), with the base region equal to 1.0. This intentionally
differs from the `["period", "index_value"]` schema of the temporal
multilateral methods, so the output does **not** feed the extension (splicing)
functions — splicing is a temporal-drift tool and has no spatial meaning.

## Graduation path

If these methods stabilise, they will either migrate to the standard
multilateral conventions or the core API will generalise towards
entity-style column naming. Until then, treat the `experimental` namespace as
subject to change.

## References

- Białek, J. (2022). Improving quality of the scanner CPI: proposition of new
  multilateral methods. *Quality & Quantity*, 57, 2893–2921.
- Rao, D. S. P., & Timmer, M. P. (2003). Purchasing Power Parities for
  Industry Comparisons Using Weighted EKS Methods. *Review of Income and
  Wealth*, 49(4), 491–511.
- Diewert, W. E. (1999). Axiomatic and Economic Approaches to International
  Comparisons. *NBER Studies in Income and Wealth*, Vol. 61, 13–107.
- van IJzeren, J. (1987). *Bias in International Index Numbers: A
  Mathematical Elucidation*. Statistical Studies 34, CBS, The Hague.
