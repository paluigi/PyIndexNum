# Plan: Fix Bilateral Index Functions in PyIndexNum

## Problem

The published PyPI version (0.1.0) of pyindexnum had three bugs in bilateral index functions:

1. **Carli/Dutot naming swap**: `carli()` (actually `dudot()` with a typo) implemented the Dutot formula, and `dutot()` (spelled `carli`) implemented the Carli formula.

2. **Tornqvist used quantity shares** instead of expenditure shares:
   - Published: `exp(sum((q_0 + q_t)/(2*Q) * ln(p_t/p_0)))` — weights by quantities
   - Correct: `exp(sum(0.5*(s_0 + s_t) * ln(p_t/p_0)))` where `s_i = p_i*q_i / sum(p_i*q_i)` — weights by expenditure shares

3. **Walsh used wrong formula entirely**:
   - Published: `sum(sqrt(p_t*p_0)*q_0) / sum(p_0*q_0)` — geometric mean of prices with base quantities
   - Correct: `sum(p_t*sqrt(q_0*q_t)) / sum(p_0*sqrt(q_0*q_t))` — geometric mean of quantities as basket

The local source code already had bugs #2 and #3 corrected. Only #1 needed fixing.

## Step 1: Fix Carli/Dutot Naming Swap [DONE]

Swapped function bodies and docstrings in `bilateral.py`:
- `carli()` now computes arithmetic mean of price relatives (correct Carli formula)
- `dutot()` now computes ratio of arithmetic means of prices (correct Dutot formula)
- Fixed typo "Dudot" → "Dutot" in error message

## Step 2: Diagnose Tornqvist/Walsh Divergence [DONE]

Diagnosis confirmed: the local source code already has correct Tornqvist and Walsh formulas. The discrepancies in the comparison CSV were from the published PyPI v0.1.0. Running the current local code against `performance_price_data.csv` produces values matching R exactly:

| Index     | PyIndexNum (local) | R PriceIndices |
|-----------|--------------------|----------------|
| Tornqvist | 1.1828549          | 1.1828549      |
| Walsh     | 1.1163318          | 1.1163318      |

## Step 3: Fix Tornqvist/Walsh [NOT NEEDED]

Already correct in local source. No code changes required.

## Step 4: Update Tests [DONE]

- Copied `performance_price_data.csv` into `tests/`
- Rewrote `test_bilateral.py` with:
  - 8 regression tests against R gold-standard values (tolerance < 1e-5)
  - 14 hand-computed unit tests with step-by-step verification (tolerance < 1e-10)
  - 10 validation/error-handling tests
- All 32 tests pass. Full suite: 106/107 pass (1 pre-existing failure in multilateral tests, unrelated).

## Files Modified

| File | Change |
|------|--------|
| `src/pyindexnum/bilateral.py` | Fixed Carli/Dutot naming swap |
| `tests/test_bilateral.py` | Rewritten with R reference regression + hand-computed unit tests |
| `tests/performance_price_data.csv` | New file — copied from price-indices-example |
