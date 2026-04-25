Consumer Price Index Methodologies: Calculation Reference

This document provides a technical specification for bilateral and multilateral index number methodologies as described in the Consumer Price Index Manual: Theory (2025). It is designed to be parsed by an AI agent for the purpose of generating a Python calculation library.

1. Bilateral Indices

Bilateral indices compare price or quantity levels between two specific time periods: a base period ($0$) and a current period ($t$).

1.1 Laspeyres Price Index
Type: Fixed-basket index (Base Period Weights).Description: Measures the change in the cost of the basket of goods and services purchased in the base period.Inputs:$p^0$: Vector of prices in period $0$ ($N$ items).$p^t$: Vector of prices in period $t$ ($N$ items).$q^0$: Vector of quantities in period $0$ ($N$ items).Alternative Input: $s^0$: Vector of expenditure shares in period $0$.Calculation Mechanism:The standard formula (aggregate form): $$P_L = \frac{\sum_{n=1}^{N} p_{n}^t q_{n}^0}{\sum_{n=1}^{N} p_{n}^0 q_{n}^0}$$The weighted average of price relatives form (using shares):$$P_L = \sum_{n=1}^{N} s_{n}^0 \left( \frac{p_{n}^t}{p_{n}^0} \right)$$
Where $s_n^0 = \frac{p_n^0 q_n^0}{\sum_{i=1}^{N} p_i^0 q_i^0}$.

1.2 Paasche Price Index
Type: Fixed-basket index (Current Period Weights).Description: Measures the change in the cost of the basket of goods and services purchased in the current period.Inputs:$p^0$: Vector of prices in period $0$.$p^t$: Vector of prices in period $t$.$q^t$: Vector of quantities in period $t$.Alternative Input: $s^t$: Vector of expenditure shares in period $t$.Calculation Mechanism:The standard formula (aggregate form): $$P_P = \frac{\sum_{n=1}^{N} p_{n}^t q_{n}^t}{\sum_{n=1}^{N} p_{n}^0 q_{n}^t}$$
The harmonic mean of price relatives form (using current shares):
$$P_P = \left[ \sum_{n=1}^{N} s_{n}^t \left( \frac{p_{n}^t}{p_{n}^0} \right)^{-1} \right]^{-1}$$
Where $s_n^t = \frac{p_n^t q_n^t}{\sum_{i=1}^{N} p_i^t q_i^t}$.

1.3 Fisher Ideal Price Index
Type: Superlative Index.Description: The geometric mean of the Laspeyres and Paasche indices. It satisfies the time reversal test and factor reversal test.Inputs:$P_L$: Laspeyres Index value.$P_P$: Paasche Index value.(Derived from $p^0, p^t, q^0, q^t$).Calculation Mechanism:
$$P_F = \sqrt{P_L \times P_P}$$

1.4 Walsh Price Index
Type: Superlative Index.Description: A pure price index that uses the geometric mean of the quantities from the two periods as the fixed basket.Inputs:
$p^0, p^t$: Price vectors.
$q^0, q^t$: Quantity vectors.
Calculation Mechanism:
$$P_W = \frac{\sum_{n=1}^{N} p_{n}^t \sqrt{q_{n}^0 q_{n}^t}}{\sum_{n=1}^{N} p_{n}^0 \sqrt{q_{n}^0 q_{n}^t}}$$

1.5 Törnqvist Price IndexType: Superlative Index.Description: A weighted geometric average of the price relatives. The weights are the arithmetic average of the expenditure shares in the two periods.Inputs:$p^0, p^t$: Price vectors.$s^0, s^t$: Expenditure share vectors.Calculation Mechanism:
$$P_T = \prod_{n=1}^{N} \left( \frac{p_{n}^t}{p_{n}^0} \right)^{\frac{1}{2} (s_{n}^0 + s_{n}^t)}$$
In logarithmic form (often used for computation):
$$\ln(P_T) = \sum_{n=1}^{N} \frac{1}{2} (s_{n}^0 + s_{n}^t) \ln\left( \frac{p_{n}^t}{p_{n}^0} \right)$$

1.6 Marshall-Edgeworth Price IndexType: Fixed-basket index (Mean Basket).Description: Uses the arithmetic mean of quantities from both periods as the basket.Inputs:$p^0, p^t$: Price vectors.$q^0, q^t$: Quantity vectors.Calculation Mechanism:
$$P_{ME} = \frac{\sum_{n=1}^{N} p_{n}^t (q_{n}^0 + q_{n}^t)}{\sum_{n=1}^{N} p_{n}^0 (q_{n}^0 + q_{n}^t)}$$

1.7 Lowe Price IndexType: Fixed-basket index (Pre-period Weights).Description: Uses quantities from a weight reference period ($b$) which precedes the price reference period ($0$). This is the standard operational formula for many CPIs.Inputs:$p^0$: Prices in price reference period $0$.$p^t$: Prices in current period $t$.$q^b$: Quantities from weight reference period $b$ (where $b < 0$).Calculation Mechanism:
$$P_{Lo} = \frac{\sum_{n=1}^{N} p_{n}^t q_{n}^b}{\sum_{n=1}^{N} p_{n}^0 q_{n}^b}$$

1.8 Young Price IndexType: Weighted Arithmetic Mean.Description: A weighted arithmetic average of price ratios between period $0$ and $t$, using expenditure shares from a weight reference period $b$, without revaluing them to period $0$ prices.Inputs:$p^0$: Prices in price reference period $0$.$p^t$: Prices in current period $t$.$s^b$: Expenditure shares from weight reference period $b$.Calculation Mechanism:
$$P_Y = \sum_{n=1}^{N} s_{n}^b \left( \frac{p_{n}^t}{p_{n}^0} \right)$$

1.9 Elementary Indices
Type: Unweighted Indices.Description: Used at the lowest level of aggregation where specific quantity weights are unavailable.Inputs:$p^0$: Vector of prices in period $0$.$p^t$: Vector of prices in period $t$.$N$: Number of items.

Calculation Mechanisms:

A. Carli Index: Arithmetic mean of price ratios.
$$P_C = \frac{1}{N} \sum_{n=1}^{N} \frac{p_{n}^t}{p_{n}^0}$$

B. Dutot Index: Ratio of arithmetic means of prices.
$$P_D = \frac{\frac{1}{N} \sum_{n=1}^{N} p_{n}^t}{\frac{1}{N} \sum_{n=1}^{N} p_{n}^0} = \frac{\sum_{n=1}^{N} p_{n}^t}{\sum_{n=1}^{N} p_{n}^0}$$

C. Jevons Index: Geometric mean of price ratios (transitive).
$$P_J = \prod_{n=1}^{N} \left( \frac{p_{n}^t}{p_{n}^0} \right)^{1/N}$$

D. Harmonic Index: Harmonic mean of price ratios.
$$P_H = \left[ \frac{1}{N} \sum_{n=1}^{N} \left( \frac{p_{n}^t}{p_{n}^0} \right)^{-1} \right]^{-1}$$

E. CSWD Index: Geometric mean of Carli and Harmonic.
$$P_{CSWD} = \sqrt{P_C \times P_H}$$

2. Multilateral IndicesMultilateral indices calculate price levels across multiple time periods ($1, \dots, T$) simultaneously to ensure transitivity and avoid chain drift.

2.1 GEKS (Gini-Eltetö-Köves-Szulc) Index
Description: The geometric mean of all possible bilateral indices between a base period and period $t$, linked through every other period $k$ in the window.Inputs:Dataset of prices and quantities for periods $1 \dots T$.Bilateral Index Function (typically Fisher, $P_F$).
Calculation Mechanism:The price level for period $t$ relative to period $1$ is:
$$P_{GEKS}^t = \prod_{k=1}^{T} \left( \frac{P_F(p^k, p^t, q^k, q^t)}{P_F(p^k, p^1, q^k, q^1)} \right)^{1/T}$$

2.2 CCDI (GEKS-Törnqvist) Index
Description: The GEKS method applied using the Törnqvist index as the underlying bilateral formula.Inputs:Dataset of prices and shares for periods $1 \dots T$.Bilateral Törnqvist Function ($P_T$).Calculation Mechanism:
$$P_{CCDI}^t = \prod_{k=1}^{T} \left( \frac{P_T(p^k, p^t, s^k, s^t)}{P_T(p^k, p^1, s^k, s^1)} \right)^{1/T}$$

2.3 Geary-Khamis (GK) Index
Description: An implicit price index that solves for reference prices and period price levels simultaneously.Inputs:$p^t_n$: Price of item $n$ in period $t$.$q^t_n$: Quantity of item $n$ in period $t$.$T$: Number of time periods.$N$: Number of products.Calculation Mechanism:This requires solving a system of equations iteratively.Reference Prices ($\alpha_n$):
$$\alpha_n = \frac{\sum_{t=1}^{T} (q_{n}^t p_{n}^t / P_{GK}^t)}{\sum_{t=1}^{T} q_{n}^t}$$
Period Price Levels ($P_{GK}^t$):
$$P_{GK}^t = \frac{\sum_{n=1}^{N} p_{n}^t q_{n}^t}{\sum_{n=1}^{N} \alpha_n q_{n}^t}$$
Algorithm:Initialize $P_{GK}^t = 1$ for all $t$.Calculate vector $\alpha$.
Update vector $P_{GK}$.
Repeat until $P_{GK}$ converges.

2.4 Time Product Dummy (TPD) Index
Description: A regression-based approach estimating price levels ($\pi_t$) and item effects ($\alpha_n$) from a panel of data.Weighted TPD
Inputs:Prices ($p_n^t$) and Expenditure Shares ($s_n^t$) for all $n, t$.
Calculation Mechanism:Solve the weighted least squares problem:
$$\min_{\pi, \alpha} \sum_{t=1}^{T} \sum_{n \in S(t)} s_{n}^t [\ln p_{n}^t - \ln \pi_t - \ln \alpha_n]^2$$
This results in the following system (solved iteratively):
Price Levels:
$$\pi_t = \exp \left( \sum_{n \in S(t)} s_{n}^t (\ln p_{n}^t - \ln \alpha_n) \right)$$
(Constraint: weights $s_n^t$ must sum to 1 in each period).
Item Effects:
$$\alpha_n = \exp \left( \frac{\sum_{t \in S^*(n)} s_{n}^t (\ln p_{n}^t - \ln \pi_t)}{\sum_{t \in S^*(n)} s_{n}^t} \right)$$
Normalize $\pi_1=1$ after convergence.