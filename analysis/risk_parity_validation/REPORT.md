# Constrained Risk Parity validation

19 October 2026. The observed USD allocation reproduces at whole-percent
precision. Binding allocation limits explain unequal risk contributions; no
material initialization bug or better feasible solution was found. The optimizer
and all financial calculations remain unchanged.

## Data and comparison

The retained [annual covariance](covariance_fresh_USD.csv) was estimated from the
app's last 60 monthly USD returns (November 2021–October 2026), using adjusted Yahoo
prices downloaded through 19 October 2026 and `12 * monthly_covariance`. The
unfinished current month is included exactly as in the app; its month-end label
does not imply future price observations. The prior live session cache was not
available, so bit-for-bit identity with that session cannot be certified. The
frozen covariance and original questionnaire are retained for offline regression.

Tested limits: equity ≤20%, bonds ≥60%, Other ≤20%, per-product ≤40%, long-only
and fully invested (a very prudent allocation box). Risk Parity does not use gamma,
so only these limits and the covariance matter.

The less-constrained comparison uses equity ≤80%, bonds ≥0%, with the same
Other/product caps.

| Asset | Current weight | Risk share | Less-constrained weight | Risk share |
| --- | ---: | ---: | ---: | ---: |
| SPY | 4.63% | 6.10% | 11.42% | 14.51% |
| EFA | 5.05% | 7.80% | 10.13% | 14.56% |
| EEM | 5.33% | 8.64% | 9.70% | 14.62% |
| VNQ | 4.99% | 8.97% | 8.72% | 14.29% |
| AGG | 40.00% | 27.30% | 28.03% | 15.82% |
| TLT | 20.00% | 28.97% | 12.00% | 14.04% |
| GLD | 5.53% | 5.85% | 7.99% | 7.47% |
| DBC | 14.47% | 6.36% | 12.01% | 4.69% |

| Diagnostic | Current | Less constrained |
| --- | ---: | ---: |
| Annual estimated volatility | 8.6280% | 9.8528% |
| Equity / bonds / Other | 20% / 60% / 20% | 39.97% / 40.03% / 20% |
| Scaled ERC objective | 66.27947 | 11.56300 |
| SLSQP success / status | True / 0 | True / 0 |
| Iterations on saved CSV | 13 | 18 |
| Allocation violations | None | None |

## Formula and interpretation

With `v = w' Sigma w`, normalized risk share is `w_i * (Sigma w)_i / v`.
Shares sum to one and are invariant to covariance annualization. Absolute Euler
contributions divide by `sqrt(v)` and sum to portfolio volatility. The existing
ERC objective is `1000 * sum((share_i - 1/n)**2)`; 1000 changes numerical scaling,
not the minimizer.

Current binding limits are full investment, equity 20%, bonds 60%, Other 20%
and AGG 40%. Some are redundant: equity plus Other caps imply at least 60% bonds.
TLT must supply at least 20% once AGG reaches its product cap. The less-constrained
solution still binds Other at 20%. Removing limits for diagnosis only gives
12.5% shares but requires about 30.69% Other, breaching the cap.

The original audit tried 68 start/derivative comparisons per profile/covariance:
equal weights, production weights and 32 feasible LP-vertex mixtures, each with
finite differences and independently checked analytical derivatives. All 68 live
USD outputs were feasible; 54 constrained and 68 less-constrained runs reported
success at tighter research tolerance. Failed terminations were not counted as
successful solutions. Best successful constrained objective improvement was only
0.0000264, with maximum weight change below 0.004 percentage points. This supports
the result for this matrix, without claiming a general global-optimality proof.

`risk_parity` now raises an error when SLSQP does not report convergence, so a
failed solve can no longer be shown as a portfolio. See the
[SLSQP documentation](https://docs.scipy.org/doc/scipy-1.15.1/reference/optimize.minimize-slsqp.html).

## Regressions and scope

Run `python -m unittest discover -s tests -v`. The five focused ERC tests check
Euler normalization/scaling, constrained feasibility and binding bounds, alternate
initialization, the less-constrained case and exact-ERC infeasibility under the
Other cap.

The full prior audit scripts, price downloads, manifests and output tables are in
the verified local pre-release backup, not required for the main application or
these tests. Live data/network access and unfinished-month estimation should be
checked on the demonstration machine.
