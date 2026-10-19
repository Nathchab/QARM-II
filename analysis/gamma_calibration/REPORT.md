# RT-to-gamma assumptions and retained calibration fixtures

This is a concise provenance note for the academic implementation. The original
18 October 2026 calibration was a synthetic research experiment. Its initial
solver-accuracy investigation preceded the corrected Mean-Variance solver; old
solver caveats and the statement that gamma was not integrated are historical.
They do not describe the current application.

The implemented mapping in [gamma_policy.py](../../src/gamma_policy.py) is:

```text
x = (RT - 5) / (97.5 - 5)
gamma_willingness = 5.56716 * (1.16315 / 5.56716)**x
gamma_final = max(gamma_willingness, gamma_capacity)
```

The original RT mean has 500 answer patterns, 61 distinct scores and an attainable
range of 5–97.5. Scores are unchanged; x is only an internal mapping coordinate.
Endpoints remain configurable and out-of-range inputs are rejected. HZ and raw
QD1/QD2 do not modify willingness gamma. Risk Parity does not consume gamma.

The endpoint illustration selected 10% and 90% of a synthetic MV volatility span
(approximately 6.36%–18.19%) and inverted a research reference curve. These target
fractions are subjective conventions, not observations of investor preferences,
empirical CRRA estimates, guaranteed losses or a suitability calibration. Current
capacity floors may override all willingness variation within a tier.

The public MV objective is `max(mu'w - (gamma/2) * w'Sigma w)`. A reciprocal
parameter inside the quadratic solver must not be mistaken for the public gamma.
Return assumptions, covariance, asset bounds and method affect portfolio outcomes.

The retained [inputs_base.csv](inputs_base.csv) and [covariance_base.csv](covariance_base.csv)
describe eight synthetic assets (SPY, EFA, EEM, VNQ, AGG, TLT, GLD, DBC). They are
offline regression fixtures, not downloaded prices or current forecasts. Four
additional `covariance_*.csv` matrices preserve the existing historical scenario
loader interface: half/one-and-a-half return spreads, stressed covariance and weak
equity returns. They do not drive active volatility compliance: that screen was
removed. Identical covariances in return-only scenarios are intentional.

The full original experiment, result tables and figures are retained in the local
pre-release backup, outside the published tree. The main regressions run with
`python -m unittest discover -s tests -v`. See the [current policy](../../FINANCIAL_POLICY_IMPLEMENTATION.md)
for approved financial capacity and allocation rules; no mapping is changed by cleanup.
