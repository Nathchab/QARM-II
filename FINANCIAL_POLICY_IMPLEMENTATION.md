# Academic financial capacity, allocation and presentation

19 October 2026. Final academic simplification implemented locally. The
capacity/allocation version remains `QARM-capacity-allocation-v2-research`.
The project demonstrates portfolio methods, financial reasoning and Python;
it is not a production investment advisory or regulatory suitability system.

The Mean-Variance-relative volatility compliance screen is removed. No replacement
volatility threshold or risk-budget policy is introduced. The earlier screen
studies are archived outside the collaborative release, not active requirements.

## Preserved methodology

Original RT, EX, KN, HZ, CX and conditional CV definitions, scoring and applicability
are unchanged. The exponential willingness-gamma mapping in `src/gamma_policy.py`
is unchanged, including its provisional anchors and configurable assumptions.
Investors do not enter gamma:

```text
gamma_final = max(gamma_willingness, gamma_capacity)
```

The corrected Mean-Variance solver, Resampled MV, Black-Litterman, Risk Parity,
portfolio builder and cost-aware backtest algorithms are unchanged. No objectives,
optimizer signatures or Black-Litterman prior parameters were changed. Risk Parity
does not consume gamma; its allocation and outcome checks operate independently
of gamma. Validation never blends or normalizes method weights to force a valid result.

Pre-implementation file hashes confirm byte-for-byte preservation of
`optimizer.py`, `gamma_policy.py`, `questionnaire_definitions.py`,
`questionnaire_scoring.py`, `questionnaire_routing.py`, `backtest.py`,
`black_litterman.py` and `portfolios.py` in `src/`. The optimizer already contained
the corrected solver when this implementation began.

## Four-question financial capacity

The new investor-facing capacity section has exactly four questions:

| Input | Choices and points |
| --- | --- |
| Original age question | Under 30: 3; 30–45: 2; 46–60: 1; over 60: 0 |
| Original income-stability question | No regular income: 0; fairly stable: 1; very stable: 2 |
| Emergency savings outside this investment account | Less than 3 months of essential expenses: 0; 3–6 months: 1; more than 6 months: 2 |
| Financial commitments | None significant: 2; manageable: 1; significant obligations or financial difficulties: 0 |

Total points 0–3 give floor 12, 4–6 give floor 6, and 7–9 give floor 2.
Either zero-point emergency savings or zero-point commitments imposes a minimum
floor of 6. Subscores, base floor, safeguards and final floor are retained.
Incomplete answers cannot become zero-point assumptions.

Raw QD1/QD2 are reused without asking horizon or liquidity questions again. The
legacy acceptable-loss question is removed from the new form; QE2 retains its
original willingness interpretation and does not determine financial capacity.
The crisis page shows that original comfort response without converting it into
a monetary capacity limit or fabricating an exact threshold from an open band.

## Mandatory allocation rules

| Source | Category | Maximum equity | Minimum bonds |
| --- | --- | --- | --- |
| Capacity floor | 12 | 20% | 60% |
| Capacity floor | 6 | 40% | 30% |
| Capacity floor | 2 | 80% | 0% |
| QD1 | Within 1 year | 20% | 60% |
| QD1 | 1–3 years | 30% | 50% |
| QD1 | 3–5 years | 60% | 30% |
| QD1 | 5–10 years | 60% | 0% |
| QD1 | More than 10 years | 80% | 0% |
| QD2 | More than 50% | 20% | 60% |
| QD2 | 25–50% | 40% | 50% |
| QD2 | 10–25% | 60% | 30% |
| QD2 | Less than 10% | 80% | 0% |
| QD2 | None | 100% | 0% |

Equity uses the minimum of the three source ceilings; bonds use the maximum of
the three floors. Existing per-product 40%, Other 20%, crypto 0%/10%, long-only
and full-investment restrictions remain. Optional upper bounds combine with
mandatory bounds through `min`; optional bond minima combine through `max`.
Optional preferences cannot waive mandatory restrictions or financial review.

These rules explicitly replace acceptable-loss-dependent limits for new policy
profiles. They need not dominate every legacy allocation restriction. The
archived research reports retain the representative before/after comparisons;
the migration preview shows the particular investor's old and new limits.

## Three presentation states

| State | Meaning and display |
| --- | --- |
| **Normal** | Reliable finite, long-only, fully invested weights satisfy effective and mandatory allocation limits. Display the method normally. |
| **Information** | Capacity sets final gamma or allocation bounds restrict exposure. Explain the adjustment neutrally; it is not itself a warning. |
| **Warning** | Material unresolved financial/liquidity facts, infeasibility, missing critical data or invalid calculations. Show one consolidated profile notice and retain all diagnostic reasons. |

Method validity and investor-level financial information are separate: common
financial/liquidity flags do not label every method Warning. They remain visible
once per profile/page. Invalid data/calculations and disclosed infeasible-request
fallbacks still set the affected method's Warning state.

All four methods remain visible whenever reliable allocation-compliant weights
exist. Each keeps expected return, annualized volatility, equity/bond exposure,
concentration, binding limits and existing risk analytics/visualizations. Volatility
differences between methods no longer create a non-compliant status. There is no
MV reference, allowance, volatility ceiling or scenario-compliance table in the UI.
Risk Parity still does not use gamma. The gamma composition chart remains an MV
sensitivity visualization, not a risk-policy ceiling.
The profile heading reads **Risk Tolerance: Aggressive** (or the original RT
classification), separately from financial capacity. Full gamma precision,
binding constraint identifiers and ERC objective/risk-share details are in
**Advanced details**; the compact method table describes calculation validity.
See [Risk Parity validation](analysis/risk_parity_validation/REPORT.md) for the
latest USD reproduction, constraint analysis and initialization checks.

Low reserves, significant commitments/difficulties, near-term QD1 needs and
positive QD2 withdrawals remain in `financial_review_flags`. They are consolidated
once per profile/page rather than repeated as warning banners for every method.
Capacity safeguards are explained as information; their underlying material
financial flags still warn. Changed drafts use the last saved profile and remain
explicitly identified. General academic assumptions appear as captions.

Feasibility checks and fallback assumptions are unchanged. A fallback may omit
explicitly disclosed tighter optional settings or add available default products;
it never weakens mandatory limits or rewrites saved preferences. It is clearly
labelled illustrative. Invalid/non-unit/short weights are not normalized. A
method violating allocation bounds is unavailable as a valid result, with its
violations retained; another valid method may be shown as a named illustrative
alternative. If no reliable alternative exists, show a diagnostic, not weights.

The backtest's trades, costs, turnover, returns and charts are unchanged. Its
sidecar checks actual rebalance holdings against allocation limits and reports
estimated volatility from preceding observations; it no longer calculates a
second MV reference or any volatility compliance test. Historical invalid
calculations/allocations join the consolidated notice. The time machine retains
all crisis charts and QE2-versus-legacy comfort semantics.

Historical scenario helpers in `src/portfolio_policy.py` and the dormant allowance
helper in `src/financial_capacity.py` remain only for reproducing archived studies.
The active assessment never calls them. Legacy scenario/reference keyword
arguments are accepted but cannot change application status. The application no
longer loads the frozen synthetic covariance files or warns about their product
coverage. Actual estimated covariance still must be finite, symmetric and positive
semidefinite; missing risk-free data remains a calculation warning.

## Explicit legacy migration

Legacy constructors and explicit-baseline calls retain their original numerical
behavior. `ClientProfile.from_questionnaire` still accepts the original capacity
schema and also dispatches the four-field schema to the new automatic factory.
The new factory is `from_financial_questionnaire`; no fabricated legacy horizon
or acceptable-loss answers are needed. Existing gamma and constraint interfaces
remain available downstream.

Recomputing a saved legacy profile first creates a preview. The active old
profile is retained until the investor clicks **Apply reviewed policy migration**.
Changes to the draft invalidate that preview. Applying migration archives old
answers, scores, gamma, limits, optional requests and product selections; it
retains the previous profile and adds a migration-history snapshot. The preview
discloses that new default product selections and optional settings will apply.
Page navigation alone cannot migrate a profile. A saved legacy profile remains
explicitly identified as legacy without changing its saved numerical behavior.

## Validation and demonstration

Run `python -m unittest discover -s tests -v`. The main suite covers original
questionnaire/document parity, all 500 RT patterns, 108 financial combinations,
QD1/QD2 pairs, automatic gamma, legacy compatibility/migration, constraints, the
four methods, corrected MV regressions and offline Streamlit navigation,
portfolios, backtests and crisis displays. Five focused ERC checks use the
retained dated USD covariance. Research artifacts are not needed to run the suite.

No financial or numerical behavior is changed by repository cleanup. Live Yahoo
downloads still require network access; offline UI tests do not certify fresh
market availability. Research policy assumptions are not validated monetary loss
capacity or investment advice.
