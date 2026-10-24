# Robo-advisor – QARM II project (HEC Lausanne)

A Streamlit web app that works like a robo-advisor:

1. **Your profile**: a short questionnaire (from `QARM_II_Questionnaire.docx`) measures
   *risk tolerance* (how much risk the client wants) and *risk capacity* (how much risk he can afford).
2. **Your products & constraints**: recommended ETFs and crypto for the asset classes the client chose,
   with allocation limits that he can tighten but not relax.
3. **Your portfolio**: Black-Litterman (recommended; market equilibrium, plus views for experienced investors),
   Mean-Variance, Resampled Mean-Variance (Michaud) and Risk Parity (ERC). Weights, risk contributions, efficient frontier, beta,
   tracking error and a γ composition map.
4. **Backtest**: out-of-sample comparison of the four methods, with transaction costs, a turnover limit and
   optional cost-aware optimisation (LN3).
5. **Time machine**: buy-and-hold replays of 2008, 2020 and 2022, compared with the loss the client said he
   could tolerate.

Everything can be shown in USD or CHF. Prices are downloaded from Yahoo Finance and cached for one day.

## How the profile drives the portfolio

Tolerance and capacity are kept separate, and each has **one** job:

| Dimension | Questions | Used for |
| --- | --- | --- |
| Risk tolerance (RT, 0–100) | QE1–QE4 | risk aversion γ of the optimisers |
| Financial capacity (0–9) | age, income, emergency savings, commitments | allocation limits |
| Horizon and liquidity | QD1, QD2 | allocation limits |
| Asset universe | QF1, QF2 (QF2 only if crypto is chosen) | recommended products, crypto limit |
| Experience | QB1, QC1, QC3 | experienced investor or not |
| Market views | QH4 (experienced only), then QH5–QH6 on the portfolio page | Black-Litterman P, Q and Ω |

**Tolerance → γ.** `γ(RT) = 12 × (1/12)^(RT/100)`, so γ goes from 12 (very prudent) to 1 (very tolerant).
The anchors are calibrated on QE3: with the optimiser's utility `U = μ − γ/2·σ²`, the client is
indifferent between two options when `γ = 2(μ₂ − μ₁)/(σ₂² − σ₁²)`. The QE3 profiles give the thresholds
8, 2.67 and 1.6, and γ evaluated at the RT score of each QE3 answer falls inside the interval that answer
reveals (tested in `tests/test_profile.py`).

**Capacity → limits.** The capacity score gives a tier (Low, Limited, Moderate, High) with a maximum in
equities, a minimum in bonds and maximums in gold/commodities and crypto. QD1 and QD2 give their own equity
maximum and bond minimum; the most restrictive source applies. Low savings or heavy commitments cap the tier
at Limited.

**Why capacity does not change γ.** A γ floor from capacity *and* capacity-based limits would count the
same information twice, and for most clients the tolerance answers would then change nothing. Here a
tolerant client with low capacity gets the most dynamic portfolio inside the limits his situation allows,
with an explanation rather than an alarm. Warnings are kept for material facts (savings below 3 months,
significant commitments, money needed within a year, likely withdrawals above 25%). Every profile also gets
one sentence with the equity/bond mix and volatility of its recommended portfolio.

**Black-Litterman by default.** Without views, μ_BL = π, the market equilibrium returns: more stable than the
60-month historical mean, so it is the recommended method for everybody. Only experienced investors (at least
2 years of investing, intermediate knowledge, and the correct answer to the diversification question QC3) who
say they have convictions (QH4) can enter views: up to 5, absolute (“GLD will return 6%”) or relative (“EEM will
beat SPY by 3%”), each with a confidence from QH6. The uncertainty is `Ω_kk = τ (PΣPᵀ)_kk (1 − c)/c`, and a view
more than two standard deviations away from the equilibrium triggers a warning.

## Run the app

Use Python 3.12, then run from the project folder:

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # Windows PowerShell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

On macOS/Linux, activate with `source .venv/bin/activate` instead. In GitHub Codespaces the dev container
installs everything; run `python -m streamlit run app.py`.

Tests: `python -m unittest discover -s tests -v`

## Code structure

```
app.py                  Streamlit menu and sidebar
app_pages/              one file per page
src/questionnaire.py    questions, answer scores, RT and capacity scores
src/client_profile.py   gamma from tolerance, allocation limits from capacity, products, messages
src/data.py             products, prices, returns and risk-free rates
src/optimizer.py        feasibility check, mean-variance, LN3 costs, resampling, risk parity, frontier
src/black_litterman.py  equilibrium returns, absolute/relative views, posterior, view checks
src/portfolios.py       the four methods and the benchmarks
src/backtest.py         out-of-sample backtest and performance measures
src/time_machine.py     crisis replay
tests/                  profile, Black-Litterman, optimizer and risk parity tests, page smoke test
analysis/               risk parity solver validation
```

## Data

19 ETFs covering equities, bonds, gold and commodities, plus Bitcoin and Ethereum. Yahoo Finance supplies
adjusted prices, the USD/CHF rate (`CHF=X`) and the 13-week US T-bill yield (`^IRX`). The CHF risk-free rate
uses yearly approximations of the SNB policy rate.

## Limitations

The questionnaire scores, capacity tiers and γ anchors are modelling choices calibrated to be internally
consistent, not estimated on investor data. This is an academic project, not investment advice.
