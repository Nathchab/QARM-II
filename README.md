# Robo-advisor – QARM II project (HEC Lausanne)

A Streamlit web app that works like a robo-advisor:

1. **Your profile**: a short questionnaire (from `QARM_II_Questionnaire.docx`) measures
   *risk tolerance* (how much risk the client wants) and *risk capacity* (how much risk he can afford).
2. **Your products & constraints**: recommended ETFs and crypto for the asset classes the client chose,
   with allocation limits that he can tighten but not relax.
3. **Your portfolio**: Black-Litterman (recommended; market equilibrium, plus views for experienced investors),
   Mean-Variance, Resampled Mean-Variance (Michaud) and Risk Parity (ERC). Weights, risk contributions, efficient frontier, beta,
   tracking error and a γ composition map.
4. **Your plan**: projection of the client's wealth with a monthly savings plan (5,000 simulated futures), probability
   of reaching a savings goal and of a fall larger than the loss he accepts, the orders to place (number of shares
   of each ETF), our 0.50% management fee, and a downloadable client report.
5. **Backtest**: out-of-sample comparison of the four methods, with transaction costs, a turnover limit and
   optional cost-aware optimisation (LN3), after our management fee.
6. **Time machine**: buy-and-hold replays of 2008, 2020 and 2022, compared with the loss the client said he
   could tolerate.

Everything can be shown in USD or CHF. Prices are downloaded from Yahoo Finance and kept in memory for one day.
The button **Start over (new client)** in the sidebar forgets every answer and choice, so the next client
starts from zero.

**Live app:** add the address here after deployment (see "Put the app online" below).

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

**Tolerance → γ.** `γ(RT) = 12 × (1/12)^(RT/100)`, so γ goes from 12 (RT = 0) to 1 (RT = 100). The answers
give an RT between 5 and 97.5, so in practice γ lies between 10.6 (very prudent) and 1.06 (very tolerant).
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
more than two standard deviations away from the equilibrium triggers a warning. The code uses the formula of
lecture LN4: `μ_BL = π + ΓPᵀ(PΓPᵀ + Ω)⁻¹(Q − Pπ)` with `Γ = τΣ`.

## Run the app on your computer

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

## Put the app online (permanent website, as in lecture LN4)

1. The code must be on GitHub with `requirements.txt` at the root (already the case).
2. Go to https://share.streamlit.io, sign in with GitHub and click **Create app** → *Yup, I have an app*.
3. Repository `Nathchab/QARM-II`, branch `main`, main file path `app.py`. Choose the address (App URL,
   ending in `.streamlit.app`).
4. **Advanced settings → Python version 3.12** (the versions in `requirements.txt` were tested with it), then
   **Deploy**. The first start takes a few minutes.
5. In *My apps* → ⋮ → **Settings** → "Who can view this app" → "This app is public and searchable".

Every push to `main` updates the website by itself. Good to know:

- Each visitor has his own session: two people can use the site at the same time without seeing each other's
  answers. **Start over (new client)** empties the session of the current visitor.
- A free Streamlit app goes to sleep after 12 hours without visitors. The address stays the same: the next
  visitor clicks "Yes, get this app back up!" and waits about a minute. Open the site before a presentation.
- The first visitor of the day waits about 10 seconds while the prices are downloaded.

## Code structure

```
app.py                  Streamlit menu and sidebar (currency, update data, start over)
app_pages/              one file per page
src/questionnaire.py    questions, answer scores, RT and capacity scores
src/client_profile.py   gamma from tolerance, allocation limits from capacity, products, messages
src/data.py             products, prices, returns and risk-free rates
src/optimizer.py        client limits, mean-variance, LN3 costs, resampling, frontier, risk parity
src/black_litterman.py  equilibrium returns, absolute/relative views, posterior (LN4), view checks
src/portfolios.py       the four methods and the benchmarks
src/backtest.py         out-of-sample backtest and performance measures
src/time_machine.py     crisis replay
src/planning.py         projection, savings goal, order list, management fee, client report
src/app_helpers.py      small functions shared by the pages
tests/                  profile, Black-Litterman and optimizer tests, and a test that opens every page
```

## Data

19 ETFs covering equities, bonds, gold and commodities, plus Bitcoin and Ethereum. Yahoo Finance supplies
adjusted prices, the USD/CHF rate (`CHF=X`) and the 13-week US T-bill yield (`^IRX`). The CHF risk-free rate
uses yearly approximations of the SNB policy rate. The market weights used by Black-Litterman are our own
round numbers (stocks and bonds dominate, as in Doeswijk, Lam & Swinkels 2014), not measured market values.

## Limitations

The questionnaire scores, capacity tiers and γ anchors are modelling choices calibrated to be internally
consistent, not estimated on investor data. This is an academic project, not investment advice.
