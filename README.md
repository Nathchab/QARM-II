# QARM II — Academic portfolio optimization

A Streamlit project for comparing Mean-Variance, Resampled Mean-Variance,
Black-Litterman and constrained Risk Parity. The application combines an adaptive
investor questionnaire, automatic gamma, allocation constraints, portfolio
analytics, transaction-cost backtests and historical crisis comparisons.

## Run locally

Use Python 3.12. From the repository root:

```sh
python -m venv .venv
```

Activate with `.venv\Scripts\Activate.ps1` on Windows PowerShell or
`source .venv/bin/activate` on macOS/Linux, then run:

```sh
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the displayed local URL, normally http://localhost:8501. Prices are downloaded
from Yahoo Finance; internet access is required for a fresh dataset. The app
supports USD and CHF reporting and an explicit data refresh.

## GitHub Codespaces

[Open in GitHub Codespaces](https://codespaces.new/Nathchab/QARM-II), or select
your branch in [Nathchab/QARM-II](https://github.com/Nathchab/QARM-II) and choose
**Code → Codespaces → Create codespace on this branch**. The Python 3.12 dev
container installs `requirements.txt` automatically. In its terminal, run:

```sh
python -m streamlit run app.py
```

Open port **8501** from the Ports panel. Keep its visibility **Private**; each
teammate can run their own codespace. Codespaces availability, usage limits and
billing depend on the account or organisation. Configuration alone does not grant
repository write access. See GitHub's [Python Codespaces guide](https://docs.github.com/en/codespaces/setting-up-your-project-for-codespaces/adding-a-dev-container-configuration/setting-up-your-python-project-for-codespaces).
The configuration and application startup were validated locally; the first
cloud Codespaces launch remains to be tested.

## Tests and structure

```sh
python -m unittest discover -s tests -v
```

The main suite uses offline synthetic data and a frozen covariance snapshot; it
does not need live price downloads. Keep `QARM_II_Questionnaire.docx` and the
small CSV fixtures under `analysis/`: tests verify the original questionnaire
and numerical results against them.

| Path | Purpose |
| --- | --- |
| `app.py`, `app_pages/` | Streamlit navigation and pages |
| `src/` | Questionnaire, profiles, data, optimizers and analytics |
| `tests/` | Main scoring, optimizer, portfolio and offline UI regressions |
| `analysis/` | Small retained fixtures and concise provenance |
| `.devcontainer/` | Python environment and port forwarding for Codespaces |

## Collaborating

The private repository is [Nathchab/QARM-II](https://github.com/Nathchab/QARM-II),
owned by the personal account **Nathchab**. Original contributors' Git history is
retained. The separate teammate-owned repository is not a publication destination.

The owner can invite teammates under **Settings → Collaborators → Add people**.
They must accept the invitation before accessing this private repository;
collaborators on a personal private repository receive write access. Each can
then create their own codespace. Codespaces access alone does not grant write
permission. See GitHub's [collaborator instructions](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/repository-access-and-collaboration/inviting-collaborators-to-a-personal-repository).

Start from `main`, create a branch such as `feature/your-change`, run the tests,
then commit, push that branch and open a pull request. Review changes before
merging into `main`. Keep local
environments, credentials, Yahoo cookies, caches and generated study outputs out
of Git. Never commit `.env` or `.streamlit/secrets.toml`.

## Academic limits

Questionnaire scores and the provisional RT-to-gamma mapping are research
assumptions, not validated investor preferences or monetary loss capacity.
`gamma_final = max(gamma_willingness, gamma_capacity)` is preserved; Risk Parity
does not use gamma. All methods retain mandatory allocation and feasibility
checks. Volatility is a reported metric, not a pass/fail limit relative to MV.
Financial/liquidity review flags remain visible independently of method validity.

Historical estimates, backtests and illustrative fallbacks are not guarantees or
investment advice. Live data availability, unfinished current-month observations
and solver termination can affect a demonstration. See the [policy notes](FINANCIAL_POLICY_IMPLEMENTATION.md),
[gamma assumptions](analysis/gamma_calibration/REPORT.md) and
[Risk Parity validation](analysis/risk_parity_validation/REPORT.md).
