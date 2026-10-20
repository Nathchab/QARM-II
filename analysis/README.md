# Analysis

`risk_parity_validation/` documents a check of the constrained Risk Parity (ERC)
solver on a dated USD covariance snapshot (60 monthly returns, October 2026). The
CSV is used by `tests/test_risk_parity_validation.py`; the app itself downloads
current prices from Yahoo Finance and does not read this folder.
