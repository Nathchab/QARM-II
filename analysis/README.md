# Retained regression fixtures

This directory deliberately contains only seven small CSV fixtures and concise
gamma/Risk Parity provenance notes. Main tests read the base expected returns,
base covariance and frozen USD covariance; four additional synthetic covariance
matrices preserve the existing historical scenario-loader interface.

These are synthetic assumptions or a dated public-market covariance snapshot,
not investor records. Streamlit downloads current prices independently. Old
research runs, figures, cache/cookie databases and intermediate reports are
archived locally outside the release. Do not add generated outputs to the
collaborative repository without a clear runtime or regression need.
