"""Descriptive outcomes and binding allocation limits; no suitability screen."""
from math import sqrt

from src.data import ASSETS


def portfolio_diagnostics(weights, mu, cov, constraints, tolerance=1e-6):
    """Annual decimal return/volatility, exposures and capital concentration.

    mu/cov must already be annualized. HHI=sum(w**2); 1/HHI is the effective
    number of holdings. Binding limits are numerical observations, not proof of
    suitability or a claim that a particular limit reduces volatility.
    """
    mu = mu.loc[weights.index]
    cov = cov.loc[weights.index, weights.index]
    exposure = {kind: float(sum(w for ticker, w in weights.items() if ASSETS[ticker][1] == kind))
                for kind in ("Equity", "Bond", "Other", "Crypto")}
    binding = []
    violations = []
    for key, kind in (("max_equity", "Equity"), ("min_bonds", "Bond"),
                      ("max_other", "Other"), ("max_crypto", "Crypto")):
        slack = (exposure[kind] - constraints[key] if key == "min_bonds"
                 else constraints[key] - exposure[kind])
        if abs(slack) <= tolerance:
            binding.append(key)
        elif slack < -tolerance:
            violations.append(key)
    for ticker, weight in weights.items():
        if abs(weight - constraints["max_weight_per_asset"]) <= tolerance:
            binding.append(f"max_weight_per_asset ({ticker})")
        elif weight > constraints["max_weight_per_asset"] + tolerance:
            violations.append(f"max_weight_per_asset ({ticker})")
        if weight < -tolerance:
            violations.append(f"long_only ({ticker})")
    if abs(weights.sum() - 1) > tolerance:
        violations.append("fully_invested")
    hhi = float(weights @ weights)
    return {
        "Expected return": float(weights @ mu),
        "Volatility": sqrt(max(0., float(weights @ cov @ weights))),
        "Equity exposure": exposure["Equity"],
        "Bond exposure": exposure["Bond"],
        "Largest holding": float(weights.max()),
        "HHI": hhi,
        "Effective holdings": 1 / hhi if hhi else 0.,
        "Binding allocation limits": binding,
        "Allocation limit violations": violations,
    }
