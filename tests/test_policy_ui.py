"""Offline acceptance flows for visible policy statuses and explicit migration."""
from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from src.client_profile import ClientProfile
from src.data import ASSETS
from src.financial_capacity import FINANCIAL_FIELDS, POLICY_VERSION
from src.portfolios import build_portfolios, estimate
from test_financial_policy import FINANCIAL_HIGH, FINANCIAL_LOW
from test_questionnaire import CAPACITY, completed_answers

PAGE = Path(__file__).resolve().parents[1] / "app.py"


class PolicyUITests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1] / "analysis/gamma_calibration"
        mu = pd.read_csv(root / "inputs_base.csv", index_col=0).expected_return
        cov = pd.read_csv(root / "covariance_base.csv", index_col=0)
        rng = np.random.default_rng(52)
        dates = pd.date_range("2020-01-31", periods=80, freq="ME")
        returns = pd.DataFrame(rng.normal(.004, .025, (80, len(ASSETS))), index=dates, columns=list(ASSETS))
        sample = pd.DataFrame(rng.multivariate_normal(mu / 12, cov / 12, 80), index=dates, columns=mu.index)
        sample.iloc[-60:] = sample.iloc[-60:] - sample.iloc[-60:].mean() + mu / 12
        returns[mu.index] = sample
        self.returns = returns
        price_dates = pd.date_range("2005-01-31", periods=260, freq="ME")
        self.prices = 100 * (1 + pd.DataFrame(rng.normal(.003, .03, (260, len(ASSETS))),
                                              index=price_dates, columns=list(ASSETS))).cumprod()
        rf = pd.Series(.001, index=dates)
        for name, kwargs in [
            ("src.data.download_prices", dict(return_value=self.prices)),
            ("src.app_helpers.get_data", dict(return_value=(returns, rf, self.prices))),
            ("src.app_helpers.get_portfolios", dict(side_effect=self.portfolios)),
            ("src.backtest.N_SIM", dict(new=2)),
        ]:
            patcher = patch(name, **kwargs)
            patcher.start()
            self.addCleanup(patcher.stop)

    def portfolios(self, currency, tickers, gamma, c, views):
        mu, cov = estimate(self.returns[tickers])
        return build_portfolios(mu, cov, .012, gamma, c, views, n_sim=2)

    def page(self, profile=None):
        app = AppTest.from_file(str(PAGE), default_timeout=30)
        if profile is not None:
            app.session_state["profile"] = profile
        app.run()
        self.assertEqual(len(app.exception), 0)
        return app

    def profile(self, capacity=None, **overrides):
        q = dict(QD1="e", QD2="e")
        q.update(overrides)
        return ClientProfile.from_financial_questionnaire(completed_answers(**q), capacity or FINANCIAL_HIGH)

    def section(self, app, section):
        app.selectbox(key="questionnaire_section_selector").select(section).run()
        self.assertEqual(len(app.exception), 0)

    def test_exactly_four_capacity_questions_no_duplicate_horizon_liquidity_or_loss(self):
        app = self.page(self.profile())
        self.section(app, "Capacity")
        keys = {item.key for item in app.selectbox if item.key.startswith("questionnaire_capacity_")}
        self.assertEqual(keys, {"questionnaire_capacity_" + key for key in FINANCIAL_FIELDS})
        self.assertFalse(any(item.key == "questionnaire_gamma_input" for item in app.number_input))
        app.selectbox(key="questionnaire_capacity_emergency_savings").select("Less than 3 months").run()
        self.section(app, "E")
        self.assertTrue(any(item.key == "questionnaire_widget_QE2" for item in app.radio))
        self.section(app, "Capacity")
        self.assertEqual(app.selectbox(key="questionnaire_capacity_emergency_savings").value, "Less than 3 months")
        self.assertEqual(app.session_state["profile"].capacity_answers["emergency_savings"], "More than 6 months")

    def test_saved_legacy_profile_requires_preview_then_explicit_migration(self):
        old = ClientProfile.from_questionnaire(completed_answers(), CAPACITY, 3)
        app = self.page(old)
        self.section(app, "Capacity")
        app.selectbox(key="questionnaire_capacity_emergency_savings").select("More than 6 months").run()
        app.selectbox(key="questionnaire_capacity_commitments").select("None significant").run()
        self.section(app, "Results")
        next(b for b in app.button if b.label == "Compute my profile").click().run()
        self.assertEqual(app.session_state["profile"].gamma, 3)
        self.assertNotEqual(getattr(app.session_state["profile"], "capacity_policy_version", None), POLICY_VERSION)
        self.assertTrue(any("migration" in str(item.value).lower() for item in app.subheader))
        next(b for b in app.button if b.label == "Apply reviewed policy migration").click().run()
        self.assertEqual(len(app.exception), 0)
        p = app.session_state["profile"]
        self.assertEqual(p.capacity_policy_version, POLICY_VERSION)
        self.assertEqual(p.migration_history[0]["gamma"], 3)
        self.assertEqual(p.migration_history[0]["recommended_constraints"], old.recommended_constraints)

    def test_changed_draft_disables_old_migration_preview(self):
        old = ClientProfile.from_questionnaire(completed_answers(), CAPACITY, 3)
        app = self.page(old)
        self.section(app, "Capacity")
        app.selectbox(key="questionnaire_capacity_emergency_savings").select("More than 6 months").run()
        app.selectbox(key="questionnaire_capacity_commitments").select("None significant").run()
        self.section(app, "Results")
        next(b for b in app.button if b.label == "Compute my profile").click().run()
        self.section(app, "Capacity")
        app.selectbox(key="questionnaire_capacity_emergency_savings").select("Less than 3 months").run()
        self.section(app, "Results")
        self.assertTrue(next(b for b in app.button if b.label == "Apply reviewed policy migration").disabled)
        self.assertEqual(app.session_state["profile"].gamma, 3)

    def test_valid_result_has_clear_label_and_metrics_without_compliance_screen(self):
        app = self.page(self.profile())
        app.switch_page("app_pages/portfolio.py").run()
        self.assertEqual(len(app.exception), 0)
        self.assertFalse(app.warning)
        status = next(item.value for item in app.dataframe if "Method status" in item.value.columns)
        self.assertEqual(len(status), 4)
        self.assertTrue(set(status["Method status"]).issubset({"Normal", "Information"}))
        self.assertTrue({"Expected return", "Volatility", "Equity exposure", "Bond exposure", "HHI"}
                        .issubset({metric.label for metric in app.metric}))

    def test_higher_volatility_stays_visible_without_warning_or_changed_weights(self):
        original = self.portfolios
        def with_risky_result(currency, tickers, gamma, c, views):
            results = original(currency, tickers, gamma, c, views)
            risky = pd.Series(0., index=tickers)
            risky["SPY"], risky["EEM"], risky["AGG"] = .4, .4, .2
            results["Resampled MV"] = risky
            return results
        with patch("src.app_helpers.get_portfolios", side_effect=with_risky_result):
            app = self.page(self.profile())
            app.session_state["method"] = "Resampled MV"
            app.switch_page("app_pages/portfolio.py").run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(len(app.warning), 0)
            self.assertFalse(any("ceiling" in metric.label.lower() for metric in app.metric))
            self.assertTrue(any(metric.label == "Expected return" for metric in app.metric))
            comparison = next(item.value for item in app.dataframe if "Resampled MV" in item.value.columns)
            self.assertAlmostEqual(comparison.loc["SPY (US large caps (S&P 500))", "Resampled MV"], .4)

    def test_capacity_and_liquidity_warning_does_not_hide_results(self):
        p = self.profile(FINANCIAL_LOW, QD1="a", QD2="a", QE1="e", QE2="e", QE3="d", QE4="e")
        app = self.page(p)
        app.switch_page("app_pages/portfolio.py").run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.warning), 1)
        self.assertFalse(any("willingness" in str(item.value) for item in app.warning))
        self.assertTrue(any("financial difficulties" in str(item.value) for item in app.warning))
        self.assertTrue(any("liquidity" in str(item.value) for item in app.warning))
        self.assertTrue(any(metric.label == "Volatility" for metric in app.metric))
        statuses = next(item.value for item in app.dataframe if "Method status" in item.value.columns)
        self.assertFalse(statuses["Method status"].eq("Warning").any())
        self.assertEqual(set(p.financial_review_flags), {"low_external_reserve", "significant_commitments_or_difficulties",
                                                       "unquantified_near_term_need", "unquantified_unexpected_withdrawal"})

    def test_capacity_binding_alone_has_information_without_warning(self):
        p = self.profile(QE1="e", QE2="e", QE3="d", QE4="e")
        self.assertGreater(p.gamma_capacity, p.gamma_willingness)
        app = self.page(p)
        app.switch_page("app_pages/portfolio.py").run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.warning), 0)
        self.assertTrue(any("Capacity and allocation limits" in str(item.value) for item in app.info))
        advanced = next(item for item in app.expander if item.label == "Advanced details")
        self.assertTrue(any("Capacity floor applied" in str(item.value) for item in advanced.get("dataframe")))

    def test_risk_tolerance_heading_and_technical_details_are_separate(self):
        p = self.profile(QE1="e", QE2="d", QE3="c", QE4="e")
        app = self.page(p)
        self.assertTrue(any(item.value == f"Risk Tolerance: {p.risk_classification}" for item in app.header))
        self.assertFalse(any("Saved risk classification" in item.value for item in app.header))
        advanced = next(item for item in app.expander if item.label == "Advanced details")
        self.assertTrue(any("final gamma" in str(item.value) for item in advanced.get("markdown")))
        app.session_state["method"] = "Risk parity (ERC)"
        app.switch_page("app_pages/portfolio.py").run()
        self.assertEqual(len(app.exception), 0)
        advanced = next(item for item in app.expander if item.label == "Advanced details")
        self.assertTrue(any("ERC objective" in str(item.value) for item in advanced.get("markdown")))
        advanced_ids = {id(item) for item in advanced.get("markdown")}
        self.assertFalse(any("Binding execution allocation limits" in str(item.value)
                             for item in app.markdown if id(item) not in advanced_ids))
        self.assertGreaterEqual(len(app.get("plotly_chart")), 3)

    def test_failed_method_has_consolidated_warning_and_named_valid_alternative(self):
        original = self.portfolios
        def failed(currency, tickers, gamma, c, views):
            raw = original(currency, tickers, gamma, c, views)
            raw["Resampled MV"] = raw["Resampled MV"] * .5
            return raw
        with patch("src.app_helpers.get_portfolios", side_effect=failed):
            app = self.page(self.profile())
            app.session_state["method"] = "Resampled MV"
            app.switch_page("app_pages/portfolio.py").run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.warning), 1)
        self.assertTrue(any("Resampled MV" in str(item.value) for item in app.warning))
        self.assertTrue(any("Illustrative alternative" in str(item.value) for item in app.info))
        self.assertTrue(any(m.label == "Expected return" for m in app.metric))

    def test_infeasible_product_selection_shows_disclosed_fallback_and_preserves_requests(self):
        app = self.page(self.profile(FINANCIAL_LOW))
        app.session_state["assets"] = ["SPY", "EFA"]
        app.switch_page("app_pages/portfolio.py").run()
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any("illustrative fallback" in str(item.value).lower() for item in app.warning))
        self.assertEqual(len(app.warning), 1)
        self.assertTrue(any("adds available default products" in str(item.value) for item in app.caption))
        self.assertEqual(app.session_state["assets"], ["SPY", "EFA"])
        self.assertTrue(any(metric.label == "Volatility" for metric in app.metric))
        self.assertTrue(any("Illustrative fallback" in str(item.value) for item in app.caption))

    def test_no_reliable_fallback_displays_diagnostic(self):
        app = self.page(self.profile(FINANCIAL_LOW))
        app.session_state["assets"] = ["SPY", "EFA"]
        with patch("src.app_helpers.available_assets", return_value=["SPY", "EFA"]):
            app.switch_page("app_pages/portfolio.py").run()
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any("No reliable simulation" in str(item.value) for item in app.error))
        self.assertFalse(any(metric.label == "Expected return" for metric in app.metric))

    def test_financial_flags_stay_consolidated_across_profile_and_simulation_pages(self):
        p = self.profile(FINANCIAL_LOW)
        saved_flags = p.financial_review_flags
        app = self.page(p)
        self.assertEqual(len(app.warning), 1)
        for page in ("app_pages/backtest.py", "app_pages/time_machine.py"):
            app.switch_page(page).run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(len(app.warning), 1)
            self.assertIn("financial difficulties", app.warning[0].value)
            self.assertEqual(app.session_state["profile"].financial_review_flags, saved_flags)

    def test_missing_risk_free_data_gives_one_warning_and_no_fabricated_metrics(self):
        app = self.page(self.profile())
        with patch("src.app_helpers.get_data", return_value=(self.returns, pd.Series(dtype=float), self.prices)):
            app.switch_page("app_pages/portfolio.py").run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.warning), 1)
        self.assertIn("risk-free rate", app.warning[0].value)
        self.assertNotIn("allocation is infeasible", app.warning[0].value)
        self.assertFalse(any(m.label == "Expected return" for m in app.metric))

    def test_frontier_failure_joins_existing_warning_without_hiding_valid_portfolio(self):
        app = self.page(self.profile(FINANCIAL_LOW))
        with patch("src.optimizer.frontier", side_effect=RuntimeError("frontier failed")):
            app.switch_page("app_pages/portfolio.py").run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.warning), 1)
        self.assertIn("frontier", app.warning[0].value)
        self.assertIn("financial difficulties", app.warning[0].value)
        self.assertTrue(any(m.label == "Expected return" for m in app.metric))

    def test_backtest_and_time_machine_work_for_new_profile_without_legacy_loss_limit(self):
        app = self.page(self.profile())
        app.switch_page("app_pages/backtest.py").run()
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any("Allocation checks at rebalancing" in str(item.value) for item in app.subheader))
        self.assertFalse(any("ceiling" in metric.label.lower() for metric in app.metric))
        app.switch_page("app_pages/time_machine.py").run()
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any("QE2 willingness response" in str(item.value) for item in app.info))
        self.assertTrue(any(metric.label == "Worst loss" for metric in app.metric))


if __name__ == "__main__":
    unittest.main()
