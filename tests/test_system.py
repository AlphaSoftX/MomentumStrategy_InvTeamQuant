import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import features
import model_ml
import strategy
import metrics
import index_members
import pipeline

def make_prices(n_months=150, n_stocks=40, seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2008-01-31", periods=n_months, freq="ME")
    x = rng.normal(0.006, 0.04, (n_months, n_stocks))
    # Add mild persistence so the toy series exercises the momentum path.
    for i in range(1, n_months):
        x[i] += 0.12 * x[i - 1]
    prices = 100 * np.exp(np.cumsum(x, axis=0))
    return pd.DataFrame(prices, index=idx, columns=[f"S{i:03d}" for i in range(n_stocks)])

def test_feature_timing_and_panel():
    p = make_prices(40, 8)
    panel = features.build_feature_panel(np.log(p), lookbacks=(3, 6, 9, 12))
    assert panel.index.names == ["date", "ticker"]
    first = panel.index.get_level_values("date").min()
    assert first >= pd.Timestamp("2009-02-28")

    # For 3m momentum at t, the formula is logP(t-1)-logP(t-4).
    t = first
    ticker = panel.xs(t, level="date").index[0]
    expected = np.log(p.loc[t - pd.offsets.MonthEnd(1), ticker]) - np.log(
        p.loc[t - pd.offsets.MonthEnd(4), ticker]
    )
    assert np.isclose(panel.loc[(t, ticker), "mom_3m"], expected)

def test_model_and_strategy_end_to_end():
    prices = make_prices(150, 40)
    eligibility = pd.DataFrame(True, index=prices.index, columns=prices.columns)
    logp = np.log(prices)
    panel = features.build_feature_panel(logp)
    # Simulate a stock becoming ineligible; it must disappear from both
    # the model cross-section and the trading selection.
    eligibility.loc[eligibility.index[-6:], eligibility.columns[-1]] = False
    keys = pd.MultiIndex.from_arrays(
        [panel.index.get_level_values("date"), panel.index.get_level_values("ticker")],
        names=["date", "ticker"],
    )
    eligible_rows = eligibility.stack(future_stack=True).reindex(keys).fillna(False)
    panel = panel.loc[eligible_rows.to_numpy(dtype=bool)]
    z = features.zscore_cross_section(panel)
    labels = features.build_labels(logp, 1)
    scores, diagnostics = model_ml.train_predict_walkforward(
        z, labels, z.columns.tolist(), k=1, embargo=1,
        min_train_months=36, alpha_grid=(0.001, 0.01, 0.03),
        min_train_rows=100,
    )
    assert not scores.empty
    assert not diagnostics.empty
    eq, turn = strategy.run_strategy(
        prices, scores, 0.25, 1, eligible=eligibility,
        weighting="inv_vol", missing_return_policy="liquidate"
    )
    assert len(eq) >= 2
    assert np.isfinite(eq.to_numpy()).all()
    assert (turn["turnover"] >= 0).all()
    assert (turn["turnover"] <= 1.0 + 1e-12).all()

def test_missing_return_not_ignored():
    idx = pd.date_range("2020-01-31", periods=4, freq="ME")
    prices = pd.DataFrame(
        {"A": [100.0, 110.0, np.nan, np.nan],
         "B": [100.0, 100.0, 100.0, 100.0]},
        index=idx,
    )
    scores = pd.DataFrame({"A": [1.0, 1.0], "B": [0.0, 0.0]}, index=[idx[0], idx[1]])
    eq, turn = strategy.run_strategy(
        prices, scores, 0.5, 1, weighting="equal",
        require_positive_score=True, missing_return_policy="liquidate"
    )
    assert turn.iloc[1]["missing_returns"] == 1

def test_spearman_ic_is_rank_based():
    pred = np.array([0.1, 0.2, 0.4, 0.3])
    actual = np.array([10, 20, 40, 30])
    assert np.isclose(model_ml._rank_ic(pred, actual), 1.0)

def test_membership_state_machine():
    events = pd.DataFrame({
        "symbol": ["A.NS", "A.NS", "A.NS", "B.NS"],
        "date": pd.to_datetime(["2020-01-31", "2020-06-30", "2020-03-31", "2020-01-31"]),
        "action": ["IN", "OUT", "IN", "IN"],
        "precision": ["exact"] * 4,
        "source": ["test"] * 4,
    })
    out = index_members.build_intervals(events, {"A.NS", "B.NS"}, as_of=pd.Timestamp("2021-01-31"))
    a = out.loc[out["symbol"] == "A.NS", "intervals"].iloc[0]
    assert len(a) >= 1
    assert pd.Timestamp(a[0][0]) == pd.Timestamp("2020-01-31")

def test_frozen_holdout_uses_pre_holdout_only():
    prices = make_prices(150, 30)
    logp = np.log(prices)
    panel = features.zscore_cross_section(features.build_feature_panel(logp))
    labels = features.build_labels(logp, 1)
    holdout = pd.Timestamp("2017-01-31")
    diag = model_ml.train_predict_walkforward(
        panel, labels, panel.columns.tolist(), k=1, embargo=1,
        min_train_months=36, min_train_rows=100,
        alpha_grid=(0.001, 0.01),
    )[1]
    mult = model_ml.choose_fixed_alpha_from_diagnostics(diag.loc[diag["date"] < holdout])
    frozen = model_ml.fit_frozen_predict(
        panel, labels, panel.columns.tolist(),
        prediction_dates=prices.index[prices.index >= holdout],
        train_end=holdout - pd.offsets.MonthEnd(1),
        k=1, embargo=1, alpha_multiplier=mult, min_train_rows=100,
    )
    assert not frozen.empty
    assert frozen.index.min() >= holdout


# ---------------------------------------------------------------- regression tests (audit fixes)
def test_turnover_accounts_for_weight_drift():
    # A doubles every month, B flat. Equal-weight rebalance back to 50/50 each month:
    # drifted weights are 2/3 and 1/3, so one-way turnover must be 1/6, not 0.
    idx = pd.date_range("2020-01-31", periods=5, freq="ME")
    prices = pd.DataFrame({"A": [100, 200, 400, 800, 1600.0], "B": [100.0] * 5}, index=idx)
    scores = pd.DataFrame({"A": [1.0] * 4, "B": [1.0] * 4}, index=idx[:4])
    _, turn = strategy.run_strategy(
        prices, scores, 1.0, 1, weighting="equal", tx_cost_bps=0, slippage_bps=0
    )
    assert np.allclose(turn["turnover"].iloc[1:], 1.0 / 6.0)

def test_sortino_is_downside_deviation():
    eq = pd.Series(np.cumprod([1.0] + [1.02, 0.99] * 12))
    r = eq.pct_change().dropna()
    expected = r.mean() * 12 / (np.sqrt((np.minimum(r, 0) ** 2).mean()) * np.sqrt(12))
    assert np.isclose(metrics.compute_metrics(eq)["Sortino"], expected)

def test_mask_panel_matches_rowwise_definition():
    p = make_prices(60, 20)
    rng = np.random.default_rng(0)
    el = pd.DataFrame(rng.random(p.shape) > 0.3, index=p.index, columns=p.columns)
    panel = features.build_feature_panel(np.log(p))
    expected = [bool(el.loc[d, t]) for d, t in panel.index]
    assert pipeline.mask_panel(panel, el).equals(panel.loc[np.asarray(expected)])

def _fake_yf(monkeypatch, fn):
    import data_loader, yfinance as yf
    monkeypatch.setattr(yf, "download", fn)
    monkeypatch.setattr(data_loader, "yf", yf)
    monkeypatch.setattr(data_loader.time, "sleep", lambda *_: None)
    return data_loader

def _yahoo_frame(sym, start, end):
    idx = pd.bdate_range(start or "2006-01-01", end or "2024-06-01")
    df = pd.DataFrame({c: 100.0 + np.arange(len(idx)) for c in
                       ["Open", "High", "Low", "Close", "Adj Close", "Volume", "Dividends", "Stock Splits"]}, index=idx)
    df.columns = pd.MultiIndex.from_product([df.columns, [sym]])
    return df

def test_loader_raises_clear_error_when_yahoo_blocks_everything(monkeypatch, tmp_path):
    import yfinance.shared as shared
    def blocked(sym, **kw):
        shared._ERRORS[sym] = "JSONDecodeError('Expecting value')"
        return pd.DataFrame()
    dl_mod = _fake_yf(monkeypatch, blocked)
    dl = dl_mod.DataLoader(data_dir=tmp_path, retry_backoff=0)
    try:
        dl.get_data(["AAA.NS", "BBB.NS"], "2020-01-01", "2021-01-01", wide=True)
        raise AssertionError("expected RuntimeError")
    except RuntimeError as exc:
        assert "download problem" in str(exc)

def test_loader_retries_transient_empty_response(monkeypatch, tmp_path):
    import yfinance.shared as shared
    calls = {"n": 0}
    def flaky(sym, start=None, end=None, **kw):
        calls["n"] += 1
        if calls["n"] < 3:
            shared._ERRORS[sym] = "YFRateLimitError('Too Many Requests')"
            return pd.DataFrame()
        return _yahoo_frame(sym, start, end)
    dl_mod = _fake_yf(monkeypatch, flaky)
    wide = dl_mod.DataLoader(data_dir=tmp_path, retry_backoff=0).get_data(
        ["AAA.NS"], "2020-01-01", "2021-01-01", wide=True)
    assert not wide.empty and calls["n"] == 3

def test_loader_does_not_poison_cache_coverage_on_failed_tail(monkeypatch, tmp_path):
    import json, yfinance.shared as shared
    dl_mod = _fake_yf(monkeypatch, lambda sym, start=None, end=None, **kw: _yahoo_frame(sym, start, "2020-06-30"))
    dl = dl_mod.DataLoader(data_dir=tmp_path, retry_backoff=0)
    dl.get_data(["AAA.NS"], "2019-01-01", "2020-07-01", wide=True)
    meta = next((tmp_path / "raw" / "1d").glob("*.json"))
    meta.write_text(json.dumps({"start": "2019-01-01", "through": "2020-06-30"}))
    def rate_limited(sym, **kw):
        shared._ERRORS[sym] = "YFRateLimitError('Too Many Requests')"
        return pd.DataFrame()
    dl_mod = _fake_yf(monkeypatch, rate_limited)
    dl_mod.DataLoader(data_dir=tmp_path, retry_backoff=0).get_data(["AAA.NS"], "2019-01-01", "2023-01-01", wide=True)
    assert json.loads(meta.read_text())["through"] == "2020-06-30"

def test_loader_works_with_old_yfinance_signature(monkeypatch, tmp_path):
    def old(symbol, start=None, end=None, interval="1d", auto_adjust=False, actions=True,
            repair=False, progress=False, threads=False):
        return _yahoo_frame(symbol, start, end)
    dl_mod = _fake_yf(monkeypatch, old)
    wide = dl_mod.DataLoader(data_dir=tmp_path, retry_backoff=0).get_data(
        ["AAA.NS"], "2020-01-01", "2021-01-01", wide=True)
    assert not wide.empty


# ---------------------------------------------------------------- index_members regression tests
def test_first_snapshot_members_that_later_leave_are_kept():
    snaps = {pd.Timestamp("2013-04-30"): {"A.NS", "B.NS"},
             pd.Timestamp("2022-05-04"): {"A.NS", "C.NS"},
             pd.Timestamp("2026-09-29"): {"A.NS", "C.NS"}}
    iv = index_members.build_intervals(index_members.snapshots_to_events(snaps), {"A.NS", "C.NS"},
                                       as_of=pd.Timestamp("2026-09-29"))
    members = index_members.members_as_of(iv, pd.Timestamp("2015-06-30"))
    assert "B.NS" in members and "C.NS" not in members and "A.NS" in members
    assert "B.NS" not in index_members.members_as_of(iv, pd.Timestamp("2023-01-31"))

def test_floor_start_does_not_override_exact_events():
    # B joined 2010 (exact), left 2012 (exact), is in the 2013 snapshot again, leaves 2022.
    snaps = {pd.Timestamp("2013-04-30"): {"B.NS"}, pd.Timestamp("2022-05-04"): set(),
             pd.Timestamp("2026-09-29"): set()}
    exact = index_members._events_frame(["B.NS", "B.NS"], pd.to_datetime(["2010-03-31", "2012-03-31"]),
                                        ["IN", "OUT"], "exact", "xls")
    ev = pd.concat([exact, index_members.snapshots_to_events(snaps)], ignore_index=True)
    iv = index_members.build_intervals(ev, set(), as_of=pd.Timestamp("2026-09-29"))
    m = lambda d: "B.NS" in index_members.members_as_of(iv, pd.Timestamp(d))
    assert not m("2005-01-31")          # floor (2000) must NOT make B a member before its exact 2010 entry
    assert m("2011-01-31") and not m("2012-09-30")

def _sample_names():
    index_members._NAME_MAP.clear()
    index_members._register_names(pd.DataFrame({
        "Company Name": ["Reliance Industries Ltd.", "Tata Motors Limited", "M&M Financial Services Ltd."],
        "Symbol": ["RELIANCE", "TATAMOTORS", "M&MFIN"]}))

def test_name_resolution_handles_suffixes_punctuation_and_ampersand():
    _sample_names()
    assert index_members.resolve_name("RELIANCE INDUSTRIES LIMITED") == "RELIANCE.NS"
    assert index_members.resolve_name("Tata Motors Ltd") == "TATAMOTORS.NS"
    assert index_members.resolve_name("M & M Financial Services Limited") == "M&MFIN.NS"
    assert index_members.resolve_name("Totally Unknown Co") is None

def test_incl_excl_sheet_with_real_column_layout_is_parsed():
    _sample_names()
    raw = pd.DataFrame({
        "Index Name": ["Nifty 500"] * 4,
        "Event Date": ["2019-03-29", "2019-03-29", "2018-09-28", "2018-09-28"],
        "Scrip Name": ["Reliance Industries Ltd", "Tata Motors Ltd", "Long Gone Corp Ltd", "Reliance Industries Ltd"],
        "Description": ["Inclusion", "Exclusion", "Exclusion", "weird text"],
    })
    sink = []
    ev = index_members.parse_incl_excl_frame(raw, sink)
    got = {(r.symbol, r.action) for r in ev.itertuples()}
    assert got == {("RELIANCE.NS", "IN"), ("TATAMOTORS.NS", "OUT")}
    assert (ev["precision"] == "exact").all()
    assert any("Long Gone" in row[0] for row in sink)      # unmappable names are reported, not guessed

def test_pdf_text_rows_resolve_by_name_then_symbol():
    _sample_names()
    lines = ["NIFTY 500 Index constituents", "Company Name Weightage(%)",
             "1 Reliance Industries Ltd. 5.31", "2 Tata Motors Limited 1.20",
             "3 Obscure Holdings Ltd 0.01", "Disclaimer: this is not a data row"]
    found, unresolved = index_members.symbols_from_text_lines(lines, {"RELIANCE", "TATAMOTORS"})
    assert found == {"RELIANCE.NS", "TATAMOTORS.NS"}
    assert unresolved == ["Obscure Holdings Ltd"]

def test_pdf_snapshot_with_too_few_resolved_names_is_rejected(monkeypatch):
    import io, zipfile
    _sample_names()
    monkeypatch.setattr(index_members, "_pdf_lines", lambda b: ["1 Reliance Industries Ltd. 5.31"])
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("cnx500_Aug2013.pdf", b"%PDF-fake")
    assert index_members._symbols_from_zip(buf.getvalue()) == set()   # partial parse must not create fake churn

def test_http_get_retries_connection_errors(monkeypatch):
    import requests
    calls = {"n": 0}
    class FakeSession:
        def get(self, url, **kw):
            calls["n"] += 1
            if calls["n"] < 3:
                raise requests.exceptions.SSLError("WRONG_VERSION_NUMBER")
            class R:
                status_code = 200
                def raise_for_status(self): pass
            return R()
    monkeypatch.setattr(index_members, "_http_session", lambda headers=None: FakeSession())
    monkeypatch.setattr(index_members.time, "sleep", lambda *_: None)
    index_members._http_get("https://web.archive.org/x")
    assert calls["n"] == 3

def test_real_pdf_round_trip_resolves_names_with_numbers_and_extra_columns():
    import io, zipfile, pytest
    canvas = pytest.importorskip("reportlab.pdfgen.canvas")
    pytest.importorskip("pypdf")
    from reportlab.lib.pagesizes import A4
    raw = ["360 ONE WAM Ltd.", "3M India Ltd."] + [f"Company {i:03d} Ltd." for i in range(498)]
    syms = ["360ONE", "3MINDIA"] + [f"CO{i:03d}" for i in range(498)]
    index_members._NAME_MAP.clear()
    index_members._register_names(pd.DataFrame({"Company Name": raw, "Symbol": syms}))
    buf = io.BytesIO(); c = canvas.Canvas(buf, pagesize=A4); y = 800
    for i, n in enumerate(raw, 1):
        c.drawString(40, y, str(i)); c.drawString(70, y, n)
        c.drawString(300, y, "Financial Services"); c.drawString(450, y, "0.20")
        y -= 14
        if y < 40:
            c.showPage(); y = 800
    c.save()
    zb = io.BytesIO()
    with zipfile.ZipFile(zb, "w") as zf:
        zf.writestr("indices_dataFeb2014/cnx500_Feb2014.pdf", buf.getvalue())
    got = index_members._symbols_from_zip(zb.getvalue())
    assert len(got) == 500 and "360ONE.NS" in got
