"""
End-to-end run on real UK data: download/cache -> rolling backtest -> latest forecasts ->
oil scenario -> results/, dashboard and README section.

    python -m ukcpi.run             # real data (ONS / Bank of England / FRED, cached in data/)
    python -m ukcpi.run --refresh   # re-download
    python -m ukcpi.run --no-rl     # skip the (slow) hybrid RL re-training per origin
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .data import load_macro
from .forecast import forecast_now, oil_scenario, rolling_backtest
from .trainer import agent_kind

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def to_markdown(res: dict) -> str:
    bt, d = res["backtest"], res["data"]
    hs = bt["horizons"]
    src = "; ".join(f"{k}: {v}" for k, v in d["sources"].items()) or "cached data"
    lines = [f"**Data:** {d['start']} to {d['end']} ({d['n']} months). Sources: {src}.",
             f"**Backtest:** {bt['n_origins']} rolling origins, forecasts evaluated from {bt['evaluated_from']}"
             + (" (includes the 2021-23 inflation surge)" if bt["evaluated_from"] <= "2022-06" else "")
             + f", run {res['run_at'][:10]}.", "",
             "Out-of-sample RMSE (percentage points of CPI inflation); in brackets: relative to the naive "
             "no-change forecast (<1 = better). ✓ = significantly better than naive (Diebold-Mariano p<0.05), "
             "✗ = significantly worse.", "",
             "| Model | " + " | ".join(f"h={h}" for h in hs) + " |", "|---|" + "---|" * len(hs)]
    for m, rows in bt["models"].items():
        cells = []
        for h in hs:
            r = rows.get(str(h))
            if not r:
                cells.append("–")
                continue
            rel = r.get("rel_rmse_vs_naive")
            mark = ""
            if rel is not None and r.get("p_value") is not None and r["p_value"] == r["p_value"] and r["p_value"] < 0.05:
                mark = " ✓" if r["dm_stat"] < 0 else " ✗"
            cells.append(f"{r['rmse']:.2f}" + (f" ({rel:.2f}){mark}" if rel is not None else ""))
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    lines += ["", f"Hybrid RL agent: {res['rl_agent']}; MPC sentiment and output gap set to 0 (no historical archive).", ""]
    fc = res["forecast_now"]
    lines += ["**Latest forecasts (CPI annual rate, %):**", "", "| Target month | Naive | AR | ARX |", "|---|---|---|---|"]
    for h, r in fc.items():
        lines.append(f"| {r['target_month']} (h={h}) | {r['Naive (no change)']} | {r['AR (CPI lags)']} | {r['ARX (+ Bank Rate, Brent)']} |")
    sc = res["oil_scenario_+20pct"]
    lines += ["", "**Oil scenario (ARX, associational):** a sustained 20% rise in Brent is associated with CPI inflation "
              + ", ".join(f"{v['cpi_change_pp']:+.2f}pp at h={h} (95% CI {v['ci95'][0]:+.2f} to {v['ci95'][1]:+.2f})" for h, v in sc.items())
              + ".", ""]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--origins", type=int, default=60)
    ap.add_argument("--no-rl", action="store_true")
    a = ap.parse_args(argv)

    df = load_macro(refresh=a.refresh)
    bt = rolling_backtest(df, n_origins=a.origins, include_rl=not a.no_rl)
    res = {"run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "data": {"n": len(df), "start": df["date"].iloc[0].strftime("%Y-%m"),
                    "end": df["date"].iloc[-1].strftime("%Y-%m"), "sources": df.attrs.get("sources", {})},
           "backtest": bt, "rl_agent": agent_kind(), "forecast_now": forecast_now(df),
           "oil_scenario_+20pct": oil_scenario(df, 20.0)}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "backtest.json").write_text(json.dumps(res, indent=2))
    md = to_markdown(res)
    (RESULTS / "backtest.md").write_text(md)
    readme = ROOT / "README.md"
    s = readme.read_text(encoding="utf-8")
    if START in s:
        readme.write_text(s[:s.index(START) + len(START)] + "\n" + md + "\n" + s[s.index(END):], encoding="utf-8")
    from .dashboard import export_dashboard
    export_dashboard(df, res, str(ROOT / "output" / "dashboard.html"))
    print(md)
    return res


if __name__ == "__main__":
    main()
