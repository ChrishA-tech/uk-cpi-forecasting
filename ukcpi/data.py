"""
Real UK macro data, downloaded once and cached in data/:

  CPI annual rate (%)   ONS series D7G7 (CPI ANNUAL RATE 00: ALL ITEMS), dataset MM23
                        fallback: FRED CPALTT01GBM659N (OECD, UK CPI growth y/y)
  Bank Rate (%)         Bank of England IADB series IUMABEDR (monthly average of official Bank Rate)
                        fallback: FRED IRSTCB01GBM156N (OECD, UK central bank rate)
  Brent (USD/bbl)       FRED POILBREUSDM (IMF, monthly average)

    from ukcpi.data import load_macro
    df = load_macro()          # columns: date, cpi_yoy, bank_rate, brent_usd (+ source attrs)

ONS data: Open Government Licence v3.0. Bank of England data: see BoE terms of use.
"""
from __future__ import annotations

import io
import json
import re
import urllib.request
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

from .exceptions import CPIForecastValidationError

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
UA = {"User-Agent": "Mozilla/5.0 (ukcpi research script)"}

ONS_CSV = "https://www.ons.gov.uk/generator?format=csv&uri=/economy/inflationandpriceindices/timeseries/d7g7/mm23"
ONS_JSON = "https://api.ons.gov.uk/timeseries/d7g7/dataset/mm23/data"
BOE_CSV = ("https://www.bankofengland.co.uk/boeapps/database/_iadb-fromshowcolumns.asp?csv.x=yes"
           "&Datefrom=01/Jan/1989&Dateto=now&SeriesCodes=IUMABEDR&CSVF=TN&UsingCodes=Y&VPD=Y&VFD=N")
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
MONTHS = {m: i for i, m in enumerate(["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
                                      "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], 1)}


def _get(url: str, timeout: int = 30) -> str:
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
        return r.read().decode("utf-8-sig", errors="replace")


# ------------------------------------------------------------------ parsers (unit-tested)
def parse_ons_csv(text: str) -> pd.DataFrame:
    """ONS 'generator' CSV: metadata rows, then annual / quarterly / monthly rows ("1989 JAN","4.9")."""
    rows = []
    for line in text.splitlines():
        parts = [p.strip().strip('"') for p in line.split(",")]
        if len(parts) >= 2 and re.fullmatch(r"\d{4} [A-Z]{3}", parts[0]) and parts[1] not in ("", "x"):
            year, mon = parts[0].split()
            rows.append((pd.Timestamp(int(year), MONTHS[mon], 1), float(parts[1])))
    if not rows:
        raise CPIForecastValidationError("No monthly rows found in ONS CSV.")
    return pd.DataFrame(rows, columns=["date", "value"])


def parse_ons_json(text: str) -> pd.DataFrame:
    """Legacy ONS timeseries API JSON: {"months": [{"date": "1989 JAN", "value": "4.9"}, ...]}."""
    months = json.loads(text).get("months", [])
    rows = [(pd.Timestamp(int(m["date"][:4]), MONTHS[m["date"][5:8].upper()], 1), float(m["value"]))
            for m in months if m.get("value") not in (None, "")]
    if not rows:
        raise CPIForecastValidationError("No monthly rows found in ONS JSON.")
    return pd.DataFrame(rows, columns=["date", "value"])


def parse_boe_csv(text: str) -> pd.DataFrame:
    """BoE IADB CSV: 'DATE,IUMABEDR' then rows like '31 Jan 1989,12.9479'. Month-end -> month start."""
    df = pd.read_csv(io.StringIO(text))
    date_col = next((c for c in df.columns if c.strip().upper() == "DATE"), None)
    val_col = next((c for c in df.columns if c.strip().upper() == "IUMABEDR"), None)
    if date_col is None or val_col is None:
        raise CPIForecastValidationError(f"Unexpected BoE CSV columns: {list(df.columns)}")
    dates = pd.to_datetime(df[date_col], format="%d %b %Y").dt.to_period("M").dt.to_timestamp()
    return pd.DataFrame({"date": dates, "value": pd.to_numeric(df[val_col], errors="coerce")}).dropna()


def parse_fred_csv(text: str, series: str) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(text), na_values=["."])
    date_col = next((c for c in df.columns if c.lower() in ("observation_date", "date")), None)
    if date_col is None or series not in df.columns:
        raise CPIForecastValidationError(f"Unexpected FRED CSV columns: {list(df.columns)}")
    out = pd.DataFrame({"date": pd.to_datetime(df[date_col]).dt.to_period("M").dt.to_timestamp(),
                        "value": pd.to_numeric(df[series], errors="coerce")})
    return out.dropna()


# ------------------------------------------------------------------ sources with fallbacks
SOURCES: Dict[str, List[Tuple[str, Callable[[], pd.DataFrame]]]] = {
    "cpi_yoy": [("ONS D7G7 (CSV)", lambda: parse_ons_csv(_get(ONS_CSV))),
                ("ONS D7G7 (API)", lambda: parse_ons_json(_get(ONS_JSON))),
                ("FRED CPALTT01GBM659N (OECD)", lambda: parse_fred_csv(_get(FRED_CSV.format(series="CPALTT01GBM659N")),
                                                                     "CPALTT01GBM659N"))],
    "bank_rate": [("Bank of England IUMABEDR", lambda: parse_boe_csv(_get(BOE_CSV))),
                  ("FRED IRSTCB01GBM156N (OECD)", lambda: parse_fred_csv(_get(FRED_CSV.format(series="IRSTCB01GBM156N")),
                                                                       "IRSTCB01GBM156N"))],
    "brent_usd": [("FRED POILBREUSDM (IMF)", lambda: parse_fred_csv(_get(FRED_CSV.format(series="POILBREUSDM")),
                                                                  "POILBREUSDM"))],
}


def fetch_series(name: str) -> Tuple[pd.DataFrame, str]:
    errors = []
    for label, fn in SOURCES[name]:
        try:
            df = fn()
            if len(df) > 24:
                return df, label
            errors.append(f"{label}: only {len(df)} rows")
        except Exception as e:  # network / format errors -> try the next source
            errors.append(f"{label}: {type(e).__name__}: {e}")
    raise CPIForecastValidationError(f"Could not download {name}. Tried: " + " | ".join(errors))


def load_macro(refresh: bool = False, start: str = "1997-06-01", data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """
    Monthly UK dataset (date, cpi_yoy, bank_rate, brent_usd), cached as data/uk_macro_monthly.csv.
    Default start: June 1997 (Bank of England operational independence, inflation-targeting regime).
    """
    cache = data_dir / "uk_macro_monthly.csv"
    meta = data_dir / "sources.json"
    if refresh or not cache.exists():
        frames, sources = [], {}
        for name in SOURCES:
            df, label = fetch_series(name)
            frames.append(df.drop_duplicates("date", keep="last").set_index("date")["value"].rename(name))
            sources[name] = label
        merged = pd.concat(frames, axis=1).sort_index()
        data_dir.mkdir(parents=True, exist_ok=True)
        merged.reset_index().rename(columns={"index": "date"}).to_csv(cache, index=False, date_format="%Y-%m-%d")
        meta.write_text(json.dumps(sources, indent=2))
    df = pd.read_csv(cache, parse_dates=["date"])
    df = validate_macro(df[df["date"] >= pd.Timestamp(start)])
    df.attrs["sources"] = json.loads(meta.read_text()) if meta.exists() else {}
    return df


def validate_macro(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the common sample where all series exist; reject interior gaps and implausible values."""
    need = ["cpi_yoy", "bank_rate", "brent_usd"]
    missing_cols = set(need) - set(df.columns)
    if missing_cols:
        raise CPIForecastValidationError(f"Missing columns: {missing_cols}")
    df = df.sort_values("date").reset_index(drop=True)
    ok = df[need].notna().all(axis=1)
    if not ok.any():
        raise CPIForecastValidationError("No month has all three series.")
    df = df.loc[ok.idxmax(): ok[::-1].idxmax()].reset_index(drop=True)
    if df[need].isna().any().any():
        gaps = df.loc[df[need].isna().any(axis=1), "date"].dt.strftime("%Y-%m").tolist()[:5]
        raise CPIForecastValidationError(f"Gaps inside the sample, e.g. {gaps}")
    full = pd.date_range(df["date"].iloc[0], df["date"].iloc[-1], freq="MS")
    if len(full) != len(df):
        raise CPIForecastValidationError("Dates are not a gap-free monthly sequence.")
    if not df["cpi_yoy"].between(-5, 30).all() or not df["bank_rate"].between(-1, 20).all() or (df["brent_usd"] <= 0).any():
        raise CPIForecastValidationError("Implausible values (CPI outside -5..30%, Bank Rate outside -1..20%, or Brent <= 0).")
    return df
