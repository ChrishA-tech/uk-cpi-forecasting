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
import time
import urllib.request
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

from .exceptions import CPIForecastValidationError

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
                    "(KHTML, like Gecko) Version/17.0 Safari/605.1.15"}
MANUAL_DIR = DATA_DIR / "manual"

ONS_CSV = "https://www.ons.gov.uk/generator?format=csv&uri=/economy/inflationandpriceindices/timeseries/d7g7/mm23"
ONS_JSON = "https://api.ons.gov.uk/timeseries/d7g7/dataset/mm23/data"
BOE_CSV = ("https://www.bankofengland.co.uk/boeapps/database/_iadb-fromshowcolumns.asp?csv.x=yes"
           "&Datefrom=01/Jan/1989&Dateto=now&SeriesCodes=IUMABEDR&CSVF=TN&UsingCodes=Y&VPD=Y&VFD=N")
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
# DBnomics mirrors IMF PCPS: the same Brent series FRED republishes as POILBREUSDM.
DBNOMICS_BRENT = ("https://api.db.nomics.world/v22/series/IMF/PCPS/M.W00.POILBRE.USD"
                  "?observations=1&format=json")
MONTHS = {m: i for i, m in enumerate(["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
                                      "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], 1)}


EIA_BRENT = "https://www.eia.gov/dnav/pet/hist_xls/RBRTEm.xls"
MAX_STALE_DAYS = 270   # a source ending longer ago than this is used only if nothing fresher downloads


def _get(url: str, timeout: int = 90, attempts: int = 3, binary: bool = False):
    """GET with retries and back-off (official statistics sites can be slow to answer scripts)."""
    last = None
    for i in range(attempts):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
                body = r.read()
                return body if binary else body.decode("utf-8-sig", errors="replace")
        except Exception as e:
            last = e
            time.sleep(3 * (i + 1))
    raise last


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


def parse_dbnomics_json(text: str) -> pd.DataFrame:
    """DBnomics v22 series response: series.docs[0].period / .value ('NA' = missing)."""
    try:
        doc = json.loads(text)["series"]["docs"][0]
        periods, values = doc["period"], doc["value"]
    except (ValueError, KeyError, IndexError, TypeError) as e:
        raise CPIForecastValidationError(f"Unexpected DBnomics response ({type(e).__name__}: {e})")
    if len(periods) != len(values):
        raise CPIForecastValidationError("DBnomics periods and values differ in length.")
    out = pd.DataFrame({"date": pd.to_datetime(pd.Series(periods).astype(str).str[:7], format="%Y-%m"),
                        "value": pd.to_numeric(pd.Series(values).replace("NA", None), errors="coerce")})
    return out.dropna()


def parse_eia_xls(content: bytes) -> pd.DataFrame:
    """EIA history workbook (RBRTEm.xls): on each sheet keep rows of (date, number); use the largest sheet."""
    try:
        sheets = pd.read_excel(io.BytesIO(content), sheet_name=None, header=None, engine="xlrd")
    except Exception as e:
        raise CPIForecastValidationError(f"Could not read EIA workbook ({type(e).__name__}: {e})")
    best = None
    for sh in sheets.values():
        if sh.shape[1] < 2:
            continue
        d = pd.to_datetime(sh.iloc[:, 0], errors="coerce")
        v = pd.to_numeric(sh.iloc[:, 1], errors="coerce")
        ok = d.notna() & v.notna()
        if ok.sum() and (best is None or ok.sum() > len(best)):
            best = pd.DataFrame({"date": d[ok].dt.to_period("M").dt.to_timestamp().values, "value": v[ok].values})
    if best is None:
        raise CPIForecastValidationError("EIA workbook has no usable series.")
    return best


def is_stale(df: pd.DataFrame, max_days: int = MAX_STALE_DAYS) -> bool:
    return (pd.Timestamp.today().normalize() - pd.Timestamp(df["date"].max())).days > max_days


# ------------------------------------------------------------------ sources with fallbacks
MANUAL_FILES = {   # browser-downloaded copies, used before any network source
    "cpi_yoy": ("D7G7.csv", ONS_CSV, lambda t: parse_ons_csv(t)),
    "bank_rate": ("IUMABEDR.csv", BOE_CSV, lambda t: parse_boe_csv(t)),
    "brent_usd": ("POILBREUSDM.csv", FRED_CSV.format(series="POILBREUSDM"), lambda t: parse_fred_csv(t, "POILBREUSDM")),
}


def _manual(name: str) -> pd.DataFrame:
    fname, _, parser = MANUAL_FILES[name]
    path = MANUAL_DIR / fname
    if not path.exists():
        raise FileNotFoundError(f"no manual file {path}")
    return parser(path.read_text(encoding="utf-8-sig", errors="replace"))


def manual_instructions() -> str:
    return "\n".join(f"  {name}: open {url}\n      in a browser and save it as {MANUAL_DIR / f}"
                     for name, (f, url, _) in MANUAL_FILES.items())


SOURCES: Dict[str, List[Tuple[str, Callable[[], pd.DataFrame]]]] = {
    "cpi_yoy": [("manual file data/manual/D7G7.csv (ONS)", lambda: _manual("cpi_yoy")),
                ("ONS D7G7 (CSV)", lambda: parse_ons_csv(_get(ONS_CSV))),
                ("ONS D7G7 (API)", lambda: parse_ons_json(_get(ONS_JSON))),
                ("FRED CPALTT01GBM659N (OECD)", lambda: parse_fred_csv(_get(FRED_CSV.format(series="CPALTT01GBM659N")),
                                                                     "CPALTT01GBM659N"))],
    "bank_rate": [("manual file data/manual/IUMABEDR.csv (Bank of England)", lambda: _manual("bank_rate")),
                  ("Bank of England IUMABEDR", lambda: parse_boe_csv(_get(BOE_CSV))),
                  ("FRED IRSTCB01GBM156N (OECD)", lambda: parse_fred_csv(_get(FRED_CSV.format(series="IRSTCB01GBM156N")),
                                                                       "IRSTCB01GBM156N"))],
    "brent_usd": [("manual file data/manual/POILBREUSDM.csv (FRED)", lambda: _manual("brent_usd")),
                  ("US EIA Europe Brent Spot FOB (RBRTE)",
                   lambda: parse_eia_xls(_get(EIA_BRENT, timeout=60, attempts=2, binary=True))),
                  ("FRED POILBREUSDM (IMF)", lambda: parse_fred_csv(_get(FRED_CSV.format(series="POILBREUSDM"),
                                                                       timeout=60, attempts=2), "POILBREUSDM")),
                  ("IMF PCPS POILBRE via DBnomics (same series as FRED POILBREUSDM)",
                   lambda: parse_dbnomics_json(_get(DBNOMICS_BRENT, timeout=60, attempts=2)))],
}


def fetch_series(name: str) -> Tuple[pd.DataFrame, str]:
    errors, stale = [], []
    for label, fn in SOURCES[name]:
        try:
            df = fn()
            if len(df) <= 24:
                errors.append(f"{label}: only {len(df)} rows")
            elif is_stale(df):
                stale.append((df, label))
                errors.append(f"{label}: stale, ends {df['date'].max():%Y-%m}")
            else:
                return df, label
        except Exception as e:  # network / format errors -> try the next source
            errors.append(f"{label}: {type(e).__name__}: {e}")
    if stale:
        df, label = max(stale, key=lambda t: t[0]["date"].max())
        print(f"!! {name}: using the freshest available data ({label}, ends {df['date'].max():%Y-%m}). "
              + " | ".join(errors))
        return df, label
    raise CPIForecastValidationError(f"Could not download {name}. Tried: " + " | ".join(errors)
                                     + "\nManual option (download in a browser, then re-run):\n" + manual_instructions())


def load_macro(refresh: bool = False, start: str = "1997-06-01", data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """
    Monthly UK dataset (date, cpi_yoy, bank_rate, brent_usd), cached as data/uk_macro_monthly.csv.
    Default start: June 1997 (Bank of England operational independence, inflation-targeting regime).
    """
    cache = data_dir / "uk_macro_monthly.csv"
    meta = data_dir / "sources.json"
    if cache.exists() and not refresh and data_dir == DATA_DIR:
        cached = pd.read_csv(cache, parse_dates=["date"])
        common_end = cached.dropna()["date"].max()
        if pd.isna(common_end) or (pd.Timestamp.today().normalize() - common_end).days > MAX_STALE_DAYS:
            print(f"Cached UK data ends {common_end:%Y-%m}; trying to update it...")
            try:
                return load_macro(refresh=True, start=start, data_dir=data_dir)
            except Exception as e:
                print(f"!! Update failed ({type(e).__name__}: {e}); using the cached data.")
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
