# ---------------------------------------------------------------------------------
# Porter Competitive Advantage Agent - Ready-to-run Notebook (Descriptive Table, Global Peers, HTML export)
# ---------------------------------------------------------------------------------
# What this notebook does
# - Prompts for ANY ticker
# - Proposes MULTIPLE global peer sets (industry, sector, global megacaps) you approve/edit
# - Builds a descriptive Porter competitive-advantage table (no numeric scores)
# - Asks GPT to write a short narrative and produce a Markdown table + JSON rows
# - Builds a peer comparison table with: Gross/Operating/Net Margins, ROIC, ROIC ex Cash,
#   ROE, FCF, Debt/EBITDA, PEG, Trailing/Forward P/E (best-effort using yfinance fields)
# - Plots 5-year P/E history with mean and ±1 std dev bands and EMBEDS the image in HTML
# - Exports ONE HTML report to your Desktop: competitive_advantages_<TICKER>.html
#
# ---------------------------------------------------------------------------------
#
import os
import json
import re
import yfinance as yf
import pandas as pd
import numpy as np
from openai import OpenAI
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from dotenv import load_dotenv
from datetime import datetime, timezone, timedelta
import time
from pathlib import Path

UNIVERSE_CSV = Path("universe.csv")                 # editable universe list
UNIVERSE_CACHE = Path(".cache_universe_info.json")  # enriched cache
CACHE_TTL_DAYS = 30                                 # auto-refresh threshold

load_dotenv()  # Automatically loads values from .env

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
print(f"[porter_agent] OPENAI_API_KEY present: {bool(OPENAI_API_KEY)}")

if not OPENAI_API_KEY:
    raise EnvironmentError("Please set your OPENAI_API_KEY environment variable before running the notebook.")

client = OpenAI(api_key=OPENAI_API_KEY)
OPENAI_MODEL = "gpt-4o"

# -------- Helpers --------

def now_ts():
    return datetime.now(timezone.utc).isoformat()


def as_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def pct_str(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "N/A"
    try:
        return f"{x*100:.1f}%"
    except Exception:
        try:
            return f"{float(x)*100:.1f}%"
        except Exception:
            return "N/A"

def is_valid_number(x):
    """Return True if x is a scalar number and not NaN."""
    return (x is not None) and (not (isinstance(x, float) and np.isnan(x)))

def first(obj, keys):
    """Return the first available key as a scalar float from a pandas Series or the first row of a DataFrame."""
    if obj is None:
        return np.nan

    # If a DataFrame was passed, use its first row
    if isinstance(obj, pd.DataFrame):
        if obj.empty:
            return np.nan
        obj = obj.iloc[0]  # convert to Series

    # Now obj is a Series
    if isinstance(obj, pd.Series):
        for k in keys:
            if k in obj.index:
                val = obj[k]
                if pd.notna(val):
                    return as_float(val)

    return np.nan

def _quarter_label(ts: pd.Timestamp) -> str:
    """Format a timestamp like 'Q3 2024'."""
    if not isinstance(ts, pd.Timestamp) or pd.isna(ts):
        return "N/A"
    quarter = ((ts.month - 1) // 3) + 1
    return f"Q{quarter} {ts.year}"

def _format_billions(value: float) -> str:
    """Return a number scaled to billions with two decimals."""
    if not is_valid_number(value):
        return "N/A"
    return f"{value / 1e9:.2f}"

def _format_number(value: float, kind: str = "default") -> str:
    """Pretty-format values based on metric type."""
    if not is_valid_number(value):
        return "N/A"
    if kind == "currency":
        return _format_billions(value)
    if kind == "per_share":
        return f"{float(value):.2f}"
    if kind == "multiple":
        return f"{float(value):.2f}"
    return f"{float(value):,.0f}"

def _pct_change(new_value: float, old_value: float) -> str:
    """Return percentage change string."""
    if not (is_valid_number(new_value) and is_valid_number(old_value)):
        return "N/A"
    if old_value == 0:
        return "N/A"
    change = (float(new_value) - float(old_value)) / abs(float(old_value))
    return f"{change * 100:.1f}%"

def _prep_quarter_df(raw_df) -> pd.DataFrame:
    """Transpose a yfinance quarterly dataframe so rows are period end dates."""
    if raw_df is None or len(raw_df) == 0:
        return pd.DataFrame()
    df = raw_df.T.copy()
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()].sort_index()
    return df

def _series_from_columns(df: pd.DataFrame, candidates) -> pd.Series:
    """Return the first matching column as a numeric Series."""
    if df is None or df.empty:
        return pd.Series(dtype=float)
    for name in candidates:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce")
    return pd.Series(np.nan, index=df.index)

def _latest_and_prior_matching_quarters(index) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return latest quarter and the matching prior-year quarter if possible."""
    if not index:
        return None, None
    latest_ts = index[-1]
    prior_ts = None
    for candidate in reversed(index[:-1]):
        if candidate.quarter == latest_ts.quarter:
            prior_ts = candidate
            break
    if prior_ts is None and len(index) >= 2:
        prior_ts = index[-2]
    return latest_ts, prior_ts

def _month_end_close_series(ticker_obj, align_index: pd.Index) -> pd.Series:
    """Return month-end close prices aligned to the provided index."""
    try:
        price = ticker_obj.history(period="5y", interval="1mo")
    except Exception:
        price = pd.DataFrame()

    if price is None or price.empty:
        return pd.Series(np.nan, index=align_index)

    closes = price["Close"].copy()
    closes.index = pd.to_datetime(closes.index)
    try:
        closes.index = closes.index.tz_localize(None)
    except Exception:
        pass
    closes.index = closes.index.to_period("M").to_timestamp("M")
    closes = closes.groupby(closes.index).last().sort_index()

    return closes.reindex(align_index, method="nearest")

def _is_cache_stale(cache_path: Path, ttl_days: int = CACHE_TTL_DAYS) -> bool:
    """Return True if cache is missing or older than ttl_days."""
    if not cache_path.exists():
        return True
    try:
        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime, tz=timezone.utc)
        return (datetime.now(timezone.utc) - mtime) > timedelta(days=ttl_days)
    except Exception:
        return True

def load_universe_csv():
    """Load user-provided universe.csv or fall back to a small cross-sector list."""
    if UNIVERSE_CSV.exists():
        df = pd.read_csv(UNIVERSE_CSV)
        if "ticker" not in df.columns and "Ticker" in df.columns:
            df = df.rename(columns={"Ticker": "ticker"})
        df["ticker"] = df["ticker"].astype(str).str.strip()
        df = df.dropna(subset=["ticker"]).drop_duplicates(subset=["ticker"])
        return df[["ticker"]]
    # Minimal fallback
    return pd.DataFrame({
        "ticker": ["AAPL","MSFT","GOOGL","AMZN","META","NVDA","TSM","ORCL","KO","PEP","STZ","DEO","JPM","V","MA"]
    })


def _safe_info(t):
    try:
        return t.info
    except Exception:
        return {}


def build_or_load_enriched_universe_cache(
    universe_df: pd.DataFrame,
    rate_limit_sec: float = 0.2,
    force_rebuild: bool = False
) -> pd.DataFrame:
    """
    Returns DataFrame with: ticker, sector, industry, marketCap.
    Auto-refreshes when older than CACHE_TTL_DAYS.
    Env override: PEERS_REBUILD_CACHE=1 to force rebuild.
    """
    env_force = os.getenv("PEERS_REBUILD_CACHE", "").strip() in ("1", "true", "True")
    force_rebuild = force_rebuild or env_force

    if not force_rebuild and UNIVERSE_CACHE.exists() and not _is_cache_stale(UNIVERSE_CACHE):
        try:
            cached = pd.read_json(UNIVERSE_CACHE)
            if "ticker" in cached.columns:
                return cached
        except Exception:
            pass  # fall through

    rows = []
    tickers = universe_df["ticker"].astype(str).str.upper().tolist()
    for tk in tickers:
        info = _safe_info(yf.Ticker(tk))
        rows.append({
            "ticker": tk,
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "marketCap": info.get("marketCap"),
        })
        if rate_limit_sec:
            time.sleep(rate_limit_sec)  # be gentle to Yahoo

    enriched = pd.DataFrame(rows)
    enriched["ticker"] = enriched["ticker"].astype(str).str.upper()
    for col in ["sector", "industry"]:
        enriched[col] = enriched[col].fillna("").astype(str)

    try:
        enriched.to_json(UNIVERSE_CACHE, orient="records")
    except Exception:
        pass
    return enriched

# -------- Data Fetch --------

def fetch_yf_data(ticker):
    t = yf.Ticker(ticker)
    info = _safe_info(t)
    fin = None
    bs = None
    cf = None
    try:
        if hasattr(t, "financials") and t.financials is not None:
            fin = t.financials.T
    except Exception:
        fin = None
    try:
        if hasattr(t, "balance_sheet") and t.balance_sheet is not None:
            bs = t.balance_sheet.T
    except Exception:
        bs = None
    try:
        if hasattr(t, "cashflow") and t.cashflow is not None:
            cf = t.cashflow.T
    except Exception:
        cf = None
    metrics = {
        "ticker": ticker,
        "shortName": info.get("shortName"),
        "longName": info.get("longName"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "marketCap": info.get("marketCap"),
        "grossMargins": info.get("grossMargins"),
        "operatingMargins": info.get("operatingMargins"),
        "profitMargins": info.get("profitMargins"),
        "returnOnEquity": info.get("returnOnEquity"),
        "debtToEquity": info.get("debtToEquity"),
        "trailingPE": info.get("trailingPE"),
        "forwardPE": info.get("forwardPE"),
        "pegRatio": info.get("pegRatio"),
        "enterpriseValue": info.get("enterpriseValue"),
        "enterpriseToEbitda": info.get("enterpriseToEbitda"),
        "sharesOutstanding": info.get("sharesOutstanding"),
    }
    return metrics, fin, bs, cf

# -------- Interactive Inputs --------
def _parse_env_ticker_list(value: str) -> list[str]:
    if not value:
        return []
    return [p.strip().upper() for p in value.split(",") if p.strip()]

env_ticker = os.getenv("PORTER_TICKER", "").strip()
if env_ticker:
    TICKER = env_ticker.upper()
    print("Configuration loaded from environment. Ticker:", TICKER)
else:
    TICKER = input("Enter ticker symbol (e.g. AAPL): ") or "AAPL"
    print("Configuration loaded. Ticker:", TICKER)

metrics, fin, bs, cf = fetch_yf_data(TICKER)
print("Fetched basic metrics for", TICKER)
print("Sector:", metrics.get("sector"), "/ Industry:", metrics.get("industry"))

# -------- Propose GLOBAL peer option sets (you approve/edit) --------
def propose_peer_sets_dynamic(subject_ticker, subject_metrics, universe_info_df, max_n=8):
    """
    Build global peers for ANY industry/sector:
    1) Same sector   (top by market cap, excluding the subject)
    2) Global megacaps fallback
    """
    sub_tk = (subject_ticker or "").upper()
    sector = (subject_metrics.get("sector") or "").strip()

    df = universe_info_df.copy()
    df["ticker"] = df["ticker"].astype(str).str.upper()
    df = df[df["ticker"] != sub_tk]
    df["marketCap"] = pd.to_numeric(df["marketCap"], errors="coerce")
    df = df.sort_values("marketCap", ascending=False, na_position="last")

    # 1) Same sector
    same_sec = df[df["sector"].str.lower() == sector.lower()].head(max_n)["ticker"].tolist()

    # 2) Global megacaps fallback
    global_megacaps = ["MSFT","GOOGL","AMZN","META","NVDA","TSM","005930.KS","ORCL","AAPL","ADBE","CRM"]
    gmega = [t for t in global_megacaps if t != sub_tk and t not in same_sec][:max_n]

    return {
        "1_same_sector": same_sec,
        "2_global_megacaps": gmega
    }
# -------- Propose GLOBAL peer option sets (dynamic, any sector/industry) --------
universe_df = load_universe_csv()
universe_info_df = build_or_load_enriched_universe_cache(universe_df)

peer_sets = propose_peer_sets_dynamic(TICKER, metrics, universe_info_df, max_n=8)

print("\nSuggested peer groups for", TICKER, ":")
print("[1] Same Sector   (", metrics.get("sector"), "): ", peer_sets["1_same_sector"])
print("[2] Global Megacaps: ", peer_sets["2_global_megacaps"])
print("[3] Enter my own manually")

env_peer_override = _parse_env_ticker_list(os.getenv("PORTER_PEER_TICKERS", ""))
if env_peer_override:
    PEER_TICKERS = env_peer_override
    print("\nUsing peers from PORTER_PEER_TICKERS:", PEER_TICKERS)
else:
    choice = input("Choose 1, 2, or 3: ") or "1"
    if choice.strip() == "3":
        manual = input("Enter comma-separated peers (e.g., MSFT, GOOGL, AMZN): ")
        PEER_TICKERS = [p.strip().upper() for p in manual.split(",") if p.strip()]
    else:
        key = {"1": "1_same_sector", "2": "2_global_megacaps"}.get(choice.strip(), "1_same_sector")
        PEER_TICKERS = list(peer_sets.get(key, []))

    print("\nSelected peers:", PEER_TICKERS)
    edit = input("Edit peers? Type comma-separated list, or press Enter to accept: ")
    if edit.strip():
        PEER_TICKERS = [p.strip().upper() for p in edit.split(",") if p.strip()]
    print("Final peers:", PEER_TICKERS)


# -------- Build financial evidence for Porter table --------
fin_row = fin.iloc[0] if isinstance(fin, pd.DataFrame) and len(fin) else None
bs_row = bs.iloc[0] if isinstance(bs, pd.DataFrame) and len(bs) else None
cf_row = cf.iloc[0] if isinstance(cf, pd.DataFrame) and len(cf) else None

revenue = first(fin_row, ["Total Revenue", "TotalRevenue", "Revenue"])
rd_expense = first(fin_row, ["Research Development", "ResearchAndDevelopment", "Research Development Expense"])
operating_income = first(fin_row, ["Operating Income", "OperatingIncome"])
operating_margin_info = metrics.get("operatingMargins")
gross_margin_info = metrics.get("grossMargins")
profit_margin_info = metrics.get("profitMargins")
roe_info = metrics.get("returnOnEquity")

ocf = first(cf_row, ["Total Cash From Operating Activities", "Operating Cash Flow", "OperatingCashFlow"])
capex = first(cf_row, ["Capital Expenditures", "CapitalExpenditures"])
fcf = np.nan
if is_valid_number(ocf) and is_valid_number(capex):
    fcf = ocf - abs(capex)

cash_like = first(bs_row, [
    "Cash And Cash Equivalents",
    "CashAndCashEquivalents",
    "Cash And Short Term Investments",
    "CashAndShortTermInvestments",
    "Total Cash",
    "Cash"
])

total_debt = first(bs_row, ["Total Debt", "Short Long Term Debt", "Long Term Debt", "ShortLongTermDebt"])

# Peer stats for context
peer_metrics = {}
for p in PEER_TICKERS:
    try:
        pm, _, _, _ = fetch_yf_data(p)
        peer_metrics[p] = pm
    except Exception as e:
        print("Warning fetching", p, e)

peer_df_raw = pd.DataFrame([{"ticker": k, **v} for k, v in peer_metrics.items()])
peer_stats = {
    "grossMargins_median": as_float(peer_df_raw["grossMargins"].median()) if "grossMargins" in peer_df_raw else np.nan,
    "operatingMargins_median": as_float(peer_df_raw["operatingMargins"].median()) if "operatingMargins" in peer_df_raw else np.nan,
    "profitMargins_median": as_float(peer_df_raw["profitMargins"].median()) if "profitMargins" in peer_df_raw else np.nan,
    "returnOnEquity_median": as_float(peer_df_raw["returnOnEquity"].median()) if "returnOnEquity" in peer_df_raw else np.nan,
}

advantage_rows = [
    {
        "porter_category": "Cost Leadership / Scale Economies",
        "specific_advantage": "Massive supply chain leverage & scale purchasing",
        "financial_metric_evidence": {
            "operating_margin": pct_str(operating_margin_info),
            "peer_operating_margin_median": pct_str(peer_stats.get("operatingMargins_median")),
            "gross_margin": pct_str(gross_margin_info),
            "peer_gross_margin_median": pct_str(peer_stats.get("grossMargins_median")),
            "market_cap": f"{metrics.get('marketCap'):,}" if metrics.get('marketCap') else "N/A",
        },
    },
    {
        "porter_category": "Differentiation / Brand",
        "specific_advantage": "Premium brand & pricing power across devices/services",
        "financial_metric_evidence": {
            "gross_margin": pct_str(gross_margin_info),
            "peer_gross_margin_median": pct_str(peer_stats.get("grossMargins_median")),
            "profit_margin": pct_str(profit_margin_info),
            "peer_profit_margin_median": pct_str(peer_stats.get("profitMargins_median")),
        },
    },
    {
        "porter_category": "Switching Costs / Ecosystem Lock-in",
        "specific_advantage": "Tight integration of hardware, OS, and services",
        "financial_metric_evidence": {
            "profit_margin": pct_str(profit_margin_info),
            "roe": pct_str(roe_info),
        },
    },
    {
        "porter_category": "Network Effects",
        "specific_advantage": "Two-sided platform of users and developers/partners",
        "financial_metric_evidence": {
            "profit_margin": pct_str(profit_margin_info),
            "gross_margin": pct_str(gross_margin_info),
        },
    },
    {
        "porter_category": "Intangible Assets / IP",
        "specific_advantage": "Proprietary silicon/algorithms, OS, patents, proprietary data",
        "financial_metric_evidence": {
            "roe": pct_str(roe_info),
            "r_and_d_pct_revenue": (f"{(rd_expense/revenue)*100:.1f}%" if pd.notna(revenue) and pd.notna(rd_expense) and revenue != 0 else "N/A"),
        },
    },
    {
        "porter_category": "Distribution / Channel Power",
        "specific_advantage": "Owned retail/online direct channel and advantaged distribution",
        "financial_metric_evidence": {
            "gross_margin": pct_str(gross_margin_info),
        },
    },
    {
        "porter_category": "Financial / Capital Strength",
        "specific_advantage": "Large, stable free cash flow enabling investment and buybacks",
        "financial_metric_evidence": {
            "free_cash_flow": (f"{int(fcf):,}" if is_valid_number(fcf) else "N/A"),
            "cash_or_equivalents": (f"{int(cash_like):,}" if is_valid_number(cash_like) else "N/A"),
            "total_debt": (f"{int(total_debt):,}" if is_valid_number(total_debt) else "N/A"),
            "debt_to_equity": (f"{as_float(metrics.get('debtToEquity')):.2f}" if metrics.get('debtToEquity') is not None else "N/A"),
        },
    },
]

company_blob = {
    "ticker": TICKER,
    "as_of": now_ts(),
    "company": metrics.get("longName") or metrics.get("shortName") or TICKER,
    "metrics": metrics,
    "peer_stats": peer_stats,
    "derived": {
        "revenue": revenue,
        "r_and_d": rd_expense,
        "operating_income": operating_income,
        "operating_cash_flow": ocf,
        "capex": capex,
        "free_cash_flow": fcf,
        "cash_or_equivalents": cash_like,
        "total_debt": total_debt,
    },
    "advantage_rows": advantage_rows,
}

# -------- GPT: Narrative + Table (+ JSON rows) --------
SYSTEM_PROMPT = (
    "You are PorterBot, an investment research assistant. Create a concise, evidence-backed table of a company's "
    "key competitive advantages mapped to Porter categories. Map only to categories that apply to each respective company. Highlight the most important competitive advantage/s in bold. Do not fabricate numbers; only use numeric fields provided. "
    "OUTPUT FORMAT STRICTLY: (1) A 5-6 line narrative. (2) A Markdown table with columns: "
    "'Porter Category' | 'Specific Company Advantage' | 'Example / Evidence (qualitative)' | 'Financial Metric Evidence (quantitative)'. "
    "(3) Then a JSON block between 'JSON_ROWS_BEGIN' and 'JSON_ROWS_END' containing an array where each item has: "
    "porter_category, specific_advantage, qualitative_example, financial_metrics (a dict of label->value strings)."
)

USER_PROMPT = (
    "Using the JSON payload below, write the narrative, table, and JSON block. Keep column names exactly as specified. "
    "For the last column of the table, concatenate the provided metrics as 'Label: Value' pairs. If a metric is missing, write 'N/A'.\n\n"
    "PAYLOAD:\n" + json.dumps(company_blob, default=str)
)

print("Calling OpenAI model", OPENAI_MODEL, "-- this requires a working API key and network access.")
model_output = None
try:
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ],
        temperature=0.2,
        max_completion_tokens=1200,
        timeout=30,
        stream=False
    )
    model_output = resp.choices[0].message.content
except Exception as e:
    print("OpenAI API call failed:", e)
    print("Continuing with local fallback for Porter table.")

    # --- Extract a clean narrative up-front so we can reuse it later ---

def _strip_code_fences(txt: str) -> str:
    # remove ``` blocks
    return re.sub(r"```.*?```", "", txt, flags=re.S)

def _first_paragraph(txt: str) -> str:
    # take from first non-empty line until a blank line
    lines = [l.rstrip() for l in txt.splitlines()]
    # drop any leading preview headers like '--- MODEL OUTPUT (truncated) ---'
    while lines and (not lines[0] or lines[0].startswith("---")):
        lines.pop(0)
    if not lines:
        return ""
    # accumulate until blank line
    buf = []
    for l in lines:
        if not l.strip():
            break
        buf.append(l)
    return "\n".join(buf).strip()

def extract_narrative_from_model_output(model_output_text: str) -> str:
    """
    Returns only the narrative paragraph(s) from the model output.
    We cut off before the Markdown table or JSON rows.
    """
    if not model_output_text:
        return ""

    txt = model_output_text.strip()

    # Stop at the Markdown table header if present
    cut_points = []
    p = txt.find("| Porter Category")
    if p != -1:
        cut_points.append(p)

    # Stop at JSON block if present
    j = txt.find("JSON_ROWS_BEGIN")
    if j != -1:
        cut_points.append(j)

    if cut_points:
        txt = txt[: min(cut_points)].strip()

    # Strip a leading '--- ...' line if it’s there
    if txt.startswith("---"):
        first_nl = txt.find("\n")
        txt = txt[first_nl + 1 :].strip() if first_nl != -1 else txt

    return txt

def extract_narrative_strict(model_output_text: str) -> str:
    """
    Cleanly isolate the narrative and fall back to the first paragraph if needed.
    """
    if not model_output_text:
        return ""

    # Remove code fences and trim whitespace first
    cleaned = _strip_code_fences(str(model_output_text)).strip()

    # Trim off table / JSON sections if present
    narrative_only = extract_narrative_from_model_output(cleaned)
    if not narrative_only:
        narrative_only = cleaned

    # Prefer the first paragraph to avoid spilling into table headings
    first_para = _first_paragraph(narrative_only)
    if first_para:
        return first_para

    return narrative_only.strip()

# fallback to whatever remains if no paragraph split

# Materialize a ready-to-embed HTML snippet now
narrative_text = extract_narrative_strict(model_output)
try:
    from markdown import markdown as _md
    NARRATIVE_HTML = _md(narrative_text) if narrative_text else "<em>No narrative extracted.</em>"
except Exception:
    NARRATIVE_HTML = (narrative_text or "No narrative extracted.").replace("\n", "<br>")


# -------- Parse JSON rows from the model output (for reliable HTML table) --------
rows_df = None
if model_output:
    json_match = re.search(r"JSON_ROWS_BEGIN\s*(\[.*?\])\s*JSON_ROWS_END", model_output, flags=re.S)
    if json_match:
        try:
            rows = json.loads(json_match.group(1))
            norm = []
            for r in rows:
                fm = r.get("financial_metrics") or {}
                fm_pairs = [f"{k}: {v}" for k, v in fm.items()]
                norm.append({
                    "Porter Category": r.get("porter_category"),
                    "Specific Company Advantage": r.get("specific_advantage"),
                    "Example / Evidence (qualitative)": r.get("qualitative_example"),
                    "Financial Metric Evidence (quantitative)": "; ".join(fm_pairs)
                })
            rows_df = pd.DataFrame(norm)
        except Exception as e:
            print("Failed to parse JSON rows:", e)

if rows_df is None:
    # Fallback: build from local advantage_rows
    norm = []
    for r in advantage_rows:
        fm = r.get("financial_metric_evidence") or {}
        fm_pairs = [f"{k}: {v}" for k, v in fm.items()]
        norm.append({
            "Porter Category": r.get("porter_category"),
            "Specific Company Advantage": r.get("specific_advantage"),
            "Example / Evidence (qualitative)": "(See narrative)",
            "Financial Metric Evidence (quantitative)": "; ".join(fm_pairs)
        })
    rows_df = pd.DataFrame(norm)

# -------- Build Peer Comparison Table (with ROIC using EBIT) --------
# -------- Helper: compute Forward P/E consistent with chart --------
def compute_forward_pe_like_chart(ticker_info: dict, last_close_price: float, ticker_obj=None) -> float:
    """
    Forward P/E consistent with the chart: price / forwardEps.
    Falls back to analyst estimates if forwardEps missing.
    """
    try:
        fwd_eps = ticker_info.get("forwardEps")

        # --- fallback 1: try analyst estimates from earnings_trend ---
        if (fwd_eps is None or (isinstance(fwd_eps, float) and np.isnan(fwd_eps))) and ticker_obj is not None:
            try:
                trend = ticker_obj.earnings_trend
                if trend is not None and not trend.empty:
                    if "forwardEps" in trend.columns:
                        vals = trend["forwardEps"].dropna()
                        if not vals.empty:
                            fwd_eps = float(vals.iloc[0])
            except Exception:
                pass

        if fwd_eps is None or float(fwd_eps) == 0 or np.isnan(fwd_eps):
            return np.nan
        if last_close_price is None or float(last_close_price) == 0 or np.isnan(last_close_price):
            return np.nan

        val = float(last_close_price) / float(fwd_eps)
        return val if np.isfinite(val) else np.nan
    except Exception:
        return np.nan

# -------- Helper: compute Forward P/E monthly series (chart & table share this) --------
def compute_forward_pe_monthly_series(ticker: str) -> pd.Series:
    """
    Returns a monthly forward P/E series for the last ~5 years.
    Method:
      - Monthly Close price (5y, 1mo)
      - EPS path from annual Net Income / sharesOutstanding
      - Optionally blend forwardEps at 'today'
      - If annual NI missing, try quarterly NI (TTM via rolling(4))
      - Interpolate EPS monthly, align to prices, compute P/E, cap outliers
    """
    t = yf.Ticker(ticker)
    info = _safe_info(t)

    # Prices
    price = t.history(period="5y", interval="1mo")
    if price is None or price.empty:
        return pd.Series(dtype=float)

    # Shares
    shares = info.get("sharesOutstanding") or np.nan

    # Annual NI -> EPS
    fin_a = t.financials if hasattr(t, "financials") else None
    ni_a = pd.Series(dtype=float)
    if fin_a is not None and not fin_a.empty:
        if "Net Income" in fin_a.index:
            ni_a = fin_a.loc["Net Income"].dropna()
        elif "NetIncome" in fin_a.index:
            ni_a = fin_a.loc["NetIncome"].dropna()
        if not ni_a.empty:
            ni_a.index = pd.to_datetime(ni_a.index)
            ni_a = ni_a.sort_index()

    eps_hist = pd.Series(dtype=float)
    if not ni_a.empty and shares and shares > 0:
        eps_hist = ni_a / float(shares)

    # If no annual EPS path, try quarterly TTM EPS
    if eps_hist.empty:
        qfin = t.quarterly_financials
        ni_q = pd.Series(dtype=float)
        if qfin is not None and not qfin.empty:
            if "Net Income" in qfin.index:
                ni_q = qfin.loc["Net Income"].dropna()
            elif "NetIncome" in qfin.index:
                ni_q = qfin.loc["NetIncome"].dropna()
            if not ni_q.empty and shares and shares > 0:
                ni_q.index = pd.to_datetime(ni_q.index)
                ni_q = ni_q.sort_index()
                eps_hist = ni_q.rolling(4).sum() / float(shares)  # TTM EPS from quarterlies

    # Blend forward EPS at 'today' if present
    fwd_eps = info.get("forwardEps")
    if fwd_eps is not None and not (isinstance(fwd_eps, float) and np.isnan(fwd_eps)):
        try:
            eps_hist.loc[pd.Timestamp.today().normalize()] = float(fwd_eps)
        except Exception:
            pass

    if eps_hist.empty:
        return pd.Series(dtype=float)

    # Interpolate to monthly, align with monthly prices
    eps_m = eps_hist.sort_index().resample("M").interpolate("linear")
    px = price["Close"]
    px.index = pd.to_datetime(px.index).tz_localize(None)
    eps_m.index = eps_m.index.tz_localize(None)

    px_aligned = px.reindex(eps_m.index, method="nearest")
    pe = (px_aligned / eps_m).replace([np.inf, -np.inf], np.nan).dropna()
    pe = pe[pe < 200]  # clip extreme outliers

    return pe

# -------- Helper: compute current Forward P/E (month-end close / forwardEps) --------
def compute_forward_pe_now(ticker: str) -> float:
    """
    Forward P/E (now) = last month-end close / info['forwardEps'].
    Fallback to info['forwardPE'] if forwardEps missing/invalid.
    """
    t = yf.Ticker(ticker)
    info = _safe_info(t)

    price = t.history(period="5y", interval="1mo")
    if price is None or price.empty:
        last_close = np.nan
    else:
        px = price["Close"].copy()
        px.index = pd.to_datetime(px.index)
        try:
            px.index = px.index.tz_localize(None)
        except Exception:
            pass
        px.index = px.index.to_period("M").to_timestamp("M")
        px = px.groupby(px.index).last().sort_index()
        last_close = float(px.dropna().iloc[-1]) if not px.dropna().empty else np.nan

    fwd_eps = info.get("forwardEps")
    fwd_pe_vendor = info.get("forwardPE")

    try:
        if isinstance(fwd_eps, (int, float)) and np.isfinite(fwd_eps) and fwd_eps > 0 \
           and isinstance(last_close, (int, float)) and np.isfinite(last_close) and last_close > 0:
            return float(last_close) / float(fwd_eps)
    except Exception:
        pass

    try:
        if isinstance(fwd_pe_vendor, (int, float)) and np.isfinite(fwd_pe_vendor) and fwd_pe_vendor > 0:
            return float(fwd_pe_vendor)
    except Exception:
        pass

    return np.nan

def build_latest_earnings_release_section(ticker: str, metrics: dict) -> str:
    """
    Build an HTML table showing the latest reported quarter vs the same quarter last year.
    Returns HTML snippet (table + optional EPS estimate block).
    """
    try:
        ticker_obj = yf.Ticker(ticker)
    except Exception:
        return "<p>Latest earnings release data unavailable (ticker lookup failed).</p>"

    q_fin = _prep_quarter_df(getattr(ticker_obj, "quarterly_financials", None))
    if q_fin.empty:
        return "<p>Latest earnings release data unavailable (no quarterly financials).</p>"

    q_cf = _prep_quarter_df(getattr(ticker_obj, "quarterly_cashflow", None))
    quarter_idx = list(q_fin.index)
    if len(quarter_idx) < 2:
        return "<p>Latest earnings release data unavailable (insufficient quarters).</p>"

    latest_ts, prior_ts = _latest_and_prior_matching_quarters(quarter_idx)
    if latest_ts is None or prior_ts is None:
        return "<p>Latest earnings release data unavailable (matching quarter not found).</p>"

    prior_label = _quarter_label(prior_ts)
    latest_label = _quarter_label(latest_ts)

    revenue = _series_from_columns(q_fin, ["Total Revenue", "TotalRevenue", "Revenue"])
    gross_profit = _series_from_columns(q_fin, ["Gross Profit", "GrossProfit"])
    operating_income = _series_from_columns(q_fin, ["Operating Income", "OperatingIncome"])
    net_income = _series_from_columns(q_fin, ["Net Income", "NetIncome"])
    ocf = _series_from_columns(q_cf, [
        "Total Cash From Operating Activities",
        "Operating Cash Flow",
        "OperatingCashFlow"
    ])
    investing_cf = _series_from_columns(q_cf, [
        "Total Cashflows From Investing Activities",
        "Investments",
        "Total Cash From Investing Activities",
        "Capital Expenditures"
    ])
    financing_cf = _series_from_columns(q_cf, [
        "Total Cash From Financing Activities",
        "Cash From Financing Activities",
        "Net Borrowings"
    ])

    shares = metrics.get("sharesOutstanding") or ticker_obj.info.get("sharesOutstanding")
    try:
        shares = float(shares) if shares not in (None, "") else np.nan
    except Exception:
        shares = np.nan

    eps_series = pd.Series(np.nan, index=q_fin.index)
    if is_valid_number(shares) and shares > 0 and not net_income.empty:
        eps_series = net_income.astype(float) / shares

    prices_quarter = _month_end_close_series(ticker_obj, q_fin.index)

    ttm_eps = eps_series.rolling(4).sum()
    trailing_pe = prices_quarter / ttm_eps

    info = _safe_info(ticker_obj)
    forward_eps = info.get("forwardEps")
    forward_pe_series = pd.Series(np.nan, index=prices_quarter.index)
    if is_valid_number(forward_eps) and forward_eps > 0:
        forward_pe_series = prices_quarter / float(forward_eps)
    elif is_valid_number(info.get("forwardPE")):
        forward_pe_series = pd.Series(float(info.get("forwardPE")), index=prices_quarter.index)

    metric_rows = [
        ("Revenue", revenue, "currency"),
        ("Gross Profit", gross_profit, "currency"),
        ("Operating Income", operating_income, "currency"),
        ("EPS", eps_series, "per_share"),
        ("Trailing P/E", trailing_pe, "multiple"),
        ("Future P/E", forward_pe_series, "multiple"),
        ("Operating Cash Flow", ocf, "currency"),
        ("Investing Cash Flow", investing_cf, "currency"),
        ("Financing Cash Flow", financing_cf, "currency"),
    ]

    def _value_at(series: pd.Series, ts: pd.Timestamp) -> float:
        if series is None or len(series) == 0:
            return np.nan
        if ts not in series.index:
            return np.nan
        try:
            return float(series.loc[ts])
        except Exception:
            return np.nan

    rows_html = "\n".join(
        (
            "<tr>"
            f"<td><strong>{label}</strong></td>"
            f"<td>{_format_number(_value_at(series, prior_ts), kind)}</td>"
            f"<td>{_format_number(_value_at(series, latest_ts), kind)}</td>"
            f"<td>{_pct_change(_value_at(series, latest_ts), _value_at(series, prior_ts))}</td>"
            "</tr>"
        )
        for label, series, kind in metric_rows
    )

    table_html = f"""
    <div style='margin:25px 0; font-family:Arial, sans-serif;'>
      <h2 style='color:#222;'>Latest Earnings Release</h2>
      <table style='border-collapse:collapse; width:100%; font-size:14px;'>
        <thead>
          <tr style='background:#f2f2f2; text-align:left;'>
            <th style='padding:8px; border:1px solid #ddd;'>USD Billions (except per share)</th>
            <th style='padding:8px; border:1px solid #ddd;'>{prior_label}</th>
            <th style='padding:8px; border:1px solid #ddd;'>{latest_label}</th>
            <th style='padding:8px; border:1px solid #ddd;'>Δ YoY</th>
          </tr>
        </thead>
        <tbody>
          {''.join(rows_html)}
        </tbody>
      </table>
    """

    closing_div = "</div>"
    return table_html + closing_div
def get_rows_for_peer_table(ticker):
    m, finr, bsr, cfr = fetch_yf_data(ticker)

    # --- Forward P/E for table: use SAME formula as chart overlay ---
    fwd_pe = compute_forward_pe_now(ticker)
    try:
        print(f"[DEBUG {ticker}] table_forward_pe_now={fwd_pe!r}")
    except Exception:
        pass

    # --- Core fields ---

    gross = m.get('grossMargins')
    oper = m.get('operatingMargins')
    netm = m.get('profitMargins')
    roe = m.get('returnOnEquity')
    enterprise_value = m.get('enterpriseValue')
    ev_to_ebitda_info = m.get('enterpriseToEbitda')
    trailing_pe = m.get('trailingPE')

    # Income/CF/BS rows
    ebit = first(finr, ["Operating Income", "OperatingIncome"])  # EBIT proxy
    if np.isnan(ebit):
        # Fallback: NI + Interest + Taxes if available
        ni = first(finr, ["Net Income", "NetIncome"])
        interest = first(finr, ["Interest Expense", "InterestExpense"])
        taxes = first(finr, ["Income Tax Expense", "Tax Provision", "IncomeTaxExpense"])
        vals = [v for v in [ni, interest, taxes] if not np.isnan(v)]
        ebit = float(np.nansum(vals)) if vals else np.nan

    da = first(cfr, ["Depreciation", "Depreciation And Amortization", "Depreciation & Amortization"])  # for EBITDA approx
    ebitda = (0 if np.isnan(ebit) else ebit) + (0 if np.isnan(da) else da)

    debt = first(bsr, ["Total Debt", "Short Long Term Debt", "Long Term Debt", "ShortLongTermDebt"])
    equity = first(bsr, ["Total Stockholder Equity", "Total Equity Gross Minority Interest", "TotalEquityGrossMinorityInterest", "Total Stockholders Equity"])
    cash_ = first(bsr, [
        "Cash And Cash Equivalents", "CashAndCashEquivalents",
        "Cash And Short Term Investments", "CashAndShortTermInvestments",
        "Total Cash", "Cash"
    ])

    # Ratios
    denom_roic = (0 if np.isnan(debt) else debt) + (0 if np.isnan(equity) else equity)
    roic = np.nan if (np.isnan(ebit) or denom_roic == 0 or np.isnan(denom_roic)) else (ebit / denom_roic)

    denom_roic_exc = denom_roic - (0 if np.isnan(cash_) else cash_)
    roic_excash = "N/A" if (isinstance(denom_roic_exc, float) and (np.isnan(denom_roic_exc) or denom_roic_exc <= 0)) else (np.nan if np.isnan(ebit) else (ebit / denom_roic_exc))

    net_debt = np.nan
    if is_valid_number(debt):
        net_debt = debt - (cash_ if is_valid_number(cash_) else 0.0)

    net_debt_ebitda = np.nan
    if is_valid_number(ebitda) and ebitda != 0 and is_valid_number(net_debt):
        net_debt_ebitda = net_debt / ebitda

    ev_to_ebitda = np.nan
    if is_valid_number(ev_to_ebitda_info):
        ev_to_ebitda = float(ev_to_ebitda_info)
    elif is_valid_number(enterprise_value) and is_valid_number(ebitda) and ebitda != 0:
        ev_to_ebitda = enterprise_value / ebitda

    def fmt_ratio(x, pct=False):
        if isinstance(x, str):
            return x
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "N/A"
        return f"{x*100:.1f}%" if pct else f"{x:.2f}"

    return {
        'Ticker': ticker,
        'Gross Margin': fmt_ratio(gross, pct=True),
        'Operating Margin': fmt_ratio(oper, pct=True),
        'Net Margin': fmt_ratio(netm, pct=True),
        'ROIC': ("N/A" if (isinstance(roic, float) and np.isnan(roic)) else fmt_ratio(roic, pct=True)),
        'ROIC ex Cash': (
            roic_excash if isinstance(roic_excash, str)
            else ("N/A" if (isinstance(roic_excash, float) and np.isnan(roic_excash)) else fmt_ratio(roic_excash, pct=True))
        ),
        'ROE': fmt_ratio(roe, pct=True),
        'EV / EBITDA': ("N/A" if np.isnan(ev_to_ebitda) else f"{ev_to_ebitda:.2f}"),
        'Net Debt / EBITDA': ("N/A" if np.isnan(net_debt_ebitda) else f"{net_debt_ebitda:.2f}"),
        'Trailing P/E': ("N/A" if trailing_pe is None else f"{trailing_pe:.2f}"),
        'Forward P/E': ("N/A" if not is_valid_number(fwd_pe) else f"{fwd_pe:.2f}"),
    }

peer_rows = [get_rows_for_peer_table(TICKER)]
for t in PEER_TICKERS:
    peer_rows.append(get_rows_for_peer_table(t))
peer_df_full = pd.DataFrame(peer_rows)

def build_peer_table_html(df: pd.DataFrame, subject_ticker: str) -> str:
    """
    Render the peer comparison DataFrame with the subject company row highlighted.
    """
    def _highlight_row(row):
        color = "background-color:#e6f2ff" if row.get("Ticker") == subject_ticker else ""
        return [color] * len(row)

    try:
        styler = (
            df.style
            .apply(_highlight_row, axis=1)
            .hide(axis="index")
        )
        return styler.to_html()
    except Exception:
        df_copy = df.copy()
        df_copy["Subject"] = np.where(df_copy["Ticker"] == subject_ticker, "◀", "")
        return df_copy.to_html(index=False, escape=False)

# -------- Forward P/E chart (using current Yahoo forwardEps) --------
def build_forward_pe_chart_html(ticker: str, period: str = "2y", interval: str = "1mo") -> str:
    """
    Plots Forward P/E over a recent window using Yahoo's *current* forwardEps:
        Forward P/E(t) = Close(t) / forwardEps_current

    Notes:
      • This is not a point-in-time / historical forward EPS series—Yahoo doesn’t expose that via yfinance.
      • The series reflects price changes over time against the current forward EPS.
      • If forwardEps is missing, falls back to a flat line at info['forwardPE'].
    """
    t = yf.Ticker(ticker)
    info = _safe_info(t)

    # Get price history
    try:
        px = t.history(period=period, interval=interval)
    except Exception:
        px = pd.DataFrame()
    if px is None or px.empty:
        return "<p>Forward P/E chart unavailable (no price history).</p>"

    # Normalize index (make tz-naive, group to unique period points)
    s = px["Close"].copy()
    s.index = pd.to_datetime(s.index)
    try:
        s.index = s.index.tz_localize(None)
    except Exception:
        pass
    # Ensure only one point per period (e.g., last of month)
    if interval.endswith("mo"):
        s.index = s.index.to_period("M").to_timestamp("M")
        s = s.groupby(s.index).last().sort_index()

    fwd_eps = info.get("forwardEps")
    fwd_pe_vendor = info.get("forwardPE")

    # Preferred: derive series from price / forwardEps
    pe_series = pd.Series(dtype=float)
    if isinstance(fwd_eps, (int, float)) and np.isfinite(fwd_eps) and fwd_eps > 0:
        pe_series = (s / float(fwd_eps)).replace([np.inf, -np.inf], np.nan).dropna()
    elif isinstance(fwd_pe_vendor, (int, float)) and np.isfinite(fwd_pe_vendor) and fwd_pe_vendor > 0:
        # If we only have a vendor forward PE (scalar), draw a flat line
        pe_series = pd.Series(float(fwd_pe_vendor), index=s.index)

    if pe_series.empty:
        return "<p>Forward P/E chart unavailable (no forward EPS/PE in Yahoo info).</p>"

    # Clean extreme spikes
    if np.isfinite(pe_series).sum() >= 10:
        upper = np.nanpercentile(pe_series.values, 99)
        pe_series = pe_series.clip(upper=upper)

    current_forward_pe = float(pe_series.iloc[-1])

    # Plot
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.plot(pe_series.index, pe_series.values, linewidth=2.0, label="Forward P/E (Price / current forward EPS)")
    ax.scatter(pe_series.index[-1], pe_series.iloc[-1], color="black", zorder=5)
    ax.text(pe_series.index[-1], pe_series.iloc[-1], f" {current_forward_pe:.1f}×", va="bottom", fontsize=9)

    # Overlay statistical bands (mean and ±1 std dev)
    pe_mean = float(np.nanmean(pe_series.values))
    pe_std = float(np.nanstd(pe_series.values))
    ax.axhline(pe_mean, color="#ff7f0e", linestyle="--", linewidth=1.4, label=f"Mean ({pe_mean:.1f}×)")
    ax.axhline(pe_mean + pe_std, color="#2ca02c", linestyle=":", linewidth=1.2, label="+1σ")
    ax.axhline(max(pe_mean - pe_std, 0), color="#d62728", linestyle=":", linewidth=1.2, label="-1σ")

    # Label with method so it’s clear
    ax.set_title(f"{ticker} — Forward P/E (Yahoo current forward EPS basis)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Forward P/E")
    ax.legend(frameon=False)

    # Footnote / caption for method transparency
    caption = ("Method: Forward P/E(t) = Close(t) / forwardEps (current). "
               "Yahoo Finance does not expose historical forward EPS via yfinance; "
               "series reflects price movement against current forward EPS. "
               "Dashed/ dotted lines show the mean and ±1σ bands of the displayed window.")
    fig.text(0.01, -0.02, caption, ha="left", va="top", fontsize=9)

    fig.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    return f"<img alt='Forward PE history (current forward EPS basis)' src='data:image/png;base64,{b64}'/>"

# -------- Build 5-year P/E history and embed chart --------
def build_pe_history_html_embed(ticker):
    """
    5-year P/E chart
      • Blue line: Trailing P/E built from price / ANNUAL EPS (Net Income / shares), stepped monthly.
      • Orange dashed line: Current Forward P/E (last month-end price / forwardEps).
    Month-end pinning is applied to BOTH price and EPS to ensure alignment.
    """
    t = yf.Ticker(ticker)

    # 5y monthly prices
    price = t.history(period="5y", interval="1mo")
    if price is None or price.empty:
        return "<p>P/E chart unavailable (no price data)</p>"

    # PRICE -> month-end index
    px_m = price["Close"].copy()
    px_m.index = pd.to_datetime(px_m.index)
    try:
        px_m.index = px_m.index.tz_localize(None)
    except Exception:
        pass
    # Pin to month-end & keep last per month
    px_m.index = px_m.index.to_period("M").to_timestamp("M")
    px_m = px_m.groupby(px_m.index).last().sort_index()

    # Build monthly index directly from price's month-end index
    monthly_idx = pd.date_range(px_m.index.min(), px_m.index.max(), freq="M")

    # Info (for shares and forward P/E overlay later)
    info = _safe_info(t)
    shares = info.get("sharesOutstanding")
    try:
        shares = float(shares) if shares not in (None, "") else np.nan
    except Exception:
        shares = np.nan

    # Annual Net Income -> annual EPS
    fin_a = getattr(t, "financials", None)
    if fin_a is None or fin_a.empty:
        return "<p>P/E chart unavailable (no annual financials)</p>"

    if "Net Income" in fin_a.index:
        ni_a = fin_a.loc["Net Income"].dropna()
    elif "NetIncome" in fin_a.index:
        ni_a = fin_a.loc["NetIncome"].dropna()
    else:
        return "<p>P/E chart unavailable (no Net Income row)</p>"

    if ni_a.empty or not shares or shares == 0 or np.isnan(shares):
        return "<p>P/E chart unavailable (insufficient EPS data)</p>"

    ni_a.index = pd.to_datetime(ni_a.index)
    try:
        ni_a.index = ni_a.index.tz_localize(None)
    except Exception:
        pass
    ni_a = ni_a.sort_index()

    eps_annual = ni_a / shares
    # Pin ANNUAL EPS to month-end too
    eps_annual.index = eps_annual.index.to_period("M").to_timestamp("M")

    # EPS monthly step over EXACT same month-end index as price
    eps_step = eps_annual.reindex(monthly_idx).ffill().bfill()

    # Align price to same monthly index (now guaranteed to match)
    px_m = px_m.reindex(monthly_idx)

    # Compute trailing P/E safely (blue line)
    valid = px_m.notna() & eps_step.notna() & (eps_step != 0)
    pe_raw = pd.Series(np.where(valid, px_m / eps_step, np.nan), index=monthly_idx)

    # Clip extreme outliers but keep history
    if np.isfinite(pe_raw).sum() >= 10:
        upper = np.nanpercentile(pe_raw.values, 99)
        pe_series = pe_raw.clip(upper=upper)
    else:
        pe_series = pe_raw
    pe_series = pe_series.replace([np.inf, -np.inf], np.nan).dropna()

    if pe_series.empty:
        return "<p>P/E chart unavailable (no valid P/E after alignment)</p>"

    # Stats for bands
    pe_mean = pe_series.mean()
    pe_std  = pe_series.std(ddof=1)

    # Compute Forward P/E overlay (same helper the table uses)
    fwd_pe_now = compute_forward_pe_now(ticker)
    try:
        fwd_pe_now = float(fwd_pe_now)
    except Exception:
        fwd_pe_now = np.nan

    # Plot
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(pe_series.index, pe_series.values, linewidth=2.0, label="Trailing P/E (Price / Annual EPS step)")
    ax.axhline(pe_mean, color="gray", linestyle="--", linewidth=1.0, label=f"Trailing P/E mean = {pe_mean:.1f}")
    ax.fill_between(pe_series.index, pe_mean - pe_std, pe_mean + pe_std, alpha=0.25, label="±1σ (trailing)")

    # Mark latest trailing P/E
    latest_date = pe_series.index[-1]
    latest_trailing = pe_series.iloc[-1]
    ax.scatter(latest_date, latest_trailing, color="black", zorder=5)
    ax.text(latest_date, latest_trailing, f" Trailing ≈ {latest_trailing:.1f}×", va="bottom", fontsize=9)

    # Overlay CURRENT Forward P/E (orange dashed), if available
    if isinstance(fwd_pe_now, float) and np.isfinite(fwd_pe_now):
        ax.axhline(fwd_pe_now, color="tab:orange", linestyle="--", linewidth=1.2,
                   label=f"Current Forward P/E (Price/forwardEps) ≈ {fwd_pe_now:.1f}×")

    ax.set_title(f"{ticker} — 5-Year P/E\nBlue: Trailing (Price/Annual EPS) · Orange: Current Forward (Price/forwardEps)")
    ax.set_xlabel("Date")
    ax.set_ylabel("P/E")
    ax.legend(frameon=False)

    fig.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    return f"<img alt='PE history (trailing vs forward overlay)' src='data:image/png;base64,{b64}'/>"

# -------- Assemble HTML sections --------
latest_earnings_html = build_latest_earnings_release_section(TICKER, metrics)
GLOBAL_STYLES = """
<style>
body {
  background-color: #ffffff;
  font-family: Arial, sans-serif;
  color: #222;
}
.report-root {
  max-width: 1100px;
  margin: 0 auto;
  padding: 30px 30px 60px;
  background: #ffffff;
}
.report-root h1, .report-root h2 {
  font-family: Arial, sans-serif;
}
.report-root table {
  width: 100%;
  border-collapse: collapse;
  font-size: 14px;
}
.report-root table th,
.report-root table td {
  border: 1px solid #e2e6ef;
  padding: 8px 10px;
}
.report-root table th {
  background: #f5f7fb;
  font-weight: 600;
}
.table-wrapper {
  width: 100%;
  overflow-x: auto;
  margin-bottom: 25px;
}
.chart-wrapper img {
  max-width: 100%;
  height: auto;
  display: block;
  margin: 10px auto;
}
</style>
"""
html_sections = [GLOBAL_STYLES, "<div class='report-root'>"]

# Header
header_html = f"""
<h1 style='font-family:Arial, sans-serif; color:#222;'>
  Competitive Advantages & Peer Comparison — {company_blob['company']} ({TICKER})
</h1>
<p style='font-family:Arial, sans-serif; color:#444;'>
  <strong>Sector:</strong> {metrics.get('sector', 'N/A')} &nbsp; | &nbsp;
  <strong>Industry:</strong> {metrics.get('industry', 'N/A')}<br>
  <strong>As of:</strong> {company_blob['as_of']}
</p>
"""
html_sections.append(header_html)

# Executive Summary from GPT (narrative) – use prebuilt NARRATIVE_HTML
if not model_output:
    narrative_html = "<em>No narrative available (GPT output missing).</em>"
else:
    narrative_html = NARRATIVE_HTML

summary_block = f"""
<div style='margin-bottom:25px; font-family:Arial, sans-serif;'>
  <h2 style='color:#222;'>Executive Summary</h2>
  <div style='color:#333; font-size:15px; line-height:1.5;'>{narrative_html}</div>
  <p style='font-size:13px; color:#777; margin-top:10px;'>
    Narrative summary generated by <strong>GPT-4o</strong>.
  </p>
</div>
"""
html_sections.append(summary_block)

# Latest earnings release table
html_sections.append(latest_earnings_html)

# Porter table
porter_table_html = rows_df.to_html(index=False, classes="porter-table")
html_sections.append("<h2>Porter Competitive Advantage Table</h2><div class='table-wrapper porter-table-wrapper'>" + porter_table_html + "</div>")

# Peer table
peer_table_html = build_peer_table_html(peer_df_full, TICKER)
html_sections.append("<h2>Peer Comparison</h2><div class='table-wrapper peer-table-wrapper'>" + peer_table_html + "</div>")

# P/E chart
html_sections.append("<h2>Forward P/E</h2><div class='chart-wrapper'>" + build_forward_pe_chart_html(TICKER, period="2y", interval="1mo") + "</div>")
# Show the current forward P/E number under the chart
current_fwd_pe = compute_forward_pe_now(TICKER)
try:
    current_fwd_pe_val = float(current_fwd_pe)
    html_sections.append(f"<p style='color:#555;font-size:13px;'>Current Forward P/E (Yahoo): <strong>{current_fwd_pe_val:.2f}×</strong></p>")
except Exception:
    pass

# -------- Write final HTML to Desktop --------
html_sections.append("</div>")
final_html = "\n".join(html_sections)

try:
    desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop', f'competitive_advantages_{TICKER}.html')
    with open(desktop_path, 'w', encoding='utf-8') as f:
        f.write(final_html)
    print(f"Saved HTML report to {desktop_path}")
except Exception as e:
    print("Could not export HTML:", e)

# -------- Save payload for reproducibility --------
outname = f"porter_company_blob_{TICKER}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(outname, 'w', encoding='utf-8') as f:
    json.dump(company_blob, f, indent=2, default=str)
print("Saved company_blob to", outname)
