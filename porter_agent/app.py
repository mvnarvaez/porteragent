import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf

UNIVERSE_CSV = Path("universe.csv")
UNIVERSE_COLUMNS = ["ticker", "sector", "industry", "marketCap"]
DEFAULT_UNIVERSE = [
    {"ticker": "AAPL", "sector": "Information Technology", "industry": "Consumer Electronics", "marketCap": None},
    {"ticker": "MSFT", "sector": "Information Technology", "industry": "Systems Software", "marketCap": None},
    {"ticker": "GOOGL", "sector": "Communication Services", "industry": "Internet Content & Information", "marketCap": None},
    {"ticker": "AMZN", "sector": "Consumer Discretionary", "industry": "Internet Retail", "marketCap": None},
    {"ticker": "META", "sector": "Communication Services", "industry": "Internet Content & Information", "marketCap": None},
    {"ticker": "NVDA", "sector": "Information Technology", "industry": "Semiconductors", "marketCap": None},
    {"ticker": "TSM", "sector": "Information Technology", "industry": "Semiconductors", "marketCap": None},
    {"ticker": "ORCL", "sector": "Information Technology", "industry": "Software Infrastructure", "marketCap": None},
    {"ticker": "KO", "sector": "Consumer Staples", "industry": "Beverages—Non-Alcoholic", "marketCap": None},
    {"ticker": "PEP", "sector": "Consumer Staples", "industry": "Beverages—Non-Alcoholic", "marketCap": None},
    {"ticker": "STZ", "sector": "Consumer Staples", "industry": "Beverages—Wineries & Distilleries", "marketCap": None},
    {"ticker": "DEO", "sector": "Consumer Staples", "industry": "Beverages—Wineries & Distilleries", "marketCap": None},
    {"ticker": "JPM", "sector": "Financials", "industry": "Banks—Diversified", "marketCap": None},
    {"ticker": "V", "sector": "Financials", "industry": "Credit Services", "marketCap": None},
    {"ticker": "MA", "sector": "Financials", "industry": "Credit Services", "marketCap": None},
]


def load_universe_info() -> pd.DataFrame:
    """Load peer universe metadata from universe.csv or fallback defaults."""
    if UNIVERSE_CSV.exists():
        df = pd.read_csv(UNIVERSE_CSV)
        df.columns = [str(c).strip() for c in df.columns]
        rename_map = {}
        for col in df.columns:
            lower = col.lower()
            if lower == "ticker":
                rename_map[col] = "ticker"
            elif lower == "sector":
                rename_map[col] = "sector"
            elif lower == "industry":
                rename_map[col] = "industry"
            elif lower in {"marketcap", "market_cap"}:
                rename_map[col] = "marketCap"
        if rename_map:
            df = df.rename(columns=rename_map)
        df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
        df = df.dropna(subset=["ticker"]).drop_duplicates(subset=["ticker"])
        for col in ["sector", "industry"]:
            if col not in df.columns:
                df[col] = ""
            else:
                df[col] = df[col].fillna("").astype(str)
        if "marketCap" not in df.columns:
            df["marketCap"] = None
        else:
            df["marketCap"] = pd.to_numeric(df["marketCap"], errors="coerce")
        return df[UNIVERSE_COLUMNS]

    return pd.DataFrame(DEFAULT_UNIVERSE)


def fetch_subject_info(ticker: str, universe_df: pd.DataFrame) -> dict:
    """Fetch sector/industry info, falling back to universe.csv metadata."""
    ticker_u = ticker.upper()
    info = {}
    try:
        info = yf.Ticker(ticker_u).info
    except Exception:
        info = {}

    sector = (info.get("sector") or "").strip()
    industry = (info.get("industry") or "").strip()

    if (not sector or not industry) and not universe_df.empty:
        row = universe_df[universe_df["ticker"] == ticker_u]
        if not row.empty:
            sector = sector or row.iloc[0]["sector"]
            industry = industry or row.iloc[0]["industry"]

    return {"ticker": ticker_u, "sector": sector, "industry": industry}


def propose_peer_sets(subject_ticker: str, subject_metrics: dict, universe_df: pd.DataFrame, max_n: int = 8):
    """Mimic the CLI's logic but allow sector fallback if sub-industry comes back empty."""
    sub_tk = subject_ticker.upper()
    sub_industry = (subject_metrics.get("industry") or "").strip()
    sector = (subject_metrics.get("sector") or "").strip()

    df = universe_df.copy()
    df["ticker"] = df["ticker"].astype(str).str.upper()
    df = df[df["ticker"] != sub_tk]
    df["marketCap"] = pd.to_numeric(df["marketCap"], errors="coerce")
    df = df.sort_values("marketCap", ascending=False, na_position="last")

    same_sub = df[df["industry"].str.lower() == sub_industry.lower()].head(max_n)["ticker"].tolist()
    label = sub_industry or "N/A"
    if not same_sub and sector:
        same_sub = df[df["sector"].str.lower() == sector.lower()].head(max_n)["ticker"].tolist()
        label = f"{sector} (sector fallback)"

    global_megacaps = ["MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSM", "005930.KS", "ORCL", "AAPL", "ADBE", "CRM"]
    mega_peers = [t for t in global_megacaps if t != sub_tk and t not in same_sub][:max_n]
    return {"same_sub": same_sub, "global": mega_peers, "label": label}


def render_app():
    st.set_page_config(page_title="Porter Competitive Advantage Agent", layout="wide")
    st.title("Porter Competitive Advantage Agent")
    st.write(
        "Enter a ticker, review/edit the suggested peers, then generate the HTML report without touching the CLI."
    )

    with st.sidebar:
        st.header("Run Settings")
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            st.success("OPENAI_API_KEY detected", icon="✅")
        else:
            st.error("Set OPENAI_API_KEY in your environment before running the report.", icon="⚠️")

    ticker_input = st.text_input("Ticker", value="AAPL").strip().upper()
    universe_info = load_universe_info()

    peer_section = st.container()
    selected_peers = []
    peer_defaults = []
    label = "N/A"

    if ticker_input:
        subject_info = fetch_subject_info(ticker_input, universe_info)
        if not subject_info["sector"] and not subject_info["industry"]:
            peer_section.warning(
                "Ticker not found in universe.csv. Showing global peer defaults until metadata is provided."
            )
        suggestions = propose_peer_sets(ticker_input, subject_info, universe_info)
        peer_defaults = suggestions["same_sub"] or suggestions["global"]
        label = suggestions["label"]

        peer_section.markdown(
            f"**Suggested peers ({label}):** {', '.join(peer_defaults) if peer_defaults else 'No matches'}"
        )
        peers_input_default = ", ".join(peer_defaults)
        peers_input = peer_section.text_input(
            "Peers (comma-separated, max 8 recommended)",
            value=peers_input_default,
            key=f"peer-text-{ticker_input}",
        )
        selected_peers = [p.strip().upper() for p in peers_input.split(",") if p.strip()]
    else:
        peer_section.info("Enter a ticker to see peer suggestions.")

    extra_peers = peer_section.text_input("Add extra tickers (comma-separated)").upper()
    if extra_peers:
        selected_peers = sorted(
            set(selected_peers + [p.strip().upper() for p in extra_peers.split(",") if p.strip()])
        )

    def run_agent(ticker: str, peers: list[str]):
        env = os.environ.copy()
        env["PORTER_TICKER"] = ticker.upper()
        if peers:
            env["PORTER_PEER_TICKERS"] = ",".join(peers)
        python_bin = sys.executable or "python3"
        cmd = [python_bin, "porter_agent/main.py"]
        return subprocess.run(
            cmd,
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            env=env,
        )

    generate = st.button("Generate Porter Report", disabled=not ticker_input)

    if generate:
        if not selected_peers:
            st.warning("Please select at least one peer before generating the report.", icon="⚠️")
        else:
            with st.spinner("Running Porter Agent..."):
                result = run_agent(ticker_input, selected_peers)

            stdout = result.stdout or ""
            stderr = result.stderr or ""
            if result.returncode != 0:
                st.error("Porter agent failed. See logs below.", icon="❌")
                st.code(stdout + "\n" + stderr)
            else:
                st.success("Report generated successfully!", icon="✅")

                def extract_path(prefix: str):
                    for line in stdout.splitlines():
                        if prefix in line:
                            return line.split(prefix, 1)[-1].strip()
                    return None

                html_path = extract_path("Saved HTML report to")
                json_path = extract_path("Saved company_blob to")

                root = Path(__file__).resolve().parents[1]
                if not html_path:
                    html_path = os.path.join(Path.home(), "Desktop", f"competitive_advantages_{ticker_input}.html")
                if not json_path:
                    matching = sorted(root.glob(f"porter_company_blob_{ticker_input}_*.json"))
                    if matching:
                        json_path = str(matching[-1])

                html_content = None
                if html_path and os.path.exists(html_path):
                    with open(html_path, "r", encoding="utf-8") as f:
                        html_content = f.read()
                    st.download_button(
                        "Download HTML report",
                        data=html_content,
                        file_name=Path(html_path).name,
                        mime="text/html",
                    )
                    iframe_html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset='utf-8'/>
<style>
  body {{
    background-color: #ffffff;
    margin: 0;
    padding: 20px;
    font-family: Arial, sans-serif;
  }}
</style>
</head>
<body>
{html_content}
</body>
</html>
"""
                    components.html(iframe_html, height=900, scrolling=True)
                else:
                    st.warning("Could not locate the HTML file on disk.", icon="ℹ️")

                if json_path and os.path.exists(json_path):
                    with open(json_path, "r", encoding="utf-8") as f:
                        json_content = f.read()
                    st.download_button(
                        "Download JSON payload",
                        data=json_content,
                        file_name=Path(json_path).name,
                        mime="application/json",
                    )

                with st.expander("Execution log"):
                    st.code(stdout if stdout else "<no stdout>")
                    if stderr:
                        st.code(stderr)


if __name__ == "__main__":
    render_app()
