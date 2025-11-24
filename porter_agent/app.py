import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf

UNIVERSE_CACHE = Path(".cache_universe_info.json")
UNIVERSE_CSV = Path("universe.csv")
DEFAULT_UNIVERSE = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "META",
    "NVDA",
    "TSM",
    "ORCL",
    "KO",
    "PEP",
    "STZ",
    "DEO",
    "JPM",
    "V",
    "MA",
]
def _load_cached_universe() -> Optional[pd.DataFrame]:
    if not UNIVERSE_CACHE.exists():
        return None
    try:
        cached = pd.read_json(UNIVERSE_CACHE)
        if "ticker" in cached.columns:
            cached["ticker"] = cached["ticker"].astype(str).str.upper()
            return cached
    except Exception:
        return None
    return None


def _load_universe_source() -> pd.DataFrame:
    if UNIVERSE_CSV.exists():
        df = pd.read_csv(UNIVERSE_CSV)
        if "ticker" not in df.columns and "Ticker" in df.columns:
            df = df.rename(columns={"Ticker": "ticker"})
        df["ticker"] = df["ticker"].astype(str).str.strip()
        df = df.dropna(subset=["ticker"]).drop_duplicates(subset=["ticker"])
        return df[["ticker"]]
    return pd.DataFrame({"ticker": DEFAULT_UNIVERSE})


def _has_metadata(df: Optional[pd.DataFrame]) -> bool:
    if df is None or df.empty:
        return False
    for col in ("sector", "industry"):
        if col in df.columns:
            series = df[col].astype(str).str.strip()
            if series.ne("").any():
                return True
    return False


@st.cache_data(show_spinner=False)
def load_universe_info() -> pd.DataFrame:
    """Load cached universe metadata for peer selection."""
    cached = _load_cached_universe()
    if cached is not None:
        return cached

    base = _load_universe_source()
    base["sector"] = ""
    base["industry"] = ""
    base["marketCap"] = None
    return base


@st.cache_data(show_spinner=False)
def fetch_subject_info(ticker: str) -> dict:
    """Fetch sector/industry info for the selected ticker."""
    info = {}
    try:
        info = yf.Ticker(ticker).info
    except Exception:
        pass
    return {
        "ticker": ticker.upper(),
        "sector": (info.get("sector") or "").strip(),
        "industry": (info.get("industry") or "").strip(),
    }


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
        rebuild_requested = st.button(
            "Refresh peer metadata",
            help="Use when you've rebuilt .cache_universe_info.json via the CLI.",
        )

    ticker_input = st.text_input("Ticker", value="AAPL").strip().upper()
    universe_info = load_universe_info()
    if not _has_metadata(universe_info):
        st.sidebar.warning(
            "Peer metadata cache is empty. Showing global fallback peers until sector data is rebuilt.",
            icon="ℹ️",
        )
        st.sidebar.info(
            "To rebuild, run `PEERS_REBUILD_CACHE=1 python porter_agent/main.py` locally, then redeploy Streamlit.",
            icon="💡",
        )
    elif rebuild_requested:
        st.sidebar.info(
            "Peer metadata reload detected. Restart the Streamlit app after rebuilding via CLI to pick up the new cache.",
            icon="ℹ️",
        )

    peer_section = st.container()
    selected_peers = []
    peer_defaults = []
    label = "N/A"

    if ticker_input:
        subject_info = fetch_subject_info(ticker_input)
        if not subject_info["sector"] and not subject_info["industry"]:
            peer_section.warning(
                "Unable to fetch sector/industry data from Yahoo Finance; peer suggestions may be empty."
            )
        suggestions = propose_peer_sets(ticker_input, subject_info, universe_info)
        peer_defaults = suggestions["same_sub"] or suggestions["global"]
        label = suggestions["label"]

        peer_options = sorted(set(peer_defaults + suggestions["global"]))
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
