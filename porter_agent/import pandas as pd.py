import pandas as pd
import yfinance as yf
import time
from urllib.error import HTTPError

# Load your file
df = pd.read_csv('/Users/valerianarvaez/Desktop/porter_agent/universe.csv')  # change path if needed

# Make sure these columns exist
for col in ["Sector", "Industry", "MarketCap"]:
    if col not in df.columns:
        df[col] = None

bad_tickers = []

for idx, row in df.iterrows():
    ticker = str(row["Ticker"]).strip()
    if not ticker:
        continue

    print(f"Fetching: {ticker} ({idx+1}/{len(df)})")

    try:
        tk = yf.Ticker(ticker)

        # Newer yfinance versions sometimes prefer get_info()
        try:
            info = tk.get_info()
        except Exception:
            info = tk.info  # fallback

        if not info or info is None:
            raise ValueError("Empty info returned")

        df.at[idx, "Sector"]    = info.get("sector")
        df.at[idx, "Industry"]  = info.get("industry") or info.get("industryDisp")
        df.at[idx, "MarketCap"] = info.get("marketCap")

    except HTTPError as e:
        print(f"HTTP error for {ticker}: {e}")
        bad_tickers.append(ticker)
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        bad_tickers.append(ticker)

    # avoid rate limiting
    time.sleep(0.5)

print("\nTickers that failed to fetch:")
print(bad_tickers)

# Save result
output_path = "/Users/valerianarvaez/Desktop/porter_agent/universe.csv"
df.to_csv(output_path, index=False)
print(f"\nDone – saved to {output_path}")
