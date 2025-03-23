import numpy as np
import pandas as pd
import yfinance as yf
import random
import sys
from datetime import datetime, timedelta
from xgboost import XGBRegressor

# ---------------------------
# Global Constants
# ---------------------------
STARTING_CAPITAL = 10000
TRANSACTION_FEE = 0.01
RISK_FREE_RATE = 0
TAKE_PROFIT_THRESHOLD = 0.03
STOP_LOSS_THRESHOLD = 0.02
MAX_BUY_PERCENT = 0.8
MIN_BUY_PERCENT = 0.2

# ---------------------------
# Trading Agent Class
# ---------------------------
class TradingAgent:
    def __init__(self, model, starting_capital=STARTING_CAPITAL):
        self.model = model
        self.capital = starting_capital
        self.shares = 0
        self.avg_buy_price = None

    def predict_next_close(self, feature_df):
        return self.model.predict(feature_df)[0]

# ---------------------------
# Single-Day Prediction Function
# ---------------------------
def single_day_prediction(agent, df, target_date_str, currently_holding=False, avg_buy_price=None):
    # Convert target_date_str to a normalized datetime (time=00:00)
    target_date = pd.to_datetime(target_date_str).normalize()

    # Sort data by Date and set "Date" as index
    df = df.sort_values("Date").reset_index(drop=True)
    df_by_date = df.set_index("Date")

    # Select all trading days strictly before target_date
    valid_dates = df_by_date.index[df_by_date.index < target_date]
    if len(valid_dates) == 0:
        sys.exit("No available trading day before the target date.")

    # The last available trading day is the "previous day"
    used_date = valid_dates[-1]
    prev_row = df_by_date.loc[used_date]

    # Build a 1-row feature DataFrame from the previous day's data
    feat_data = {
        "Feature1": [prev_row["Feature1"]],
        "Feature2": [prev_row["Feature2"]],
        "Feature3": [prev_row["Feature3"]],
        "SMA_10": [prev_row["SMA_10"]],
        "EMA_10": [prev_row["EMA_10"]],
        "RSI": [prev_row["RSI"]],
        "MACD": [prev_row["MACD"]],
        "ADX": [prev_row["ADX"]],
        "BB_UPPER": [prev_row["BB_UPPER"]],
        "BB_LOWER": [prev_row["BB_LOWER"]],
        "VWAP": [prev_row["VWAP"]],
    }
    feat_df = pd.DataFrame(feat_data)

    # Predict next day's closing price
    predicted_close = agent.predict_next_close(feat_df)

    # Simulate a slight variance in execution price
    execution_price = predicted_close * (1 + random.uniform(-0.005, 0.015))

    last_price = prev_row["Close"]
    predicted_change_pct = ((predicted_close - last_price) / last_price) * 100

    action = "HOLD"
    fraction_percent = 0
    if not currently_holding:
        # Decide whether to BUY
        if predicted_change_pct > 0:
            fraction = np.clip((predicted_change_pct / 100) * 10, MIN_BUY_PERCENT, MAX_BUY_PERCENT)
            fraction_percent = int(round(fraction * 10))
            action = "BUY"
    else:
        # Already holding, decide whether to SELL
        if avg_buy_price is not None:
            if (execution_price >= avg_buy_price * (1 + TAKE_PROFIT_THRESHOLD) or
                execution_price <= avg_buy_price * (1 - STOP_LOSS_THRESHOLD)):
                action = "SELL"
                fraction_percent = 10

    # Print the single-day analysis
    print("\nSingle-Day Analysis for", target_date.date())
    print("Using PREVIOUS day data from", used_date.date())
    print(f"Predicted Close on {target_date_str}: ${predicted_close:.2f}")
    print(f"Estimated Execution Price: ${execution_price:.2f}")
    print(f"Yesterday's Close: ${last_price:.2f}")
    print(f"Predicted % Change: {predicted_change_pct:.2f}%")

    if action == "BUY":
        print("\nRecommended Action: BUY")
        print(f"Amount to invest: ~ {fraction_percent}0% of total funds.\n")
    elif action == "SELL":
        print("\nRecommended Action: SELL")
        print(f"Amount to sell: ~ {fraction_percent}0% of your shares.\n")
    else:
        print("\nRecommended Action: HOLD\n")

# ---------------------------
# Main Function
# ---------------------------
def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python Test_3.py YYYY-MM-DD")
        print("  python Test_3.py YYYY-MM-DD holding <avg_buy_price>")
        sys.exit(1)

    # Prompt for today's date, but add +1 day so yfinance includes that day
    current_date_str = input("Enter today's date (YYYY-MM-DD): ")
    current_dt = datetime.strptime(current_date_str, "%Y-%m-%d")
    end_dt = current_dt + timedelta(days=1)
    end_date_str = end_dt.strftime("%Y-%m-%d")

    ticker = "TSLA"
    start_date = "2010-06-29"  # Tesla's IPO date

    # Fetch TSLA data from Yahoo Finance (no date shift hack)
    print(f"\nFetching {ticker} data from {start_date} to {end_date_str}...")
    data = yf.download(ticker, start=start_date, end=end_date_str, interval="1d", auto_adjust=False)
    data.reset_index(inplace=True)

    # Save the raw data to CSV
    data.to_csv("TSLA_Data.csv", index=False)
    print("Data saved to TSLA_Data.csv")

    # Load data from CSV
    try:
        df = pd.read_csv("TSLA_Data.csv", parse_dates=["Date"])
    except Exception as e:
        sys.exit("Error reading CSV: " + str(e))

    # Verify we have a 'Date' column
    if "Date" not in df.columns:
        sys.exit("CSV missing 'Date' column.")

    # Convert key columns to numeric
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Rename columns for model
    df.rename(columns={
        "Open": "Feature1",
        "High": "Feature2",
        "Low": "Feature3"
    }, inplace=True)

    # Keep only weekdays
    df = df[df["Date"].dt.dayofweek < 5]

    # Compute technical indicators
    df["SMA_10"] = df["Close"].rolling(window=10).mean()
    df["EMA_10"] = df["Close"].ewm(span=10, adjust=False).mean()
    df["RSI"] = 100 - (100 / (1 + df["Close"].pct_change().rolling(14).mean()))
    df["MACD"] = df["Close"].ewm(span=12, adjust=False).mean() - df["Close"].ewm(span=26, adjust=False).mean()
    df["ADX"] = df["Close"].diff().abs().rolling(14).mean() / df["Close"].rolling(14).mean()
    df["BB_MID"] = df["Close"].rolling(20).mean()
    df["BB_STD"] = df["Close"].rolling(20).std()
    df["BB_UPPER"] = df["BB_MID"] + 2 * df["BB_STD"]
    df["BB_LOWER"] = df["BB_MID"] - 2 * df["BB_STD"]
    df["TYPICAL_PRICE"] = (df["Feature1"] + df["Feature2"] + df["Close"]) / 3
    df["VWAP"] = (df["TYPICAL_PRICE"] * df["Volume"]).rolling(10).sum() / df["Volume"].rolling(10).sum()
    df.fillna(method="bfill", inplace=True)

    # Prepare training data (y=NextClose), using linear interpolation for the last row
    df_train = df.copy()
    df_train["NextClose"] = df_train["Close"].shift(-1)

    # Interpolate any missing NextClose values (the last day) and forward-fill any remaining
    df_train["NextClose"] = df_train["NextClose"].interpolate(method="linear")
    df_train["NextClose"].fillna(method="ffill", inplace=True)

    feature_cols = ["Feature1", "Feature2", "Feature3", "SMA_10", "EMA_10",
                    "RSI", "MACD", "ADX", "BB_UPPER", "BB_LOWER", "VWAP"]
    df_features = df_train[feature_cols]
    df_target = df_train["NextClose"]

    # Train XGBoost model
    model = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42)
    model.fit(df_features, df_target)

    agent = TradingAgent(model=model, starting_capital=STARTING_CAPITAL)

    # Single-day prediction
    target_date_str = sys.argv[1]
    if len(sys.argv) == 2:
        single_day_prediction(agent, df_train, target_date_str, currently_holding=False, avg_buy_price=None)
    elif len(sys.argv) == 4:
        hold_str = sys.argv[2]
        avg_buy = float(sys.argv[3])
        if hold_str.lower() == "holding":
            single_day_prediction(agent, df_train, target_date_str, currently_holding=True, avg_buy_price=avg_buy)
    else:
        print("Usage:")
        print("  python Test_3.py YYYY-MM-DD")
        print("  python Test_3.py YYYY-MM-DD holding <avg_buy_price>")

if __name__ == "__main__":
    main()
