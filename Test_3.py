import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import random
import sys
from datetime import datetime, timedelta
from xgboost import XGBRegressor
from pandas.tseries.offsets import BDay

# Constants
STARTING_CAPITAL = 10000
TRANSACTION_FEE = 0.01
RISK_FREE_RATE = 0
TAKE_PROFIT_THRESHOLD = 0.03
STOP_LOSS_THRESHOLD = 0.02
MAX_BUY_PERCENT = 0.8
MIN_BUY_PERCENT = 0.2

class TradingAgent:
    def __init__(self, model, starting_capital=STARTING_CAPITAL):
        self.model = model
        self.capital = starting_capital
        self.shares = 0
        self.avg_buy_price = None

    def predict_next_close(self, feature_df):
        return self.model.predict(feature_df)[0]

def single_day_prediction(agent, df, target_date_str, currently_holding=False, avg_buy_price=None):
    tz = 'America/New_York'
    # Process target_date: localize if needed and normalize (set to midnight)
    target_date = pd.to_datetime(target_date_str)
    if target_date.tzinfo is None:
        target_date = target_date.tz_localize(tz)
    else:
        target_date = target_date.tz_convert(tz)
    target_date = target_date.normalize()

    # Set the DataFrame index and ensure it is timezone-aware and normalized
    df_by_date = df.set_index("Date")
    df_by_date.index = pd.to_datetime(df_by_date.index)
    if df_by_date.index.tz is None:
        df_by_date.index = df_by_date.index.tz_localize(tz)
    else:
        df_by_date.index = df_by_date.index.tz_convert(tz)
    df_by_date.index = df_by_date.index.normalize()

    # Select all trading days strictly before the target date
    valid_dates = df_by_date.index[df_by_date.index < target_date].sort_values()
    if len(valid_dates) == 0:
        sys.exit("No available trading day before the target date.")

    # Use the last available trading day
    used_date = valid_dates[-1]
    prev_row = df_by_date.loc[used_date]

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
    predicted_close = agent.predict_next_close(feat_df)
    execution_price = predicted_close * (1 + random.uniform(-0.005, 0.015))
    last_price = prev_row["Close"]
    predicted_change_pct = ((predicted_close - last_price) / last_price) * 100

    action = "HOLD"
    fraction_percent = 0
    if not currently_holding:
        if predicted_change_pct > 0:
            fraction = np.clip((predicted_change_pct / 100) * 10, MIN_BUY_PERCENT, MAX_BUY_PERCENT)
            fraction_percent = int(round(fraction * 10))
            action = "BUY"
        else:
            action = "HOLD"
    else:
        if avg_buy_price is None:
            action = "HOLD"
        else:
            if execution_price >= avg_buy_price * (1 + TAKE_PROFIT_THRESHOLD):
                action = "SELL"
                fraction_percent = 10
            elif execution_price <= avg_buy_price * (1 - STOP_LOSS_THRESHOLD):
                action = "SELL"
                fraction_percent = 10
            else:
                action = "HOLD"

    print("\nSingle-Day Analysis for", target_date.date())
    print("Using PREVIOUS day data from", used_date.date())
    print(f"Predicted Close on {target_date_str}: ${predicted_close:.2f}")
    print(f"Estimated Execution Price: ${execution_price:.2f}")
    print(f"Yesterday's Close: ${last_price:.2f}")
    print(f"Predicted % Change: {predicted_change_pct:.2f}%")
    if action == "BUY":
        print(f"\nRecommended Action: BUY")
        print(f"Amount to invest: ~ {fraction_percent}0% of total funds.\n")
    elif action == "SELL":
        print(f"\nRecommended Action: SELL")
        print(f"Amount to sell: ~ {fraction_percent}0% of your shares.\n")
    else:
        print("\nRecommended Action: HOLD\n")

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python Test_3.py YYYY-MM-DD")
        print("  python Test_3.py YYYY-MM-DD holding <avg_buy_price>")
        sys.exit(1)

    # Use the target date entered by the user and add one day to use as the exclusive end date.
    target_date_str = sys.argv[1]
    target_dt = pd.to_datetime(target_date_str)
    end_date_str = (target_dt + pd.Timedelta(days=1)).strftime('%Y-%m-%d')

    tsla = yf.Ticker("TSLA")
    try:
        # Using the adjusted end_date_str so that data includes trading days up to target_dt.
        tesla_df = tsla.history(start="2024-09-01", end=end_date_str, interval="1d")
        tesla_df.reset_index(inplace=True)
        tesla_df.rename(columns={
            "Date": "Date", 
            "Open": "Feature1", 
            "High": "Feature2", 
            "Low": "Feature3", 
            "Close": "Close", 
            "Volume": "Volume"
        }, inplace=True)
        tesla_df["Date"] = pd.to_datetime(tesla_df["Date"])
        # Keep only weekdays (trading days)
        tesla_df = tesla_df[tesla_df["Date"].dt.dayofweek < 5]
        tesla_df["SMA_10"] = tesla_df["Close"].rolling(window=10).mean()
        tesla_df["EMA_10"] = tesla_df["Close"].ewm(span=10, adjust=False).mean()
        tesla_df["RSI"] = 100 - (100 / (1 + tesla_df["Close"].pct_change().rolling(14).mean()))
        tesla_df["MACD"] = tesla_df["Close"].ewm(span=12, adjust=False).mean() - tesla_df["Close"].ewm(span=26, adjust=False).mean()
        tesla_df["ADX"] = tesla_df["Close"].diff().abs().rolling(14).mean() / tesla_df["Close"].rolling(14).mean()
        tesla_df["BB_MID"] = tesla_df["Close"].rolling(20).mean()
        tesla_df["BB_STD"] = tesla_df["Close"].rolling(20).std()
        tesla_df["BB_UPPER"] = tesla_df["BB_MID"] + 2 * tesla_df["BB_STD"]
        tesla_df["BB_LOWER"] = tesla_df["BB_MID"] - 2 * tesla_df["BB_STD"]
        tesla_df["TYPICAL_PRICE"] = (tesla_df["Feature1"] + tesla_df["Feature2"] + tesla_df["Close"]) / 3
        tesla_df["VWAP"] = (tesla_df["TYPICAL_PRICE"] * tesla_df["Volume"]).rolling(10).sum() / tesla_df["Volume"].rolling(10).sum()
        tesla_df.fillna(method="bfill", inplace=True)
    except Exception as e:
        sys.exit("Error fetching data: " + str(e))

    df_train = tesla_df.copy()
    df_train["NextClose"] = df_train["Close"].shift(-1)
    df_train.dropna(inplace=True)

    feature_cols = ["Feature1", "Feature2", "Feature3", "SMA_10", "EMA_10", "RSI", "MACD", "ADX", "BB_UPPER", "BB_LOWER", "VWAP"]
    df_features = df_train[feature_cols]
    df_target = df_train["NextClose"]

    model = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42)
    model.fit(df_features, df_target)
    agent = TradingAgent(model=model, starting_capital=STARTING_CAPITAL)

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
