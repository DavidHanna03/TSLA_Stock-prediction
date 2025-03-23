import numpy as np
import pandas as pd
import yfinance as yf
import random
import sys
from datetime import datetime, timedelta
import xgboost as xgb

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
        self.model = model  # This is an xgboost Booster
        self.capital = starting_capital
        self.shares = 0
        self.avg_buy_price = None

    def predict_next_close(self, feature_df):
        dmatrix = xgb.DMatrix(feature_df)
        return self.model.predict(dmatrix)[0]

# ---------------------------
# Single-Day Prediction Function
# ---------------------------
def single_day_prediction(agent, df, target_date_str, currently_holding=False, avg_buy_price=None):
    target_date = pd.to_datetime(target_date_str).normalize()
    df = df.sort_values("Date").reset_index(drop=True)
    df_by_date = df.set_index("Date")
    valid_dates = df_by_date.index[df_by_date.index < target_date]
    if len(valid_dates) == 0:
        sys.exit("No available trading day before the target date.")
    used_date = valid_dates[-1]
    prev_row = df_by_date.loc[used_date]

    feat_data = {
        "Feature1": [prev_row["Feature1"]],
        "Feature2": [prev_row["Feature2"]],
        "Feature3": [prev_row["Feature3"]],
        "SMA_10": [prev_row["SMA_10"]],
        "SMA_5": [prev_row["SMA_5"]],
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
        if avg_buy_price is not None:
            if (execution_price >= avg_buy_price * (1 + TAKE_PROFIT_THRESHOLD) or
                execution_price <= avg_buy_price * (1 - STOP_LOSS_THRESHOLD)):
                action = "SELL"
                fraction_percent = 10

    # Print only the single-day analysis
    print(f"\nSingle-Day Analysis for {target_date.date()}")
    print(f"Using PREVIOUS day data from {used_date.date()}")
    print(f"Predicted Close on {target_date_str}: ${predicted_close:.2f}")
    print(f"Estimated Execution Price: ${execution_price:.2f}")
    print(f"Yesterday's Close: ${last_price:.2f}")
    print(f"Predicted % Change: {predicted_change_pct:.2f}%\n")
    if action == "BUY":
        print("Recommended Action: BUY")
        print(f"Amount to invest: ~ {fraction_percent}0% of total funds.\n")
    elif action == "SELL":
        print("Recommended Action: SELL")
        print(f"Amount to sell: ~ {fraction_percent}0% of your shares.\n")
    else:
        print("Recommended Action: HOLD\n")

# ---------------------------
# Main Function
# ---------------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python Test_3.py YYYY-MM-DD [holding <avg_buy_price>]")
        sys.exit(1)

    # Use the first command-line argument as the target (and today's) date
    target_date_str = sys.argv[1]
    current_dt = datetime.strptime(target_date_str, "%Y-%m-%d")
    end_dt = current_dt + timedelta(days=1)
    end_date_str = end_dt.strftime("%Y-%m-%d")

    ticker = "TSLA"
    start_date = "2010-06-29"
    data = yf.download(ticker, start=start_date, end=end_date_str, interval="1d", auto_adjust=False)
    data.reset_index(inplace=True)
    data.to_csv("TSLA_Data.csv", index=False)

    try:
        df = pd.read_csv("TSLA_Data.csv", parse_dates=["Date"])
    except Exception as e:
        sys.exit("Error reading CSV: " + str(e))
    if "Date" not in df.columns:
        sys.exit("CSV missing 'Date' column.")
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.rename(columns={
        "Open": "Feature1",
        "High": "Feature2",
        "Low": "Feature3"
    }, inplace=True)
    df = df[df["Date"].dt.dayofweek < 5]

    # Compute technical indicators
    df["SMA_10"] = df["Close"].rolling(window=10).mean()
    df["SMA_5"] = df["Close"].rolling(window=5).mean()
    df["EMA_10"] = df["Close"].ewm(span=10, adjust=False).mean()
    df["RSI"] = 100 - (100 / (1 + df["Close"].pct_change().rolling(14).mean()))
    df["MACD"] = df["Close"].ewm(span=12, adjust=False).mean() - df["Close"].ewm(span=26, adjust=False).mean()
    df["ADX"] = df["Close"].diff().abs().rolling(14).mean() / df["Close"].rolling(14).mean()
    df["BB_MID"] = df["Close"].rolling(window=20).mean()
    df["BB_STD"] = df["Close"].rolling(window=20).std()
    df["BB_UPPER"] = df["BB_MID"] + 2 * df["BB_STD"]
    df["BB_LOWER"] = df["BB_MID"] - 2 * df["BB_STD"]
    df["TYPICAL_PRICE"] = (df["Feature1"] + df["Feature2"] + df["Close"]) / 3
    df["VWAP"] = (df["TYPICAL_PRICE"] * df["Volume"]).rolling(10).sum() / df["Volume"].rolling(10).sum()
    df.fillna(method="bfill", inplace=True)

    # Prepare training data
    df_train = df.copy()
    df_train["NextClose"] = df_train["Close"].shift(-1)
    df_train["NextClose"] = df_train["NextClose"].interpolate(method="linear")
    df_train["NextClose"].fillna(method="ffill", inplace=True)
    feature_cols = [
        "Feature1", "Feature2", "Feature3",
        "SMA_10", "SMA_5", "EMA_10",
        "RSI", "MACD", "ADX",
        "BB_UPPER", "BB_LOWER", "VWAP"
    ]
    df_features = df_train[feature_cols]
    df_target = df_train["NextClose"]

    # Train/Validation Split
    train_size = int(len(df_features) * 0.8)
    X_train = df_features.iloc[:train_size]
    y_train = df_target.iloc[:train_size]
    X_val = df_features.iloc[train_size:]
    y_val = df_target.iloc[train_size:]
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    params = {
        'max_depth': 8,
        'learning_rate': 0.01,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'reg:squarederror',
        'seed': 42
    }
    evals = [(dtrain, 'train'), (dval, 'eval')]
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=evals,
        early_stopping_rounds=10,
        verbose_eval=False
    )

    agent = TradingAgent(model=booster, starting_capital=STARTING_CAPITAL)
    if len(sys.argv) == 2:
        single_day_prediction(agent, df_train, target_date_str, currently_holding=False, avg_buy_price=None)
    elif len(sys.argv) == 4:
        hold_str = sys.argv[2]
        avg_buy = float(sys.argv[3])
        if hold_str.lower() == "holding":
            single_day_prediction(agent, df_train, target_date_str, currently_holding=True, avg_buy_price=avg_buy)
    else:
        print("Usage: python Test_3.py YYYY-MM-DD [holding <avg_buy_price>]")

if __name__ == "__main__":
    main()
