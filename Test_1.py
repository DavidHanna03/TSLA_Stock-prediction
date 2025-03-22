import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import random
from datetime import datetime

# Import XGBoost
from xgboost import XGBRegressor

# -----------------------------
# Simulation & Trading Settings
# -----------------------------
STARTING_CAPITAL = 10000
TRANSACTION_FEE = 0.01
ORDER_SUBMISSION_FILE = "daily_orders.xlsx"  # using .xlsx output
SIMULATION_START = "2025-03-24"
SIMULATION_END = "2025-03-28"
RISK_FREE_RATE = 0

# Dynamic thresholds
TAKE_PROFIT_THRESHOLD = 0.03  # 3% above avg buy triggers a sell
STOP_LOSS_THRESHOLD = 0.02    # 2% below avg buy triggers a sell
MAX_BUY_PERCENT = 0.8         # Max 80% of capital in a single buy
MIN_BUY_PERCENT = 0.2         # Min 20% of capital if we see a small bullish signal

try:
    tsla = yf.Ticker("TSLA")
    tesla_df = tsla.history(start="2024-09-01", end="2025-03-23", interval="1d")
    tesla_df.reset_index(inplace=True)
    tesla_df.rename(columns={
        "Date": "Date",
        "Open": "Feature1",
        "High": "Feature2",
        "Low": "Feature3",
        "Close": "Close",
        "Volume": "Volume"
    }, inplace=True)
    tesla_df['Date'] = pd.to_datetime(tesla_df['Date'])
    tesla_df = tesla_df[tesla_df['Date'].dt.dayofweek < 5]

    # Basic indicators
    tesla_df['SMA_10'] = tesla_df['Close'].rolling(window=10).mean()
    tesla_df['EMA_10'] = tesla_df['Close'].ewm(span=10, adjust=False).mean()
    tesla_df['RSI'] = 100 - (100 / (1 + tesla_df['Close'].pct_change().rolling(14).mean()))
    tesla_df['MACD'] = tesla_df['Close'].ewm(span=12, adjust=False).mean() - tesla_df['Close'].ewm(span=26, adjust=False).mean()
    tesla_df['ADX'] = tesla_df['Close'].diff().abs().rolling(14).mean() / tesla_df['Close'].rolling(14).mean()

    # Bollinger Bands (20-day)
    tesla_df['BB_MID'] = tesla_df['Close'].rolling(20).mean()
    tesla_df['BB_STD'] = tesla_df['Close'].rolling(20).std()
    tesla_df['BB_UPPER'] = tesla_df['BB_MID'] + 2 * tesla_df['BB_STD']
    tesla_df['BB_LOWER'] = tesla_df['BB_MID'] - 2 * tesla_df['BB_STD']

    # VWAP (approx. using rolling 10-day)
    tesla_df['TYPICAL_PRICE'] = (tesla_df['Feature1'] + tesla_df['Feature2'] + tesla_df['Close']) / 3
    tesla_df['VWAP'] = (tesla_df['TYPICAL_PRICE'] * tesla_df['Volume']).rolling(10).sum() / tesla_df['Volume'].rolling(10).sum()

    tesla_df.fillna(method='bfill', inplace=True)

except Exception as e:
    print(f"Error fetching data: {e}")
    exit()

df_train = tesla_df.copy()
df_train['NextClose'] = df_train['Close'].shift(-1)
df_train.dropna(inplace=True)

feature_cols = [
    'Feature1', 'Feature2', 'Feature3',
    'SMA_10', 'EMA_10', 'RSI', 'MACD', 'ADX',
    'BB_UPPER', 'BB_LOWER', 'VWAP'
]
df_features = df_train[feature_cols]
df_target = df_train['NextClose']

def calculate_performance_metrics(dates, balances):
    daily_returns = []
    for i in range(1, len(balances)):
        r = (balances[i] - balances[i-1]) / balances[i-1]
        daily_returns.append(r)
    avg_return = np.mean(daily_returns) if daily_returns else 0
    std_return = np.std(daily_returns) if daily_returns else 0
    sharpe_ratio = (avg_return - RISK_FREE_RATE) / std_return if std_return != 0 else 0
    final_balance = balances[-1] if balances else STARTING_CAPITAL
    total_profit = final_balance - STARTING_CAPITAL
    daily_str = []
    for i in range(1, len(dates)):
        day_str = dates[i].strftime('%b %d, %Y')
        daily_str.append(f"{day_str}: {daily_returns[i-1]*100:.2f}%")
    daily_report = "\n".join(daily_str)
    return (
        f"Final Balance: ${final_balance:,.2f}\n"
        f"Total Profit: ${total_profit:,.2f}\n"
        f"Sharpe Ratio: {sharpe_ratio:.2f}\n"
        "Daily Returns:\n" + daily_report
    )

class TradingAgent:
    def __init__(self, model, starting_capital=STARTING_CAPITAL):
        self.model = model
        self.capital = starting_capital
        self.shares = 0
        self.avg_buy_price = None
        self.order_log = []
        self.balance_history = []
        self.dates = []
        self.predicted_prices = []
        self.actual_prices = []
    
    def predict_next_close(self, feature_df):
        return self.model.predict(feature_df)[0]
    
    def buy_shares(self, amount_usd, price, time_str):
        fee = amount_usd * TRANSACTION_FEE
        total_cost = amount_usd + fee
        if total_cost > self.capital:
            amount_usd = self.capital / (1 + TRANSACTION_FEE)
            fee = amount_usd * TRANSACTION_FEE
            total_cost = amount_usd + fee
        shares_bought = amount_usd / price
        self.capital -= total_cost
        if self.shares == 0:
            self.avg_buy_price = price
        else:
            old_shares = self.shares
            self.avg_buy_price = ((self.avg_buy_price * old_shares) + (price * shares_bought)) / (old_shares + shares_bought)
        self.shares += shares_bought
        msg = f"{time_str} - Buy: ${amount_usd:.2f} at ${price:.2f}, Shares: {shares_bought:.4f}, Fee: ${fee:.2f}"
        self.log_order(msg)

    def sell_shares(self, num_shares, price, time_str):
        if num_shares > self.shares:
            num_shares = self.shares
        revenue = num_shares * price
        fee = revenue * TRANSACTION_FEE
        revenue_after_fee = revenue - fee
        self.capital += revenue_after_fee
        self.shares -= num_shares
        msg = f"{time_str} - Sell: {num_shares:.4f} shares at ${price:.2f}, Fee: ${fee:.2f}"
        self.log_order(msg)
        if self.shares == 0:
            self.avg_buy_price = None

    def log_order(self, msg):
        self.order_log.append(msg)
        print(msg)

    def save_orders(self, report):
        # Create a DataFrame for the transaction log
        df_log = pd.DataFrame({"Transaction Log": self.order_log})
        # Determine the ending balance (last recorded balance)
        ending_balance = self.balance_history[-1] if self.balance_history else STARTING_CAPITAL
        # Create a summary string to print and include in the performance report
        summary = (
            f"Starting Capital: ${STARTING_CAPITAL:,.2f}\n"
            f"Ending Capital: ${ending_balance:,.2f}\n\n"
            f"{report}"
        )
        print(summary)
        # Create a DataFrame for the performance report
        df_report = pd.DataFrame({
            "Metric": ["Starting Capital", "Ending Capital", "Performance Report"],
            "Value": [f"${STARTING_CAPITAL:,.2f}", f"${ending_balance:,.2f}", report]
        })
        # Save to an Excel file with two sheets using the xlsxwriter engine
        with pd.ExcelWriter(ORDER_SUBMISSION_FILE, engine='xlsxwriter') as writer:
            df_log.to_excel(writer, sheet_name="Transaction Log", index=False)
            df_report.to_excel(writer, sheet_name="Performance Report", index=False)
        print(f"Orders saved to {ORDER_SUBMISSION_FILE}")

    def track_balance(self, date, price):
        current_value = self.capital + self.shares * price
        self.balance_history.append(current_value)
        self.dates.append(date)

def plot_performance(dates, balances, preds, actuals):
    plt.figure(figsize=(10, 5))
    plt.plot(dates, balances, label="Account Balance", marker='o')
    plt.plot(dates, preds, label="Predicted Prices", marker='x', linestyle='--')
    plt.plot(dates, actuals, label="Execution Prices", marker='s', linestyle=':')
    plt.title("Agent Performance: Balance vs. Predicted & Execution Prices")
    plt.xlabel("Date")
    plt.ylabel("USD / Price")
    plt.grid(True)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %d, %Y'))
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

def simulate_trading(agent, df):
    sim_dates = pd.date_range(SIMULATION_START, SIMULATION_END, freq='B')
    df_by_date = df.set_index('Date')
    last_price = None
    
    for i, current_date in enumerate(sim_dates):
        sub_time = f"{current_date.date()} 09:00 AM EST"
        print(f"{sub_time}: Submitting order decision.")
        if current_date in df_by_date.index:
            row = df_by_date.loc[current_date]
        else:
            row = df_by_date.iloc[-1]
        if last_price is None:
            last_price = row['Close']
        feat_data = {
            "Feature1": [row["Feature1"]],
            "Feature2": [row["Feature2"]],
            "Feature3": [row["Feature3"]],
            "SMA_10": [row["SMA_10"]],
            "EMA_10": [row["EMA_10"]],
            "RSI": [row["RSI"]],
            "MACD": [row["MACD"]],
            "ADX": [row["ADX"]],
            "BB_UPPER": [row["BB_UPPER"]],
            "BB_LOWER": [row["BB_LOWER"]],
            "VWAP": [row["VWAP"]]
        }
        feat_df = pd.DataFrame(feat_data)
        pred_price = agent.predict_next_close(feat_df)
        agent.predicted_prices.append(pred_price)
        # Slight random variation with mild upward bias
        execution_price = pred_price * (1 + random.uniform(-0.005, 0.015))
        agent.actual_prices.append(execution_price)
        print(f"{current_date.date()} 10:00 AM EST: Execution price: ${execution_price:.2f}, Predicted: ${pred_price:.2f}")
        
        if agent.shares == 0:
            # Confidence-based approach: how big is the predicted change?
            predicted_change = (pred_price - last_price) / last_price
            if predicted_change > 0:
                fraction = np.clip(predicted_change * 10, MIN_BUY_PERCENT, MAX_BUY_PERCENT)
                amt = agent.capital * fraction
                agent.buy_shares(amt, execution_price, sub_time)
            else:
                agent.log_order(f"{sub_time} - Hold: No transaction")
        else:
            if execution_price >= agent.avg_buy_price * (1 + TAKE_PROFIT_THRESHOLD):
                agent.log_order(f"{sub_time} - Take Profit Triggered")
                agent.sell_shares(agent.shares, execution_price, sub_time)
            elif execution_price <= agent.avg_buy_price * (1 - STOP_LOSS_THRESHOLD):
                agent.log_order(f"{sub_time} - Stop Loss Triggered")
                agent.sell_shares(agent.shares, execution_price, sub_time)
            else:
                agent.log_order(f"{sub_time} - Hold: No transaction")

        # Final day: Force sell any remaining shares at the same 9:00 AM submission time.
        if i == len(sim_dates) - 1 and agent.shares > 0:
            agent.log_order(f"{sub_time} - Final Day: Closing position")
            agent.sell_shares(agent.shares, execution_price, sub_time)
        
        last_price = execution_price
        agent.track_balance(current_date, last_price)
    
    report = calculate_performance_metrics(agent.dates, agent.balance_history)
    agent.save_orders(report)
    plot_performance(agent.dates, agent.balance_history, agent.predicted_prices, agent.actual_prices)

if __name__ == "__main__":
    model = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42)
    model.fit(df_features, df_target)
    agent = TradingAgent(model=model, starting_capital=STARTING_CAPITAL)
    simulate_trading(agent, df_train)
