# ML Trading Agent for Tesla Stocks

This repository contains code for a machine learning–based trading agent that predicts Tesla stock price movements and makes simulated trading decisions. It uses historical data from Yahoo Finance, computes technical indicators, trains an XGBoost regression model, and then applies decision logic to buy, sell, or hold based on predicted percentage changes.

---

## Files Overview

### Test_1.py

- **Purpose:**  
  Runs a full multi-day simulation (March 24–28, 2025) of trading decisions. It tracks your portfolio starting with \$10,000, applies a 1% transaction fee, and logs daily decisions (Buy, Sell, or Hold).

- **How It Works:**  
  - Downloads historical Tesla stock data using the `yfinance` library.
  - Computes technical indicators:
    - SMA (Simple Moving Average)
    - EMA (Exponential Moving Average)
    - RSI (Relative Strength Index)
    - MACD (Moving Average Convergence Divergence)
    - ADX (Average Directional Index)
    - Bollinger Bands
    - VWAP (Volume Weighted Average Price)
  - Trains an XGBoost regressor to predict the next day’s closing price.
  - Simulates daily trading decisions using the following strategy:
    - **Buy** if a predicted increase is detected.
    - **Sell** if the take profit threshold is met (≥3%) or stop loss is triggered (≤−2%).
    - **Hold** if neither condition is met.
    - Automatically **sells all shares** on the final simulation day to close all positions.
  - Logs each decision and transaction.
  - Tracks daily account balance and generates a plot of balance vs predicted vs execution prices.

- **Outputs:**
  - Detailed transaction logs.
  - Performance report (final balance, total profit, Sharpe ratio, daily returns).
  - Plot showing balance vs predicted vs execution prices.
  - Excel file (`daily_orders.xlsx`) with:
    - `Transaction Log` sheet
    - `Performance Report` sheet

---

### Test_3.py

- **Purpose:**  
  Provides a single-day prediction tool for daily trading decisions. This script is designed to help you fill out a daily Google Form with a recommended action—whether to buy, sell, or hold—based on the most recent trading day’s data.

- **How It Works:**  
  - Downloads the same historical data and computes technical indicators as in `Test_1.py`.
  - Trains the same XGBoost model.
  - For a given target date (e.g., 2025-03-23), it finds the most recent trading day (e.g., 2025-03-20) and uses that day’s data as input.  
    This is because there is no trading done on weekends, which is why it defaults to the most recent valid trade day.
  - The XGBoost model predicts the target day’s closing price.
  - It calculates the percentage change from the previous day’s closing price.

- **Decision Logic:**
  - **Not Holding Shares:**
    - If the predicted percentage change is positive,
    - It recommends **BUY** and scales the investment amount based on the change (clamped between 20% and 80%).
  - **Holding Shares:**
    - If you run it with a `"holding"` flag and provide your average buy price,
    - It compares the estimated execution price to your buy price.
    - If the price is at least 3% above or 2% below your average buy price, it recommends **SELL** (100% of shares); otherwise, it recommends **HOLD**.

- This output is used to manually update your trading decisions in your daily form.

- **Usage Modes:**

  #### Not Holding Shares  
  Run:
  ```bash
  python3 Test_3.py YYYY-MM-DD
