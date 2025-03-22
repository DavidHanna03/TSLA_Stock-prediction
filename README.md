# ML Trading Agent for Tesla Stocks

This repository contains code for a machine learning–based trading agent that predicts Tesla stock price movements and makes simulated trading decisions. It uses historical data from Yahoo Finance, computes technical indicators, trains an XGBoost regression model, and then applies decision logic to buy, sell, or hold based on predicted percentage changes.

---

## Files Overview

### Test_1.py

- **Purpose:**  
  Runs a full multi-day simulation (March 24–28, 2025) of trading decisions. It tracks your portfolio starting with \$10,000, applies a 1% transaction fee, and logs daily decisions (Buy, Sell, or Hold).

- **How It Works:**  
  - Downloads historical Tesla stock data using the `yfinance` library.
  - Computes technical indicators (SMA, EMA, RSI, MACD, ADX, Bollinger Bands, VWAP).
  - Trains an XGBoost regressor to predict the next day’s closing price.
  - Simulates daily trading decisions based on thresholds (e.g., take profit at +3%, stop loss at -2%).
  - Logs transactions and outputs performance reports in an Excel file (`daily_orders.xlsx`), along with performance plots.

---

### Test_3.py

- **Purpose:**  
  Provides a single-day prediction tool intended for daily Google Form submissions. It uses the previous trading day’s data to predict the target day’s closing price and outputs a recommendation.

- **Usage Modes:**

  #### Not Holding Shares
  Run:
  ```bash
  python3 Test_3.py YYYY-MM-DD
