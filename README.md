# ML Trading Agent for Tesla Stocks

This repository contains code for a machine learning–based trading agent that predicts Tesla stock price movements and makes simulated trading decisions. It uses historical data from Yahoo Finance, computes technical indicators, trains an XGBoost regression model, and then applies decision logic to buy, sell, or hold based on predicted percentage changes.

---

## Files Overview

### Test_3.py

- **Purpose:**  
  Provides a single-day prediction tool for daily trading decisions. This script is designed to help you fill out a daily Google Form with a recommended action—whether to buy, sell, or hold—based on the most recent trading day’s data.

- **How It Works:**  
  - Downloads historical Tesla stock data using the `yfinance` library.
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
  ```
   #### Holding Shares  
  Run:
  ```bash
  python3 Test_3.py YYYY-MM-DD holding <avg_buy_price>
  ```
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


  ---
  ## Requirements:
  - **Before Running the scripts must install
    Run:
      ```bash
      pip install yfinance xgboost pandas matplotlib openpyxl
    ```
      
   ---
  ## Setup & Execution Steps:
  1. Clone or download this repository.
  2. Make sure Python 3 is installed on your system.
  3. Install dependencies using the command above.
  4. Run the simulation:
       - To simulate a full 5-day period (Test_1.py):
           Run:
         ```bash
         python3 Test_1.py
         ```
      - To get a daily trading recommendation (Test_3.py):
          -If the user is not holding shares
        Run:
        ```bash
        python3 Test_3.py 2025-03-24
        ```
     - If the user is holding shares at a certain average buy price (ex $192.45)
       Run:
       ```bash
       python3 Test_3.py 2025-03-24 holding 192.45
       ```
       ---
       ## Data & Indicators
       - ** Stock Data Source:
           - Tesla stock data is pulled from Yahoo Finance using the yfinance API.
       - ** Technical Indicators Used:
           - SMA (Simple Moving Average)
           - EMA (Exponential Moving Average)
           - RSI (Relative Strength Index)
           - MACD (Moving Average Convergence Divergence)
           - ADX (Average Directional Index)
           - Bollinger Bands (Upper, Lower)
           - VWAP (Volume Weighted Average Price)
---
##Output Summary 
- ** Test_3.py:
    - Prints single-day trading recommendations.
    - States predicted closing price, estimated execution price, and expected % change.
    - Recommends whether to BUY, SELL, or HOLD, based on your holding status and prediction confidence.
- ** Test_1.py:
    - Logs each trading decision (Buy/Sell/Hold).
    - Tracks account balance and shares.
    - Produces: daily_orders.xlsx with logs and performance report. & Balance and price prediction plots via matplotlib. 
--- 
##Notes 
1. to simulate the provided period March 24-28, 2025 is done by using Test_1.py.
2. to predict a decision (Buy, Sell, Hold) for only one day based on the last-day closing price we use Test_3.py (and this is repeated every day to fill the Google forms that are required)
3. There is also a report included in the repository that explains the model used and the reasons why I used it.


       
                
  
