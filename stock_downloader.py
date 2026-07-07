print("Program started")

import yfinance as yf
import ta

print("packages installed successfully")

# User input
symbol = input("Enter company symbol (example: COALINDIA.NS): ")
start_date = input("Enter start date (YYYY-MM-DD): ")
end_date = input("Enter end date (YYYY-MM-DD): ")

print("\nFetching company details...\n")

# Get company info
ticker = yf.Ticker(symbol)

try:
    company_name = ticker.info.get("longName", "Unknown Company")
except:
    company_name = "Unknown Company"

print(f"Company Name = {company_name}")

print("\nDownloading historical data...\n")

# Download data
data = yf.download(symbol, start=start_date, end=end_date, progress=False)

# Check data
if data.empty:
    print("❌ No data found. Check symbol or date range.")
else:
    # Reset index so Date becomes a column
    data.reset_index(inplace=True)

    # Remove multi-level columns (IMPORTANT FIX)
    data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]

    # Keep only required columns
    data = data[["Date", "Open", "High", "Low", "Close", "Volume"]]

    print("\nDataset Information:\n")
    print(data.info())

    print("\nMissing Values:\n")
    print(data.isnull().sum())

    print("\nStatistical Summary:\n")
    print(data.describe())

    print("\n✅ Clean Data:\n")
    print(data)

   # Remove duplicate rows
    data.drop_duplicates(inplace=True)

   # Fill missing values if any
    data = data.ffill()

    print("\nDataset Shape:")
    print(data.shape)

   # Calculate SMA
    data['SMA20']=data['Close'].rolling(20).mean()

    data['SMA50']=data['Close'].rolling(50).mean()

# Calculate EMA
    data['EMA20']=data['Close'].ewm(span=20).mean()

    data['EMA50']=data['Close'].ewm(span=50).mean()

# Calculate RSI
    from ta.momentum import RSIIndicator

    rsi=RSIIndicator(close=data['Close'])

    data['RSI']=rsi.rsi()

# Calculate MACD
    from ta.trend import MACD

    macd=MACD(data['Close'])

    data['MACD']=macd.macd()

# Calculate Bollinger Bands
    from ta.volatility import BollingerBands

    bb=BollingerBands(data['Close'])

    data['Upper']=bb.bollinger_hband()

    data['Lower']=bb.bollinger_lband()

    import matplotlib.pyplot as plt

# Plotting closing price
    plt.figure(figsize=(12,6))
    plt.plot(data['Date'], data['Close'])
    plt.title("Closing Price Trend")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid(True)
    plt.savefig("closing_price.png")
    plt.show()
# Plotting SMA20 and SMA50
    plt.figure(figsize=(12,6))
    plt.plot(data['Date'], data['Close'], label='Close')
    plt.plot(data['Date'], data['SMA20'], label='SMA20')
    plt.plot(data['Date'], data['SMA50'], label='SMA50')

    plt.legend()
    plt.title("Moving Averages")
    plt.grid(True)
    plt.savefig("sma_plot.png")

    plt.show()
# Plotting EMA20 and EMA50
    plt.figure(figsize=(12,6))

    plt.plot(data['Date'], data['Close'], label='Close')
    plt.plot(data['Date'], data['EMA20'], label='EMA20')
    plt.plot(data['Date'], data['EMA50'], label='EMA50')

    plt.title("Exponential Moving Averages")
    plt.legend()
    plt.grid(True)

    plt.show()

 # Plotting RSI
    plt.figure(figsize=(12,4))

    plt.plot(data['Date'], data['RSI'])

    plt.axhline(70, linestyle='--')
    plt.axhline(30, linestyle='--')

    plt.title("RSI Indicator")
    plt.savefig("rsi_plot.png")

    plt.show()

#verify
    print("\nIndicators Check:\n")

    print(
data[['Close','SMA20','SMA50','EMA20','EMA50','RSI','MACD','Upper','Lower']].tail()
)

# Save CSV
    filename = symbol.replace(".", "_") + "_clean.csv"
    data.to_csv(filename, index=False)

    print(f"\n✅ CSV saved as {filename}")