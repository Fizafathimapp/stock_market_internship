print("Program started")

import yfinance as yf

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

    # Save CSV
filename = symbol.replace(".", "_") + "_clean.csv"
data.to_csv(filename, index=False)

print(f"\n✅ CSV saved as {filename}")