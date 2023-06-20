import yfinance as yf

msft = yf.Ticker("MSFT")
print(msft)

# get stock info
print(msft.info)

# Dump data to CSV
yf.download(tickers="MSFT", start="2023-04-25", end="2023-06-19", period="max", interval="5m").to_csv('data/msft.csv')