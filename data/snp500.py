import yfinance as yf
import pandas as pd
from datetime import datetime
import os

def download_sp500_data():
    print("Downloading ...")
    
    # ^GSPC  is for snp 500
    sp500 = yf.Ticker("^GSPC")
    data = sp500.history(period="max")
    
    data = data.reset_index()
    data['Date'] = data['Date'].dt.strftime('%m/%d/%Y')
    
    os.makedirs('data', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"data/SP500_Historical_Data_{timestamp}.csv"

    data.to_csv(filename, index=False)
    
    print(f"Data downloaded successfully and saved to {filename}")
    print(f"Data range: {data['Date'].iloc[-1]} to {data['Date'].iloc[0]}")
    print(f"Total records: {len(data)}")

if __name__ == "__main__":
    download_sp500_data()