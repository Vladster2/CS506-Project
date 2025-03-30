import yfinance as yf
import pandas as pd
from datetime import datetime
import os
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.ticker import FuncFormatter
"""
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
"""

def plot_movies():
    # Load data from CSV file
    movies = pd.read_csv("data/output.csv", parse_dates=["release_date"], dayfirst=True)

    # Extract the year from the release date and add it as a new column 'Year'
    movies["Year"] = movies["release_date"].dt.year  # Assuming 'release_date' column exists

    # Group by 'Year' and sum the revenue for each year
    aggregated_revenue = movies.groupby("Year")["revenue"].sum().reset_index()

    # Sort by Year
    aggregated_revenue = aggregated_revenue.sort_values("Year")

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(aggregated_revenue["Year"], aggregated_revenue["revenue"], marker='o', linestyle='-', color='b')
    plt.xlabel("Year")
    plt.ylabel("Revenue ($)")

    # Format the y-axis labels to be whole numbers
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x):,}'))

    plt.title("Aggregated Movie Revenue Over Time")
    plt.grid()
    plt.show()

#print(plot_movies())

def all_movies_except_horror():
    file_path = "data/movie_dataset.csv"  
    movies = pd.read_csv(file_path) 
    filtered_movies = movies[~movies['genres'].str.contains("Horror", case=False, na=False)]
    filtered_movies.to_csv("data/movies_no_horror.csv", index=False)

all_movies_except_horror()


