import pandas as pd

# Load the CSV file and parse dates
df = pd.read_csv(
    "data/output.csv", 
    parse_dates=["release_date"],  # Automatically parse dates
    dayfirst=True  # Important for DD-MM-YYYY format
)

# Find the row with the oldest release year
oldest_movie = df.loc[df["release_date"].idxmin()]

# Display full details
print("Oldest horror movie in the dataset:")
print("-----------------------------------")
print(oldest_movie.to_string())