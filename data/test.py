import pandas as pd

# Load the CSV file and parse dates
df = pd.read_csv(
    "data/movie_dataset.csv", 
    parse_dates=["release_date"],  # Automatically parse dates
    dayfirst=True  # Important for DD-MM-YYYY format
)

# Find the row with the oldest release year
youngest_movie = df.loc[df["release_date"].idxmax()]

# Display full details
print("Oldest movie in the dataset:")
print("-----------------------------------")
print(youngest_movie.to_string())