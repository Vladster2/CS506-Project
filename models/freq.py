import pandas as pd
import matplotlib.pyplot as plt

# 1. Load data
df = pd.read_csv("data/movies_adjusted.csv", parse_dates=["release_date"])

# 2. Explode genres into one-genre-per-row
df["genre"] = df["genres"].str.split()           # split on whitespace
df_exploded = df.explode("genre").reset_index(drop=True)

# 3. Define recession periods
recessions = {
    "1958-1962": ("1958-01-01", "1962-12-31"),
    "1973-1976": ("1973-01-01", "1976-12-31"),
    "1980-1983": ("1980-01-01", "1983-12-31"),
    "1990-1992": ("1990-01-01", "1992-12-31"),
    "2000-2002": ("2000-01-01", "2002-12-31"),
    "2007-2010": ("2007-01-01", "2010-12-31"),
    "2020-2022": ("2020-01-01", "2022-12-31"),
}

# helper to tag each movie with its recession period
def assign_recession_period(date):
    for label, (start, end) in recessions.items():
        if pd.Timestamp(start) <= date <= pd.Timestamp(end):
            return label
    return None

df_exploded["recession_period"] = df_exploded["release_date"].apply(assign_recession_period)

# 4. Filter to only those in a recession period
df_rec = df_exploded.dropna(subset=["recession_period"])

# 5. Count and pivot
counts = (
    df_rec
    .groupby(["recession_period", "genre"])
    .size()
    .unstack(fill_value=0)
    .loc[recessions.keys()]   # ensure chronological order
)

# 6. Plot
ax = counts.plot(
    kind="bar",
    stacked=True,
    figsize=(12, 8),
    width=0.8
)

ax.set_xlabel("Recession Period")
ax.set_ylabel("Number of Movies")
ax.set_title("Movie Counts by Genre During Recession Periods")
ax.legend(title="Genre", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()
