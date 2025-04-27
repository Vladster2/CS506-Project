import pandas as pd
import matplotlib.pyplot as plt

# 1. Load & parse
df = pd.read_csv('data/movie_dataset.csv')
df['release_date'] = pd.to_datetime(
    df['release_date'].astype(str).str.strip(),
    errors='coerce'
)
df['release_year'] = df['release_date'].dt.year

# 2. Define your recession buckets & flag
periods = {
    '1958-1962': (1958, 1962),
    '1973-1976': (1973, 1976),
    '1980-1983': (1980, 1983),
    '1990-1992': (1990, 1992),
    '2000-2002': (2000, 2002),
    '2007-2010': (2007, 2010),
}
def assign_period(y):
    if pd.isna(y):
        return None
    for label, (start, end) in periods.items():
        if start <= int(y) <= end:
            return label
    return None

df['recession_period'] = df['release_year'].apply(assign_period)
df['is_recession'] = df['recession_period'].notna()

# 3. Count total movies per flag
total = (
    df
    .groupby('is_recession')['id']
    .nunique()
    .reset_index(name='total_movies')
)

# 4. Explode genres so each row has one genre tag
df['genre_list'] = df['genres'].str.split()
exploded = df.explode('genre_list').dropna(subset=['genre_list'])

# 5. Count (unique) movies per genre × recession flag
genre_counts = (
    exploded
    .groupby(['is_recession', 'genre_list'])['id']
    .nunique()
    .reset_index(name='genre_movies')
)

# 6. Merge totals & compute frequency
merged = genre_counts.merge(total, on='is_recession')
merged['frequency'] = merged['genre_movies'] / merged['total_movies']

# 7. Pivot to get a table: index=genre, columns=[False,True] → freq
freq_pivot = (
    merged
    .pivot(index='genre_list', columns='is_recession', values='frequency')
    .fillna(0)
    .rename(columns={False: 'Non-Recession', True: 'Recession'})
)

# 8. Plot
ax = freq_pivot.plot(
    kind='bar', 
    figsize=(12, 8), 
    width=0.8
)
ax.set_ylabel('Proportion of Movies')
ax.set_title('Genre Frequency: Recession vs Non-Recession Periods')
ax.legend(title='')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
