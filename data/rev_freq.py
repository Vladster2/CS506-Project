import pandas as pd
import matplotlib.pyplot as plt

# 1. Load & parse dates
df = pd.read_csv('data/movie_dataset.csv')
df['release_date'] = pd.to_datetime(
    df['release_date'].astype(str).str.strip(),
    errors='coerce'
)
df['release_year'] = df['release_date'].dt.year

# 2. Flag recession vs non-recession
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

# 3. Compute total revenue in each flag
total_rev = (
    df
    .groupby('is_recession')['revenue']
    .sum()
    .reset_index(name='total_revenue')
)

# 4. Explode genres
df['genre_list'] = df['genres'].str.split()
exploded = df.explode('genre_list').dropna(subset=['genre_list'])

# 5. Sum revenue by genre × flag
genre_rev = (
    exploded
    .groupby(['is_recession','genre_list'])['revenue']
    .sum()
    .reset_index(name='genre_revenue')
)

# 6. Merge & compute revenue share
merged = genre_rev.merge(total_rev, on='is_recession')
merged['revenue_share'] = merged['genre_revenue'] / merged['total_revenue']

# 7. Pivot to get (genres × [Non-Recession, Recession])
share_pivot = (
    merged
    .pivot(index='genre_list', columns='is_recession', values='revenue_share')
    .fillna(0)
    .rename(columns={False: 'Non-Recession', True: 'Recession'})
)

# 8. Plot
ax = share_pivot.plot(
    kind='bar',
    figsize=(12, 8),
    width=0.8
)
ax.set_ylabel('Share of Total Revenue')
ax.set_title('Genre Revenue Share: Recession vs Non-Recession')
ax.legend(title='')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
