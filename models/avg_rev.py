import pandas as pd
import matplotlib.pyplot as plt

# 1) Load & parse dates
df = pd.read_csv('data/movie_dataset.csv')
df['release_date'] = pd.to_datetime(
    df['release_date'].astype(str).str.strip(),
    errors='coerce'
)
df['release_year'] = df['release_date'].dt.year

# 2) Flag recession vs non-recession
periods = {
    '1958-1962': (1958, 1962),
    '1973-1976': (1973, 1976),
    '1980-1983': (1980, 1983),
    '1990-1992': (1990, 1992),
    '2000-2002': (2000, 2002),
    '2007-2010': (2007, 2010),
}
def assign_period(y):
    if pd.isna(y): return None
    for label, (start, end) in periods.items():
        if start <= int(y) <= end:
            return label
    return None

df['recession_period'] = df['release_year'].apply(assign_period)
df['is_recession'] = df['recession_period'].notna()

# 3) Explode genres
df['genre_list'] = df['genres'].str.split()
exploded = df.explode('genre_list').dropna(subset=['genre_list'])

# 4) For each (is_recession, genre) compute:
#    - total revenue
#    - movie count (unique IDs)
#    - average revenue = total_rev / count
agg = (
    exploded
    .groupby(['is_recession', 'genre_list'])
    .agg(
        total_revenue = ('revenue', 'sum'),
        movie_count   = ('id',      'nunique')
    )
    .reset_index()
)
agg['avg_revenue'] = agg['total_revenue'] / agg['movie_count']

# 5) Pivot so we have genres × [Non-Recession, Recession]
avg_rev_pivot = (
    agg
    .pivot(index='genre_list', columns='is_recession', values='avg_revenue')
    .fillna(0)
    .rename(columns={False: 'Non-Recession', True: 'Recession'})
)

# 6) Plot average revenue per movie
ax = avg_rev_pivot.plot(
    kind='bar',
    figsize=(12, 8),
    width=0.8
)
ax.set_ylabel('Average Revenue per Movie')
ax.set_title('Average Genre Revenue: Recession vs Non-Recession')
ax.legend(title='')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
