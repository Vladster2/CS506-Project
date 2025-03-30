import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

sp500 = pd.read_csv('sp500.csv', parse_dates=['Date'])
sp500 = sp500[['Date', 'Close']]
sp500['Month'] = sp500['Date'].dt.to_period('M')

monthly_sp500 = sp500.groupby('Month')['Close'].mean().reset_index()
monthly_sp500['pct_change'] = monthly_sp500['Close'].pct_change() * 100
monthly_sp500 = monthly_sp500.rename(columns={'Close': 'avg_sp500_price'})

movies = pd.read_csv('movies.csv', parse_dates=['release_date'], dayfirst=True)
movies['Month'] = movies['release_date'].dt.to_period('M')
movies['roi'] = movies['revenue'] - movies['budget']
monthly_roi = movies.groupby('Month')['roi'].mean().reset_index()

monthly_df = pd.merge(monthly_roi, monthly_sp500[['Month', 'pct_change']], on='Month', how='inner')
monthly_df['Month'] = monthly_df['Month'].astype(str)
monthly_df = monthly_df.dropna()

X = monthly_df[['pct_change', 'roi']].values
X_scaled = StandardScaler().fit_transform(X)

k = 3
kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
labels = kmeans.fit_predict(X_scaled)
centroids = kmeans.cluster_centers_
monthly_df['cluster'] = labels

monthly_df['Month_dt'] = pd.to_datetime(monthly_df['Month'])

recession_periods = [
    ("1958-01", "1962-12"),
    ("1973-01", "1976-12"),
    ("1980-01", "1983-12"),
    ("1990-01", "1992-12"),
    ("2000-01", "2002-12"),
    ("2007-01", "2010-12"),
    ("2020-01", "2022-12"),
]
recession_periods = [(pd.to_datetime(start), pd.to_datetime(end)) for start, end in recession_periods]

fig, ax = plt.subplots(figsize=(14, 7))

scatter = ax.scatter(
    monthly_df['pct_change'], monthly_df['roi'],
    c=monthly_df['cluster'], cmap='tab10', alpha=0.7, edgecolor='k'
)

centroids_orig = StandardScaler().fit(X).inverse_transform(centroids)
ax.scatter(
    centroids_orig[:, 0], centroids_orig[:, 1],
    c='black', marker='X', s=200, label='Centroids'
)

x_min, x_max = ax.get_xlim()
for start, end in recession_periods:
    months_in_range = monthly_df[(monthly_df['Month_dt'] >= start) & (monthly_df['Month_dt'] <= end)]
    if not months_in_range.empty:
        recession_x = months_in_range['pct_change']
        recession_roi = months_in_range['roi']
        ax.scatter(
            recession_x, recession_roi,
            facecolors='none', edgecolors='red', s=80, label='Recession Month' if 'Recession Month' not in ax.get_legend_handles_labels()[1] else ""
        )

ax.set_xlabel("Monthly % Change in S&P 500 Price")
ax.set_ylabel("Average Monthly Movie ROI ($)")
ax.set_title("K-Means++ Clustering: ROI vs. S&P 500 Change (Recessions Highlighted)")
ax.grid(True)
plt.colorbar(scatter, label='Cluster')
ax.legend()
plt.tight_layout()

os.makedirs('results', exist_ok=True)
plt.savefig('results/movie_roi_vs_sp500_pct_change_recessions.png', dpi=300)
plt.show()
