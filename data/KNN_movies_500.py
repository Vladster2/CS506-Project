import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
"""
# Read data
movies = pd.read_csv("data/movie_dataset.csv")  # Update with actual path
sp500 = pd.read_csv("data/SP500_Historical_Data_20250323_170505.csv")  # Update with actual path

# Convert dates
movies["release_date"] = pd.to_datetime(movies["release_date"], format="%d-%m-%Y")
sp500["Date"] = pd.to_datetime(sp500["Date"], format="%m/%d/%Y")

# Filter horror movies
horror_movies = movies[movies["genres"].str.contains("Horror", na=False)]

# Calculate S&P 500 trends (e.g., 30-day rolling trend)
sp500["Returns"] = sp500["Close"].pct_change()
sp500["Trend"] = np.where(sp500["Returns"].rolling(30).mean() < 0, 1, 0)  # 1 = Downtrend

# Merge datasets based on closest date before release
horror_movies["market_trend"] = horror_movies["release_date"].apply(
    lambda x: sp500.loc[sp500["Date"] <= x, "Trend"].iloc[-1] if not sp500[sp500["Date"] <= x].empty else np.nan
)

# Drop NaN values
horror_movies = horror_movies.dropna(subset=["market_trend"])

# Prepare features and target
X = horror_movies[["market_trend"]]  # Market trend as feature
y = horror_movies["revenue"]  # Revenue as target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train KNN model
knn = KNeighborsRegressor(n_neighbors=3)
knn.fit(X_train_scaled, y_train)

# Generate predictions
y_pred = knn.predict(X_test_scaled)

# Evaluate
score = knn.score(X_test_scaled, y_test)
print(f"KNN Model Score (R²): {score}")

# Scatter plot for actual vs predicted values
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.title("KNN Regression: Actual vs Predicted Revenue")
plt.xlabel("Actual Revenue")
plt.ylabel("Predicted Revenue")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2)  # Line y = x
plt.grid(True)
plt.show()
"""

genres_list = ['Horror', 'Fantasy', 'Science-Fiction', 'Action', 'Adventure', 'Family', 'Thriller', 'Comedy', 'Crime', 'Drama']
for i in range(len(genres_list)):
    movie_df = pd.read_csv("data/movie_dataset.csv")
    movie_df = movie_df[movie_df['genres'].str.contains(genres_list[i], case=False, na=False)]
    stock_df = pd.read_csv("data/SP500_Historical_Data_20250323_170505.csv")

    movie_df['revenue_minus_budget'] = movie_df['revenue'] - movie_df['budget']

    data = pd.merge(movie_df[['revenue_minus_budget']], stock_df[['Close']], left_index=True, right_index=True)

    X = data[['revenue_minus_budget']].values 
    y = data['Close'].values  

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    knn = KNeighborsRegressor(n_neighbors=1) 
    knn.fit(X_scaled, y)

    y_pred = knn.predict(X_scaled)

    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    print(f'R-squared {genres_list[i]}: {r2}')






