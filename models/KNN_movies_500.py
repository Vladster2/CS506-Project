import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

genres_list = ['Horror', 'Fantasy', 'Science-Fiction', 'Action', 'Adventure', 'Family', 'Thriller', 'Comedy', 'Crime', 'Drama', 'Science-Fiction', 'Mystery', 'War', 'History', 'Documentary', 'Animation']
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






