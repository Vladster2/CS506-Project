import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os

def load_movie_data():
    """Load and preprocess movie dataset"""
    movie_df = pd.read_csv('data/movie_dataset.csv')
    
    # Convert release_date to datetime and extract year
    movie_df['release_date'] = pd.to_datetime(movie_df['release_date'], errors='coerce')
    movie_df['release_year'] = movie_df['release_date'].dt.year
    
    # Calculate ROI (Return on Investment)
    movie_df['roi'] = (movie_df['revenue'] - movie_df['budget']) / movie_df['budget']
    
    # Replace infinite values with NaN and then drop NaN values
    movie_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    return movie_df

def load_economic_data():
    """Load and preprocess economic data"""
    economic_df = pd.read_csv('data/SP500_Historical_Data_20250323_170505.csv')
    
    # Convert Date to datetime
    economic_df['Date'] = pd.to_datetime(economic_df['Date'])
    economic_df['Year'] = economic_df['Date'].dt.year
    
    # Calculate yearly returns
    economic_df = economic_df.groupby('Year').agg({
        'Close': ['first', 'last']
    }).reset_index()
    
    economic_df.columns = ['Year', 'Open', 'Close']
    economic_df['yearly_return'] = (economic_df['Close'] - economic_df['Open']) / economic_df['Open']
    
    # Define recession years (negative yearly returns)
    economic_df['recession'] = economic_df['yearly_return'] < 0
    
    # Create recession severity score (0-1 scale)
    economic_df['recession_severity'] = np.abs(np.minimum(economic_df['yearly_return'], 0))
    max_severity = economic_df['recession_severity'].max()
    if max_severity > 0:
        economic_df['recession_severity'] = economic_df['recession_severity'] / max_severity
    
    return economic_df

def prepare_features_for_model(movie_df, economic_df):
    """Prepare features for the model by aggregating movie data by year"""
    # Merge movie data with economic data
    movie_df = pd.merge(
        movie_df,
        economic_df[['Year', 'recession']],
        left_on='release_year',
        right_on='Year',
        how='inner'
    )
    
    # Extract all unique genres
    all_genres = set()
    for genres in movie_df['genres'].dropna():
        all_genres.update(genres.split())
    
    # Select top genres for analysis
    top_genres = ['Drama', 'Comedy', 'Action', 'Thriller', 'Documentary', 'Horror', 
                 'Romance', 'Science-Fiction', 'Fantasy', 'Adventure']
    
    # Create yearly aggregated features
    yearly_data = []
    
    for year, group in movie_df.groupby('release_year'):
        year_data = {'Year': year}
        
        # Economic target variable
        recession_data = economic_df[economic_df['Year'] == year]
        if not recession_data.empty:
            year_data['recession'] = recession_data['recession'].iloc[0]
        else:
            continue  # Skip years with no economic data
        
        # Movie count
        year_data['movie_count'] = len(group)
        
        # Average budget, revenue, and ROI
        year_data['avg_budget'] = group['budget'].mean()
        year_data['avg_revenue'] = group['revenue'].mean()
        year_data['avg_roi'] = group['roi'].mean()
        
        # Genre counts
        for genre in top_genres:
            year_data[f'{genre}_count'] = sum(group['genres'].str.contains(genre, na=False))
            
            # Genre percentage
            year_data[f'{genre}_percent'] = year_data[f'{genre}_count'] / year_data['movie_count'] if year_data['movie_count'] > 0 else 0
        
        yearly_data.append(year_data)
    
    yearly_df = pd.DataFrame(yearly_data)
    return yearly_df

def train_recession_prediction_model(yearly_df):
    """Train a Random Forest model to predict recessions"""
    # Define features and target
    feature_cols = [col for col in yearly_df.columns if col not in ['Year', 'recession']]
    X = yearly_df[feature_cols]
    y = yearly_df['recession']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return model, X.columns

def plot_feature_importance(model, feature_names):
    """Plot feature importance from the Random Forest model"""
    # Get feature importances
    importances = model.feature_importances_
    
    # Create DataFrame for easier manipulation
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    # Extract genre features only
    genre_features = [f for f in feature_importance_df['Feature'] if '_count' in f or '_percent' in f]
    genre_importance_df = feature_importance_df[feature_importance_df['Feature'].isin(genre_features)]
    
    # Extract genre names from feature names
    genre_importance_df['Genre'] = genre_importance_df['Feature'].apply(
        lambda x: x.split('_')[0]
    )
    
    # Aggregate importance by genre
    genre_agg = genre_importance_df.groupby('Genre')['Importance'].sum().reset_index()
    
    # Sort by importance
    genre_agg = genre_agg.sort_values('Importance', ascending=True)
    
    # Select top 5 genres
    top_genres = genre_agg.tail(5)
    
    # Plot
    plt.figure(figsize=(12, 8))
    bars = plt.barh(top_genres['Genre'], top_genres['Importance'], color='skyblue')
    
    # Add grid
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Add labels and title
    plt.xlabel('Importance Score', fontsize=14)
    plt.title('Feature Importance (Random Forest) for Recession Prediction', fontsize=16)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/genre_importance_recession_prediction.png', dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return genre_agg

if __name__ == "__main__":
    print("Loading data...")
    movie_df = load_movie_data()
    economic_df = load_economic_data()
    
    print("Preparing features...")
    yearly_df = prepare_features_for_model(movie_df, economic_df)
    
    print("Training model...")
    model, feature_names = train_recession_prediction_model(yearly_df)
    
    print("Plotting feature importance...")
    genre_importance = plot_feature_importance(model, feature_names)
    
    print("\nGenre Importance for Recession Prediction:")
    print(genre_importance.sort_values('Importance', ascending=False))