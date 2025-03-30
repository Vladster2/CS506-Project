import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import os

# Set up plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set(font_scale=1.2)

def load_movie_data():
    """Load and preprocess the movie dataset."""
    movie_df = pd.read_csv('data/movie_dataset.csv')
    
    movie_df['release_date'] = pd.to_datetime(movie_df['release_date'], format='%d-%m-%Y', errors='coerce')
    movie_df['budget'] = pd.to_numeric(movie_df['budget'], errors='coerce')
    movie_df['revenue'] = pd.to_numeric(movie_df['revenue'], errors='coerce')
    movie_df['roi'] = (movie_df['revenue'] - movie_df['budget']) / movie_df['budget']
    movie_df['release_year'] = movie_df['release_date'].dt.year
    
    return movie_df

def load_economic_data():
    """Load and preprocess S&P 500 historical data as an economic indicator."""
    sp500_df = pd.read_csv('data/SP500_Historical_Data_20250323_170505.csv')
    
    # Convert date column to datetime
    sp500_df['Date'] = pd.to_datetime(sp500_df['Date'])
    
    # Calculate yearly metrics
    sp500_yearly = sp500_df.copy()
    sp500_yearly['Year'] = sp500_yearly['Date'].dt.year
    
    yearly_metrics = sp500_yearly.groupby('Year').agg(
        yearly_return=('Close', lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] if len(x) > 1 else 0),
        volatility=('Close', lambda x: x.pct_change().std())
    ).reset_index()
    
    # Create a recession severity score (higher = worse economic conditions)
    yearly_metrics['recession_severity'] = (
        -yearly_metrics['yearly_return'] + 
        yearly_metrics['volatility'] / yearly_metrics['volatility'].max()
    )
    
    # Normalize the severity score to 0-1 range
    min_severity = yearly_metrics['recession_severity'].min()
    max_severity = yearly_metrics['recession_severity'].max()
    yearly_metrics['recession_severity'] = (yearly_metrics['recession_severity'] - min_severity) / (max_severity - min_severity)
    
    return yearly_metrics

def analyze_genre_roi_vs_economic_hardship(movie_df, economic_df, genres):
    """
    Perform linear regression between economic hardship severity and ROI for multiple genres.
    Plot one regression line for each genre.
    """
    plt.figure(figsize=(14, 10))
    
    # Create a color palette for different genres
    colors = plt.cm.tab10(np.linspace(0, 1, len(genres)))
    
    # Store regression results
    regression_results = []
    
    # Process each genre
    for i, genre in enumerate(genres):
        # Extract movies of the specified genre
        genre_movies = movie_df[movie_df['genres'].str.contains(genre, na=False)]
        
        # Merge with economic data
        merged_data = pd.merge(
            genre_movies,
            economic_df,
            left_on='release_year',
            right_on='Year',
            how='inner'
        )
        
        # Filter out invalid ROI values and NaN values
        valid_data = merged_data[
            np.isfinite(merged_data['roi']) & 
            np.isfinite(merged_data['recession_severity'])
        ]
        
        if len(valid_data) < 2:
            print(f"Warning: Insufficient data for genre '{genre}'. Skipping.")
            continue
        
        # Plot scatter points
        plt.scatter(
            valid_data['recession_severity'], 
            valid_data['roi'],
            alpha=0.3,
            color=colors[i],
            label=f"{genre} Movies"
        )
        
        # Perform linear regression
        X = valid_data[['recession_severity']].values
        y = valid_data['roi'].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Generate points for the regression line
        x_range = np.linspace(0, 1, 100).reshape(-1, 1)
        y_pred = model.predict(x_range)
        
        # Plot regression line
        plt.plot(
            x_range, 
            y_pred, 
            linewidth=2,
            color=colors[i],
            label=f"{genre} Trend (coef={model.coef_[0]:.3f})"
        )
        
        # Calculate correlation
        correlation = np.corrcoef(valid_data['recession_severity'], valid_data['roi'])[0, 1]
        
        # Store results
        regression_results.append({
            'genre': genre,
            'correlation': correlation,
            'coefficient': model.coef_[0],
            'intercept': model.intercept_,
            'sample_size': len(valid_data)
        })
    
    # Add plot details
    plt.title('Movie ROI vs. Economic Hardship Severity by Genre', fontsize=16)
    plt.xlabel('Economic Hardship Severity (0-1 scale)', fontsize=14)
    plt.ylabel('Return on Investment (ROI)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Set y-axis range from 0 to 2
    plt.ylim(-1, 30)
    
    # Add a table with regression results
    # table_text = []
    # for result in regression_results:
    #     table_text.append(f"{result['genre']} (n={result['sample_size']}): Corr={result['correlation']:.3f}, Coef={result['coefficient']:.3f}")
    
    # plt.figtext(
    #     0.5, 0.01, 
    #     '\n'.join(table_text),
    #     ha='center',
    #     fontsize=10,
    #     bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8)
    # )
    
    # Add legend with smaller font
    plt.legend(fontsize=9, loc='upper left')
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Make room for the table
    
    # Save the figure
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/genre_roi_vs_economic_hardship.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return regression_results

if __name__ == "__main__":
    print("Loading data...")
    movie_df = load_movie_data()
    economic_df = load_economic_data()

    above_05 = {}

    # Count movies by genre during high recession periods (severity > 0.5)
    for index, row in movie_df.iterrows():
        year = row['release_year']
        if pd.isna(year):
            continue
            
        # Get recession severity for this year
        recession_data = economic_df[economic_df['Year'] == year]['recession_severity']
        
        # Skip if no economic data for this year
        if recession_data.empty:
            continue
            
        recession_severity = recession_data.values[0]
        
        # Check if recession severity is above threshold
        if recession_severity > 0.5:
            # Parse genres - they're separated by spaces in the dataset
            if pd.notna(row['genres']):
                genres = row['genres'].split()
                for genre in genres:
                    if genre not in above_05:
                        above_05[genre] = 0
                    above_05[genre] += 1
    
    # Print genres with counts during high recession periods
    print("\nGenres released during high recession periods (severity > 0.5):")
    for genre, count in sorted(above_05.items(), key=lambda x: x[1], reverse=True):
        print(f"{genre}: {count} movies")
    
    # Define genres to compare
    comparison_genres = ['Horror', 'Comedy', 'Drama', 'Action', 'Romance', 'Science-Fiction', 'Thriller', 'Fantasy']


    
    print("Analyzing genre performance during economic hardship...")
    regression_results = analyze_genre_roi_vs_economic_hardship(movie_df, economic_df, comparison_genres)
    
    # Print regression results in a table format
    print("\nRegression Results by Genre:")
    print("-" * 75)
    print(f"{'Genre':<15} {'Sample Size':<12} {'Correlation':<15} {'Coefficient':<15} {'Intercept':<15}")
    print("-" * 75)
    for result in regression_results:
        print(f"{result['genre']:<15} {result['sample_size']:<12} {result['correlation']:<15.3f} {result['coefficient']:<15.3f} {result['intercept']:<15.3f}")
    print("-" * 75)
    
    # Interpret the results
    print("\nInterpretation:")
    
    # Find horror result
    horror_result = next((r for r in regression_results if r['genre'] == 'Horror'), None)
    
    if horror_result:
        if horror_result['coefficient'] > 0:
            print("- Horror movies show a POSITIVE relationship with economic hardship severity.")
            print("  This suggests horror films tend to be MORE profitable during economic downturns.")
        else:
            print("- Horror movies show a NEGATIVE relationship with economic hardship severity.")
            print("  This suggests horror films tend to be LESS profitable during economic downturns.")
        
        # Compare with other genres
        positive_coef_genres = [r for r in regression_results if r['coefficient'] > 0]
        sorted_genres = sorted(regression_results, key=lambda x: x['coefficient'], reverse=True)
        
        horror_rank = next((i+1 for i, r in enumerate(sorted_genres) if r['genre'] == 'Horror'), None)
        
        if horror_rank:
            print(f"- Among {len(regression_results)} analyzed genres, Horror ranks #{horror_rank} in terms of")
            print("  positive relationship with economic hardship.")
        
        if len(positive_coef_genres) > 0:
            print(f"- {len(positive_coef_genres)} out of {len(regression_results)} genres show a positive relationship")
            print("  with economic hardship, suggesting these genres may be 'recession-resistant'.")
    
    print("\nAnalysis complete! Results saved to the 'results' directory.") 