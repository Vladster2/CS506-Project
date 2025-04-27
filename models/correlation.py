import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.feature_selection import mutual_info_regression

# Load and prepare data
sp500 = pd.read_csv('data/SP500_Historical_Data_20250323_170505.csv', parse_dates=['Date'])
movies = pd.read_csv('data/output.csv', parse_dates=['release_date'])

# Convert date formats
sp500['Date'] = pd.to_datetime(sp500['Date'], format='%m/%d/%Y')
movies['release_date'] = pd.to_datetime(movies['release_date'], format='%d-%m-%Y')

# Merge datasets
merged = pd.merge_asof(movies.sort_values('release_date'), 
                      sp500.sort_values('Date'),
                      left_on='release_date',
                      right_on='Date',
                      direction='nearest')

# Enhanced time-lagged analysis function
def enhanced_time_lagged_analysis(df, target_col, feature_col, windows):
    results = {}
    
    for window_name, days in windows.items():
        corrs = []
        for lag in range(-days, days+1):
            df[f'lagged_{feature_col}'] = df[feature_col].shift(lag)
            valid_data = df.dropna(subset=[target_col, f'lagged_{feature_col}'])
            
            # Calculate multiple metrics
            pearson = valid_data[target_col].corr(valid_data[f'lagged_{feature_col}'])
            spearman = valid_data[target_col].corr(valid_data[f'lagged_{feature_col}'], method='spearman')
            mi = mutual_info_regression(valid_data[[f'lagged_{feature_col}']], valid_data[target_col])
            
            corrs.append({
                'lag': lag,
                'pearson': pearson,
                'spearman': spearman,
                'mi': mi[0]
            })
        
        results[window_name] = pd.DataFrame(corrs)
    
    return results

# Analyze 1-week and 2-week windows
windows = {
    '1_week': 7,
    '2_weeks': 30
}

lag_results = enhanced_time_lagged_analysis(merged, 'revenue', 'Close', windows)

# Visualization
plt.figure(figsize=(18, 12))

# 1-Week Window Analysis
plt.subplot(2, 2, 1)
plt.plot(lag_results['1_week']['lag'], lag_results['1_week']['pearson'], label='Pearson')
plt.plot(lag_results['1_week']['lag'], lag_results['1_week']['spearman'], label='Spearman')
plt.plot(lag_results['1_week']['lag'], lag_results['1_week']['mi'], label='MI')
plt.title('1-Week Window Correlation Analysis')
plt.xlabel('Days Relative to Release Date')
plt.ylabel('Correlation/MI Score')
plt.axvline(0, color='gray', linestyle='--')
plt.legend()

# 2-Week Window Analysis
plt.subplot(2, 2, 2)
plt.plot(lag_results['2_weeks']['lag'], lag_results['2_weeks']['pearson'], label='Pearson')
plt.plot(lag_results['2_weeks']['lag'], lag_results['2_weeks']['spearman'], label='Spearman')
plt.plot(lag_results['2_weeks']['lag'], lag_results['2_weeks']['mi'], label='MI')
plt.title('2-Week Window Correlation Analysis')
plt.xlabel('Days Relative to Release Date')
plt.ylabel('Correlation/MI Score')
plt.axvline(0, color='gray', linestyle='--')
plt.legend()

# Combined MI Comparison
plt.subplot(2, 2, 3)
plt.plot(lag_results['1_week']['lag'], lag_results['1_week']['mi'], label='1-Week MI')
plt.plot(lag_results['2_weeks']['lag'], lag_results['2_weeks']['mi'], label='2-Week MI')
plt.title('Mutual Information Comparison')
plt.xlabel('Days Relative to Release Date')
plt.ylabel('MI Score')
plt.axvline(0, color='gray', linestyle='--')
plt.legend()

# Peak Correlation Table
max_corrs = pd.DataFrame({
    'Window': ['1-Week', '2-Week'],
    'Max Pearson': [
        lag_results['1_week']['pearson'].max(),
        lag_results['2_weeks']['pearson'].max()
    ],
    'Max Spearman': [
        lag_results['1_week']['spearman'].max(),
        lag_results['2_weeks']['spearman'].max()
    ],
    'Max MI': [
        lag_results['1_week']['mi'].max(),
        lag_results['2_weeks']['mi'].max()
    ]
})

plt.subplot(2, 2, 4)
plt.axis('off')
plt.table(cellText=max_corrs.values,
          colLabels=max_corrs.columns,
          loc='center',
          cellLoc='center')
plt.title('Peak Correlation Values')

plt.tight_layout()
plt.show()

# Print statistical significance
for window in ['1_week', '2_weeks']:
    print(f"\n{window.replace('_', ' ').title()} Significance:")
    print(f"Pearson p-value: {stats.pearsonr(merged['revenue'], merged['Close'])[1]:.4f}")
    print(f"Spearman p-value: {stats.spearmanr(merged['revenue'], merged['Close'])[1]:.4f}")