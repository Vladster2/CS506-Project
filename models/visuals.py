# Courtest of deepseek
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the adjusted dataset
df = pd.read_csv("data/movies_adjusted.csv")

# Filter out movies with missing data
df = df.dropna(subset=['budget', 'revenue', 'budget_adjusted', 'revenue_adjusted'])

# Set up visualization style
sns.set(style="whitegrid")
plt.figure(figsize=(14, 8))

# ----------------------------------
# Budget Comparison
# ----------------------------------
plt.subplot(2, 1, 1)
sns.scatterplot(
    x=df["budget"], 
    y=df["budget_adjusted"],
    hue=df["release_year"],
    palette="viridis",
    alpha=0.7,
    size=df["vote_average"],
    sizes=(20, 200)
)
plt.plot([0, df["budget"].max()], [0, df["budget_adjusted"].max()], 'r--')  # Reference line
plt.xscale('log')
plt.yscale('log')
plt.title("Budget Comparison: Original vs Inflation-Adjusted (Log Scale)")
plt.xlabel("Original Budget (USD)")
plt.ylabel("Adjusted Budget (USD)")
plt.legend(title="Release Year", bbox_to_anchor=(1.05, 1), loc='upper left')

# ----------------------------------
# Revenue Comparison
# ----------------------------------
plt.subplot(2, 1, 2)
sns.scatterplot(
    x=df["revenue"], 
    y=df["revenue_adjusted"],
    hue=df["release_year"],
    palette="cool",
    alpha=0.7,
    size=df["vote_average"],
    sizes=(20, 200)
)
plt.plot([0, df["revenue"].max()], [0, df["revenue_adjusted"].max()], 'r--')  # Reference line
plt.xscale('log')
plt.yscale('log')
plt.title("Revenue Comparison: Original vs Inflation-Adjusted (Log Scale)")
plt.xlabel("Original Revenue (USD)")
plt.ylabel("Adjusted Revenue (USD)")
plt.legend(title="Release Year", bbox_to_anchor=(1.05, 1), loc='upper left')

# Adjust layout and save
plt.tight_layout()
plt.savefig("results/inflation_comparison.png", dpi=300, bbox_inches='tight')
plt.show()