# CS506-Project Proposal 

## Project Plan

### Project Description
Our project investigates the relationship between economic conditions and movie performance, with a specific focus on horror films. We aim to test the hypothesis that during economic downturns, horror movies not only increase in production but also outperform other genres in terms of return on investment (ROI).

### Goals
1. Analyze the correlation between economic indicators (S&P 500, recession periods) and horror movie performance metrics (box office revenue, ROI)
2. Compare horror movie performance against other genres during economic downturns
3. Identify patterns in horror movie production volume during different economic cycles

### Data Collection
We will use a dataset that has box office information from each movie and within the dataset it will also contain information such as movie name, genre, and release date information. We will use a dataset from Kaggle for this information regarding movies: https://www.kaggle.com/datasets/karthiknamboori1/movie-datasets. We will also need information about the S&P 500 for this project. We can download a dataset from https://www.nasdaq.com/market-activity/index/spx/historical that contains this data.

## Midterm Report Video

https://www.youtube.com/watch?v=hgbGlpkEcKk

## Description of data processing done so far

We created seperate .csv files that just have horror movies from our first movie dataset .csv file. We have also created .csv files that contain a movie's budget and revenue adjusted for inflation. 

## Linear Regression Model

We perfomed linear regression to examine the relationship between economic conditions and movie performance across different genres.

1. **Data Processing**: We preprocess movie data to calculate ROI (Return on Investment), while economic data is transformed into a recession severity score based on S&P 500 yearly returns and volatility.

2. **Economic Hardship Metric**: We create a normalized recession severity score (0-1 scale) where higher values indicate worse economic conditions, derived from negative yearly returns and market volatility.

3. **Genre Comparison**: We analyze multiple genres (Horror, Comedy, Drama, Action, Romance, Science-Fiction, Thriller, Fantasy) to compare their performance during different economic conditions.

4. **Regression Analysis**: For each genre, we perform linear regression between economic hardship severity and movie ROI, calculating correlation coefficients and regression slopes to quantify the relationship.

5. **Visualization**: We generate scatter plots with regression lines for each genre, allowing visual comparison of how different genres respond to economic downturns.

### Scatter Plot with Regression Line for Horror Movies

![Linear Regression Results](results/linear_regression.png)

### Regression Results Table

Our linear regression analysis comparing economic hardship severity with movie ROI across genres yielded the following results:

| Genre           | Sample Size  | Correlation     | Coefficient     | Intercept      |
|----------------|--------------|----------------|----------------|----------------|
| Horror          | 360          | -0.069          | -26046.786      | 11110.064      |
| Comedy          | 1185         | -0.020          | -27049.653      | 16591.332      |
| Drama           | 1550         | -0.015          | -18093.318      | 11575.840      |
| Action          | 949          | -0.039          | -2.160          | 3.175          |
| Romance         | 613          | 0.022           | 1.179           | 3.166          |
| Science-Fiction | 442          | -0.011          | -2.285          | 5.527          |
| Thriller        | 966          | -0.037          | -6671.453       | 3294.238       |
| Fantasy         | 343          | 0.031           | 6.805           | 2.530          |

These results show the correlation coefficients, regression coefficients, and intercepts for each genre, providing quantitative measures of how movie ROI relates to economic conditions across different types of films.

### Movie Production During Economic Downturns

Our analysis of movie releases during high recession periods (economic hardship severity > 0.5) revealed the following distribution by genre:

```
Drama: 321 movies
Comedy: 253 movies
Thriller: 184 movies
Action: 176 movies
Romance: 151 movies
Adventure: 128 movies
Crime: 110 movies
Family: 82 movies
Science-Fiction: 75 movies
Fantasy: 54 movies
Mystery: 49 movies
Horror: 48 movies
Animation: 41 movies
War: 27 movies
Music: 22 movies
History: 21 movies
Documentary: 12 movies
Foreign: 8 movies
Western: 7 movies
```

## KNN Model

We also used a KNN model to see if there was a relationship between S&P 500 close prices and the return on investment (ROI) of movies based on genre. After running a KNN model with these two different variables we found the following information:

```
R-squared Horror: 0.759705151925964
R-squared Fantasy: 0.9825107332839583
R-squared Science-Fiction: 0.983449523590625
R-squared Action: 0.9727091902434848
R-squared Adventure: 0.9871167995233564
R-squared Family: 0.971742928596874
R-squared Thriller: 0.929324090010516
R-squared Comedy: 0.9365027951550314
R-squared Crime: 0.9823181695215141
R-squared Drama: 0.8750054404047017
```

## Preliminary Results

These findings from our linear regression model indicate patterns in movie production trends during economic downturns and suggest that certain genres are more resilient to economic hardship than others. Something we noticed that is related to our inital hypothesis was that horror movies are not commonly made during economic downturns. This information leads us to think that there is a pattern or corellation between movie production and the S&P 500, but our initial hypothesis of focusing on horror movies might be incorrect. We will use this information to inspect our data more closely and look at other genres to see if there is more of a correlation with the S&P 500 compared to horror movies.

From our inital KNN model we found that are R-squared values are relatively high which could suggest we have found a correlation between a movie's return on investment and the close prices of the S&P 500. Of the popular genres we tested in our KNN model we found that horror movies seem to have the lowest correlation which further suggests that our inital hypothesis might be incorrect. 

These two models have starting to give us a clear pattern of our data which will allow us to investigate our data further and reach our final conclusions. 