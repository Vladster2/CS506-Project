# CS506-Project Proposal 

## Project Description
This project aims to investigate a correlation between the box office performance of horror and action/adventure films and the movements of the S&P 500 index. Specifically, we will analyze whether high box office revenues from horror movies are associated with downturns in the S&P 500 and conversely if successful action/adventure films correlate with positive market trends.

## Goals
We will assess and evaluate the correlation between box office revenues of action/adventure and horror movies with the S&P 500 index performance. We will develop a predictive model to determine if box office performance in these genres can serve as an indicator for whether the S&P 500 will perform positively or negatively.

## Data Collection
We will use a dataset that has box office information from each movie and within the dataset it will also contain information such as movie name, genre, and release date information. We will use a dataset from Kaggle for this information regarding movies: https://www.kaggle.com/datasets/karthiknamboori1/movie-datasets. We will also need information about the S&P 500 for this project. We can download a dataset from https://www.nasdaq.com/market-activity/index/spx/historical that contains this data.

## Data modelling
We would first employ linear regression models to assess the strength and significance of correlations. Our dependent variable would be to define the S&P 500 index performance over our chosen time interval. Our indepedent variable would be box office revenues for horror movies and box office revenues for action/adventure movies. For control variables, we might consider including other factors that might influence the S&P 500, such as interest rates, unemployment rates, and consumer confidence indices. If there is time we would employ other machine learning models like decision trees or neural networks.

## Data Visualization
To analyze the relationship between movie box office revenues and S&P 500 performance, we will use scatter plots to highlight trends and correlations. Residual plots can be used to assess the validity of our regression model by visualizing the difference between actual data points and predicted values. Additionally, correlation heatmaps can be generated to illustrate the strength of relationships between variables which can provide insights into potential predictive factors.

## Test Plan
We will withhold 20% of data for testing and use the other 80% of our data for training. We will also train our models on earlier years' data and testing them on more recent data as well. Model accuracy will be assessed using performance metrics such as Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R-squared values.