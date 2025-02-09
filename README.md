# CS506-Project

## Project Description

This project aims to investigate the potential correlation between the box office performance of horror and action/adventure films and the movements of the S&P 500 index. Specifically, we will analyze whether high box office revenues from horror movies are associated with downturns in the S&P 500, and conversely, if successful action/adventure films correlate with positive market trends.


## Goals
We will assess and evaluate the correlation between box office revenues of action/adventure and horror movies with the S&P 500 index performance. We will develop a predictive model to determine if box office performance in these genres can serve as an indicator for whether the S&P 500 will perform positively or negatively.

## Data Collection
We will use a dataset that has box office information from each movie and within the dataset it will also contain information such as movie name, genre, and release data. We will use a dataset from Kaggle for this information. We will also need information about the S&P 500 for this project. We can download a dataset from www.nasdaq.com that contains this data.

## Data modelling

We would first employ linear regression models to assess the strength and significance of correlations. Dependent variables: Define the S&P 500 index performance over the chosen time interval. Independent variables : Box office revenues for horror movies and box office revenues for superhero or adventure movies. For control variables, we might consider including other factors that might influence the S&P 500, such as: Interest rates, unemployment rates, and consumer confidence indices.If there is time we would employ other machine learning models like decision trees or neural networks.

## Data Visualization

To analyze the relationship between movie box office revenues and S&P 500 performance, we will use scatter plots to highlight trends and correlations. Residual plots will be employed to assess the validity of our regression model by visualizing the difference between actual data points and predicted values. Additionally, correlation heatmaps will be generated to illustrate the strength of relationships between key variables, providing insights into potential predictive factors.

## Test Plan

The dataset will be divided into training (80%) and testing (20%) subsets to evaluate model performance. To ensure temporal robustness, we will implement a temporal validation approach by training models on earlier years' data and testing them on more recent data. Model accuracy will be assessed using performance metrics such as Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R-squared values.