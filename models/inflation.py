import pandas as pd


movies = pd.read_csv("data/output.csv", parse_dates=["release_date"])
movies["release_year"] = movies["release_date"].dt.year


cpi = pd.read_csv("data/CPI.csv")


movies = movies.merge(cpi, left_on="release_year", right_on="Year", how="left")


target_year = 2016
target_cpi = cpi.loc[cpi["Year"] == target_year, "CPI"].values[0]


movies["inflation_multiplier"] = target_cpi / movies["CPI"]


movies["budget_adjusted"] = movies["budget"] * movies["inflation_multiplier"]
movies["revenue_adjusted"] = movies["revenue"] * movies["inflation_multiplier"]


movies.drop(columns=["CPI", "inflation_multiplier"], inplace=True)


movies.to_csv("movies_adjusted.csv", index=False)