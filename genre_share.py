"""
load_sp500.py
Creates:
  • results/genre_frequency_table.md
  • results/genre_market_share_area.png
"""

import pandas as pd
import matplotlib.pyplot as plt
import pathlib, os

# ------------------------------------------------------------------
# 0.  DIRECTORY HANDLING
# ------------------------------------------------------------------
BASE    = pathlib.Path(__file__).resolve().parent     # CS506-Project/
DATA    = BASE / "data"
RESULTS = BASE / "results"; RESULTS.mkdir(exist_ok=True)

# ------------------------------------------------------------------
# 1.  PARAMETERS
# ------------------------------------------------------------------
TARGET_GENRES = [
    "horror","comedy","drama","action",
    "romance","science fiction","thriller","fantasy"
]

RECESSION_PERIODS = [
    ("1958-01", "1962-12"),
    ("1973-01", "1976-12"),
    ("1980-01", "1983-12"),
    ("1990-01", "1992-12"),
    ("2000-01", "2002-12"),
    ("2007-01", "2010-12"),
    ("2020-01", "2022-12"),
]
RECESSION_PERIODS = [(pd.to_datetime(a), pd.to_datetime(b)) for a, b in RECESSION_PERIODS]

def is_recession(ts):
    return any(s <= ts <= e for s, e in RECESSION_PERIODS)

# ------------------------------------------------------------------
# 2.  LOAD FILES  (adjust filenames if different)
# ------------------------------------------------------------------
sp500_path  = DATA / "SP500_Historical_Data_20250323_170505.csv"              # adjust if file is named differently
movies_path = DATA / "movie_dataset.csv"      # adjust if using movies.csv


sp = pd.read_csv(sp500_path,  parse_dates=["Date"])
mv = pd.read_csv(movies_path, parse_dates=["release_date"], dayfirst=True)

# ------------------------------------------------------------------
# 3.  PREPARE S&P 500  → monthly hardship score
# ------------------------------------------------------------------
sp["Month"] = sp["Date"].dt.to_period("M")
mon_sp = sp.groupby("Month")["Close"].mean().reset_index()
mon_sp["pct_change"] = mon_sp["Close"].pct_change()
mon_sp["hardship"]   = mon_sp["pct_change"].clip(upper=0).abs()   # bigger ↓ → worse
mh_min, mh_max = mon_sp["hardship"].min(), mon_sp["hardship"].max()
mon_sp["hardship_norm"] = (mon_sp["hardship"] - mh_min) / (mh_max - mh_min)
hard_months = mon_sp.loc[mon_sp["hardship_norm"] > 0.5, "Month"].astype(str)

# ------------------------------------------------------------------
# 4.  PREPARE MOVIES  → explode by target genre
# ------------------------------------------------------------------
mv["Month"] = mv["release_date"].dt.to_period("M")
mv["genres_clean"] = mv["genres"].fillna("").str.lower().str.replace("-", " ")

def match_genres(g):
    return [tg for tg in TARGET_GENRES if tg in g]

mv["target"] = mv["genres_clean"].apply(match_genres)
movies_exp = mv.explode("target").dropna(subset=["target"])

# ------------------------------------------------------------------
# 5.  GENRE-FREQUENCY BASELINE TABLE
# ------------------------------------------------------------------
overall   = movies_exp["target"].value_counts()
hard_only = movies_exp[movies_exp["Month"].astype(str).isin(hard_months)]["target"].value_counts()

tbl = (pd.DataFrame({"Overall": overall, "Hardship": hard_only})
         .fillna(0).astype(int))
tbl["Overall %"]  = (tbl["Overall"]  / tbl["Overall"].sum()*100).round(1)
tbl["Hardship %"] = (tbl["Hardship"] / tbl["Hardship"].sum()*100).round(1)
tbl = tbl.sort_values("Overall", ascending=False)

md = "| Genre | Overall Share (%) | Hardship Share (%) |\n|---|---|---|\n"
for g, r in tbl.iterrows():
    md += f"| {g.title()} | {r['Overall %']} | {r['Hardship %']} |\n"

(RESULTS / "genre_frequency_table.md").write_text(md)
print("✔ genre_frequency_table.md written to results/\n")

# ------------------------------------------------------------------
# 6.  STACKED-AREA “MARKET SHARES” PLOT
# ------------------------------------------------------------------
# =========  A) YEARLY SMOOTHED AREA PLOT  =========
# build month→genre counts (if you didn't keep it)
# =========  B) TOP-N AREA PLOT  =========
top_n = 4                                      # change if you want more/less
month_genre = (movies_exp
               .groupby(['Month', 'target'])
               .size()
               .unstack(fill_value=0))

top_genres = month_genre.sum().nlargest(top_n).index.tolist()

# lump everything else into "Other"
month_genre['Other'] = month_genre.drop(columns=top_genres).sum(axis=1)
shares_top = month_genre[top_genres + ['Other']]
shares_top = shares_top.div(shares_top.sum(axis=1), axis=0)

# yearly resample so it matches the smoother style
shares_top_year = shares_top.resample('Y').mean()

fig, ax = plt.subplots(figsize=(12, 6))
cols = plt.get_cmap("tab10").colors
shares_top_year.plot.area(ax=ax, color=cols[:len(shares_top_year.columns)], alpha=.9)

RECESSION_COLOR = "#8B0000"    # dark-red
RECESSION_ALPHA = 0.20         # 20 % opacity
RECESSION_HATCH = "//"         # diagonal stripes


for s, e in RECESSION_PERIODS:
    ax.axvspan(
        s, e,
        color=RECESSION_COLOR,
        alpha=RECESSION_ALPHA,
        hatch=RECESSION_HATCH,
        linewidth=0          # no border, just hatch
    )


ax.set_ylabel("Share of Releases")
ax.set_xlabel("Year")
ax.set_title(f"Top-{top_n} Genres vs Other (grey = recession)")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(RESULTS / "genre_share_topN_area.png", dpi=300)
plt.close()
print("✔  genre_share_topN_area.png saved to results/")


