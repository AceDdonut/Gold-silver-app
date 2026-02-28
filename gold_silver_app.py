import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

st.title("Gold vs Silver Price Analysis")

# -----------------------------
# LOAD DATA
# -----------------------------
def load_data():

    with open("gold-prices-vs-silver-prices-historical-chart.csv", "r") as f:
        lines = f.readlines()

    start_row = 0
    for i, line in enumerate(lines):
        if line.strip().startswith("date"):
            start_row = i + 1
            break

    df = pd.read_csv(
        "gold-prices-vs-silver-prices-historical-chart.csv",
        skiprows=start_row,
        header=None,
        names=["Date", "Gold", "Silver"],
        usecols=[0,1,2],
        engine="python"
    )

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Year"] = df["Date"].dt.year
    df["Gold"] = pd.to_numeric(df["Gold"], errors="coerce")
    df["Silver"] = pd.to_numeric(df["Silver"], errors="coerce")

    df = df.dropna()

    return df

df = load_data()

# -----------------------------
# YEAR RANGE SLIDER
# -----------------------------
min_year = int(df["Year"].min())
max_year = int(df["Year"].max())

st.sidebar.header("Filter Options")

year_range = st.sidebar.slider(
    "Select Year Range",
    min_year,
    max_year,
    (2000, 2020)
)

filtered_df = df[(df["Year"] >= year_range[0]) & (df["Year"] <= year_range[1])]

# -----------------------------
# GRAPH
# -----------------------------
st.subheader("Price Trends")

fig, ax = plt.subplots()
ax.plot(filtered_df["Date"], filtered_df["Gold"], label="Gold")
ax.plot(filtered_df["Date"], filtered_df["Silver"], label="Silver")
ax.legend()
st.pyplot(fig)

# -----------------------------
# ESTIMATE PRICE FOR ANY YEAR
# -----------------------------
st.subheader("Estimate Price for Selected Year")

selected_year = st.number_input(
    "Enter a Year to Estimate Price",
    min_value=min_year,
    max_value=2035,
    value=2025
)

# Linear Regression Model
X = df["Year"].values.reshape(-1, 1)

# Gold model
gold_model = LinearRegression()
gold_model.fit(X, df["Gold"])

gold_prediction = gold_model.predict([[selected_year]])[0]

# Silver model
silver_model = LinearRegression()
silver_model.fit(X, df["Silver"])

silver_prediction = silver_model.predict([[selected_year]])[0]

st.write(f"Estimated Gold Price in {selected_year}: ${gold_prediction:.2f}")
st.write(f"Estimated Silver Price in {selected_year}: ${silver_prediction:.2f}")
