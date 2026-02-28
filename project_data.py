import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# ==============================
# LOAD DATA SAFELY
# ==============================

gold = pd.read_csv(
    "gold-prices-vs-silver-prices-historical-chart.csv",
    sep=None,
    engine="python",
    on_bad_lines="skip"
)

silver = pd.read_csv(
    "LBMA-silver_D-silver_D_USD.csv",
    sep=None,
    engine="python",
    on_bad_lines="skip"
)

print("Gold Data Preview:")
print(gold.head())

print("\nSilver Data Preview:")
print(silver.head())

# ==============================
# SIMPLE REGRESSION EXAMPLE
# ==============================

# Try to automatically detect numeric column
gold_numeric = gold.select_dtypes(include=[np.number])
silver_numeric = silver.select_dtypes(include=[np.number])

if len(gold_numeric.columns) > 0:
    y_gold = gold_numeric.iloc[:, 0].dropna()
    X_gold = np.arange(len(y_gold)).reshape(-1, 1)

    model_gold = LinearRegression()
    model_gold.fit(X_gold, y_gold)

    print("\nGold Regression Coefficient:", model_gold.coef_[0])
    print("Gold Regression Intercept:", model_gold.intercept_)

if len(silver_numeric.columns) > 0:
    y_silver = silver_numeric.iloc[:, 0].dropna()
    X_silver = np.arange(len(y_silver)).reshape(-1, 1)

    model_silver = LinearRegression()
    model_silver.fit(X_silver, y_silver)

    print("\nSilver Regression Coefficient:", model_silver.coef_[0])
    print("Silver Regression Intercept:", model_silver.intercept_)

print("\nProgram completed successfully.")
