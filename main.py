
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Create folders if they don't exist
os.makedirs("outputs", exist_ok=True)
os.makedirs("insights", exist_ok=True)

# Load dataset
df = pd.read_csv("data/walmart.csv")

# Convert Date column
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

# Sort by date
df = df.sort_values("Date")

# Feature Engineering
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Week"] = df["Date"].dt.isocalendar().week.astype(int)
df["Day"] = df["Date"].dt.day

# Features and Target
X = df[
    [
        "Store",
        "Holiday_Flag",
        "Temperature",
        "Fuel_Price",
        "CPI",
        "Unemployment",
        "Year",
        "Month",
        "Week",
        "Day",
    ]
]

y = df["Weekly_Sales"]

# Train-Test Split
split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# Model
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

metrics = f"""
WALMART SALES FORECASTING RESULTS
=================================

MAE  : {mae:,.2f}
RMSE : {rmse:,.2f}
R2   : {r2:.4f}
"""

print(metrics)

with open("outputs/metrics.txt", "w") as f:
    f.write(metrics)

# Actual vs Predicted Plot
plt.figure(figsize=(12, 6))

plt.plot(
    y_test.values[:200],
    label="Actual"
)

plt.plot(
    predictions[:200],
    label="Predicted"
)

plt.title("Actual vs Predicted Weekly Sales")
plt.xlabel("Records")
plt.ylabel("Weekly Sales")
plt.legend()
plt.tight_layout()

plt.savefig(
    "outputs/actual_vs_predicted.png"
)

plt.close()

# Store-wise Sales Plot
store_sales = (
    df.groupby("Store")["Weekly_Sales"]
    .sum()
    .sort_values(ascending=False)
)

plt.figure(figsize=(12, 6))
store_sales.plot(kind="bar")

plt.title("Store Wise Total Sales")
plt.xlabel("Store")
plt.ylabel("Total Sales")

plt.tight_layout()

plt.savefig(
    "outputs/store_sales.png"
)

plt.close()

# Monthly Sales Trend
monthly_sales = (
    df.groupby("Month")["Weekly_Sales"]
    .mean()
)

plt.figure(figsize=(10, 5))

monthly_sales.plot(marker="o")

plt.title("Average Monthly Sales")
plt.xlabel("Month")
plt.ylabel("Average Sales")

plt.tight_layout()

plt.savefig(
    "outputs/monthly_sales.png"
)

plt.close()

# Business Insights
top_store = store_sales.idxmax()
best_month = monthly_sales.idxmax()

insights = f"""
BUSINESS INSIGHTS
=================

Top Performing Store:
Store {top_store}

Best Sales Month:
Month {best_month}

Key Findings:
-------------
1. Sales vary significantly across stores.
2. Holiday periods influence weekly sales.
3. Economic indicators affect sales performance.
4. Monthly trends help identify peak demand periods.
5. Random Forest captures non-linear sales patterns effectively.

Recommendation:
Use forecasting models to improve inventory planning and staffing decisions.
"""

with open(
    "insights/business_insights.txt",
    "w"
) as f:
    f.write(insights)

print("Project completed successfully.")
print("Check outputs and insights folders.")