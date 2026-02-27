import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# Load dataset
df = pd.read_csv("Hyderabad_IceCream_2024_2025_With_Sales.csv")

# Features & Target
X = df[["Temperature", "Month", "Weekend", "Holiday"]]
y = df["Sales"]

# Train model
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    random_state=42
)

model.fit(X, y)

# Evaluate
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)
mae = mean_absolute_error(y, y_pred)

print("Model Trained Successfully")
print("R2 Score:", r2)
print("MAE:", mae)

# Save model
joblib.dump(model, "sales_model.pkl")
print("Model Saved as sales_model.pkl")