import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import joblib

# -------------------------------
# 1. Load and Clean the Data
# -------------------------------
df = pd.read_csv("gold_price_dataset_extended.csv", index_col=0, parse_dates=True)
print("Missing values per column:")
print(df.isnull().sum())

# Drop rows with missing values
df.dropna(inplace=True)

# -------------------------------
# 2. Preprocess the Data
# -------------------------------
features = ['usd_index', 'sp500', 'crude_oil', 'silver_price', 'vix', '10Y_treasury_yield']
target = 'gold_price'

X = df[features].values
y = df[target].values

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler for later use in Django
joblib.dump(scaler, "scaler.joblib")
print("Scaler saved as scaler.joblib")

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# -------------------------------
# 3. Train Multiple Models
# -------------------------------
models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "SVR": SVR()
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {"MSE": mse, "R2": r2}
    print(f"{name}: MSE = {mse:.2f}, R2 = {r2:.2f}")

# Optionally, you could choose to inspect the results here.

# -------------------------------
# 4. Save All Models
# -------------------------------
# Save the dictionary of models
joblib.dump(models, "all_models.joblib")
print("All models saved as all_models.joblib")
