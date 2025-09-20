# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

# -----------------------------
# Step 1: Load Dataset
# -----------------------------
df = pd.read_csv("train_data.csv")  # tera CSV file

# -----------------------------
# Step 2: Preprocessing
# -----------------------------
# Categorical columns
cat_cols = ['traffic', 'driving', 'mode']
num_cols = ['distance_km', 'battery_percent', 'outside_temp', 'battery_temp', 'payload']

# One-Hot Encoding for categorical columns
df_encoded = pd.get_dummies(df, columns=cat_cols)

# Features and target
X = df_encoded.drop('energy_consumed_per_km', axis=1)
y = df_encoded['energy_consumed_per_km']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Step 3: Train Model
# -----------------------------
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -----------------------------
# Step 4: Evaluate
# -----------------------------
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Test MSE: {mse:.6f}")

# -----------------------------
# Step 5: Save Model
# -----------------------------
joblib.dump(model, "energy_model.pkl")
print("Model saved as energy_model.pkl")
