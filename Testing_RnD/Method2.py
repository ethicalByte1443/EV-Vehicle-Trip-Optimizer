import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib

# --------------------------
# Load dataset
# --------------------------
df = pd.read_csv("processed_ev_dataset.csv")

# Map Traffic_Condition to numeric
traffic_map = {"Light": 1, "Medium": 2, "Heavy": 3}
df["Traffic_Condition"] = df["Traffic_Condition"].map(traffic_map)

# Numeric columns (including Traffic_Condition now)
numeric_cols = ["Trip Distance","Speed","Current","Total Voltage",
                "battery_temperature","outside_temperature",
                "Trip Time Length","Avg_Speed","Traffic_Condition",
                "Battery_Delta_T","Power_Draw","Efficiency"]

# Ensure numeric dtype
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

# Fill missing values
df.fillna(0, inplace=True)

# Target column
target = "Trip Energy Consumption"
X = df[numeric_cols]
y = df[target]

# Preprocessing pipeline
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_cols)
])

# Models
models = {
    "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=200, random_state=42)
}

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
results = {}

for name, model in models.items():
    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("model", model)
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    
    results[name] = {
        "R2": r2_score(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred))
    }

# Show results
for model_name, metrics in results.items():
    print(f"\n📊 {model_name} Performance:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

# Save pipeline
joblib.dump(pipe, "ev_energy_model.pkl")
print("\n✅ Model pipeline saved as 'ev_energy_model.pkl'")
