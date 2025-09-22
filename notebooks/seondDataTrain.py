import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv(r"C:\Users\Aseem\Desktop\GIT\EV-Vehicle-Battery Optimization\notebooks\EV_Energy_Consumption_Dataset.csv")

df.drop(columns=["Vehicle_ID", "Timestamp"], inplace=True)

df.info()

# Filter dataset for voltage 300-400V
voltage_range_df = df[(df['Battery_Voltage_V'] >= 300) & (df['Battery_Voltage_V'] <= 400)]

# Speed range
speed_min = voltage_range_df['Speed_kmh'].min()
speed_max = voltage_range_df['Speed_kmh'].max()

# Acceleration range
acc_min = voltage_range_df['Acceleration_ms2'].min()
acc_max = voltage_range_df['Acceleration_ms2'].max()

# Driving Mode distribution
driving_mode_counts = voltage_range_df['Driving_Mode'].value_counts()

print(f"Voltage 300-400V me Speed range: {speed_min} - {speed_max} km/h")
print(f"Voltage 300-400V me Acceleration range: {acc_min} - {acc_max} m/s²")
print("Driving Mode distribution:\n", driving_mode_counts)

# Create a new column 'Energy_Consumption_kWh_per_km' by dividing 'Energy_Consumption_kWh' by 'Distance_Travelled_km'
df['Energy_Consumption_kWh_per_km'] = df['Energy_Consumption_kWh'] / df['Distance_Travelled_km']

# Replace 'Energy_Consumption_kWh' column with 'Energy_Consumption_kWh_per_km'
df['Energy_Consumption_kWh'] = df['Energy_Consumption_kWh_per_km']

# Drop the temporary column 'Energy_Consumption_kWh_per_km'
df.drop(columns=['Energy_Consumption_kWh_per_km'], inplace=True)



df.info()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
# Define features and target
X = df.drop('Energy_Consumption_kWh', axis=1)
y = df['Energy_Consumption_kWh']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R2 Score: {r2}")

import os
model_filename = r"C:\Users\Aseem\Desktop\GIT\EV-Vehicle-Battery Optimization\models\rf_energy_model.pkl"
os.makedirs(os.path.dirname(model_filename), exist_ok=True)
joblib.dump(model, model_filename)

print(f"Model saved to {model_filename}")