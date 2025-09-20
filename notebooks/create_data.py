# generate_dataset.py

import pandas as pd
import numpy as np

np.random.seed(42)

# Number of rows
n = 1000

# ------------------------
# Feature generation
# ------------------------
distance_km = np.random.uniform(5, 300, n)          # 5 to 300 km
battery_percent = np.random.uniform(10, 100, n)    # 10% to 100%
outside_temp = np.random.uniform(-10, 50, n)       # -10°C to 50°C
battery_temp = np.random.uniform(20, 60, n)        # 20°C to 60°C
traffic = np.random.choice(['Light', 'Medium', 'Heavy'], n)
driving = np.random.choice(['Highway', 'Intracity', 'Uphill', 'Downhill'], n)
mode = np.random.choice(['Eco', 'Normal', 'Sport', 'Custom'], n)
payload = np.random.uniform(0, 200, n)             # 0 to 200 kg

# ------------------------
# Target generation (energy_consumed_per_km)
# ------------------------
# Base consumption
base = 0.15  

# Map factors
traffic_map = {'Light': 0.95, 'Medium': 1.0, 'Heavy': 1.1}
driving_map = {'Highway': 0.95, 'Intracity': 1.05, 'Uphill': 1.2, 'Downhill': 0.9}
mode_map = {'Eco': 0.9, 'Normal': 1.0, 'Sport': 1.1, 'Custom': 1.0}

energy_consumed_per_km = [
    base * traffic_map[t] * driving_map[d] * mode_map[m] * (1 + b/100 + p*0.001)
    for t,d,m,b,p in zip(traffic, driving, mode, battery_temp, payload)
]

# ------------------------
# Create DataFrame
# ------------------------
df = pd.DataFrame({
    'distance_km': distance_km,
    'battery_percent': battery_percent,
    'outside_temp': outside_temp,
    'battery_temp': battery_temp,
    'traffic': traffic,
    'driving': driving,
    'mode': mode,
    'payload': payload,
    'energy_consumed_per_km': energy_consumed_per_km
})

# Save CSV
df.to_csv("train_data.csv", index=False)
print("Dataset saved as train_data.csv with 1000 rows")
