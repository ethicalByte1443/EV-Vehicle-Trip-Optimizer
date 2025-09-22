import requests
import random

# --------------------------
# Generate dummy data
# --------------------------
dummy_data = {
    "speed_kmh": random.uniform(40, 100),
    "acceleration_ms2": random.uniform(0.5, 3.0),
    "battery_percent": random.uniform(10, 20),
    "battery_voltage": random.uniform(350, 400),
    "battery_temp": random.uniform(20, 40),
    "driving_mode": random.randint(1, 3),      # 1,2,3
    "road_type": random.randint(1, 3),         # 1,2,3
    "traffic_condition": random.randint(1, 3), # 1,2,3
    "slope_percent": random.uniform(-5, 5),
    "weather_condition": random.randint(1, 3), # 1,2,3
    "temperature_c": random.uniform(5, 35),
    "humidity_percent": random.uniform(20, 90),
    "wind_speed_ms": random.uniform(0, 10),
    "tire_pressure_psi": random.uniform(30, 35),
    "vehicle_weight_kg": random.uniform(1200, 2000),
    "distance_travelled_km": random.uniform(5, 100)
}

# --------------------------
# Send to API
# --------------------------
API_URL = "http://127.0.0.1:8000/optimize_trip"  # change to /enhance_trip if needed

response = requests.post(API_URL, json=dummy_data)
print(dummy_data)
if response.status_code == 200:
    print("✅ Response from API:")
    print(response.json())
else:
    print(f"❌ Error: {response.status_code}")
    print(response.text)
