import requests
import random

# --------------------------
# Generate dummy data
# --------------------------
dummy_data = {
    "distance_km": random.uniform(5, 100),
    "battery_percent": random.uniform(10, 100),
    "outside_temp": random.uniform(5, 35),
    "battery_temp": random.uniform(20, 40),
    "traffic": random.choice(["Light", "Medium", "Heavy"]),
    "driving": random.choice(["Highway", "Intracity", "Uphill", "Downhill"]),
    "mode": random.choice(["Eco", "Normal", "Sport", "Custom"]),
    "payload": random.uniform(0, 300)  # kg
}

# --------------------------
# Send to API
# --------------------------
API_URL = "http://127.0.0.1:8001/optimize_trip"  # your API port

response = requests.post(API_URL, json=dummy_data)

if response.status_code == 200:
    print("✅ Response from API:")
    print(response.json())
else:
    print(f"❌ Error: {response.status_code}")
    print(response.text)
