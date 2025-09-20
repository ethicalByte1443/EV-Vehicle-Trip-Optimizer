from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib

# Load trained ML model
model = joblib.load(r"C:\Users\Aseem\Desktop\GIT\EV-Vehicle-Battery Optimization\notebooks\energy_model.pkl")

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model for input
class TripInput(BaseModel):
    distance_km: float
    battery_percent: float
    outside_temp: float
    battery_temp: float
    traffic: str
    driving: str
    mode: str
    payload: float

# Preprocess input for ML model
def preprocess_input(data: TripInput):
    df = pd.DataFrame([data.dict()])
    df = pd.get_dummies(df, columns=["traffic", "driving", "mode"])

    expected_cols = [
        "distance_km", "battery_percent", "outside_temp", "battery_temp", "payload",
        "traffic_Heavy", "traffic_Light", "traffic_Medium",
        "driving_Downhill", "driving_Highway", "driving_Intracity", "driving_Uphill",
        "mode_Custom", "mode_Eco", "mode_Normal", "mode_Sport"
    ]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0

    df = df[expected_cols]
    return df

# Intelligent recommended settings
def intelligent_settings(data, energy_per_km, total_mass):
    # Speed based on battery & payload & traffic & terrain
    base_speed = 65
    speed = base_speed

    if data.battery_percent < 40:
        speed -= 10
    elif data.battery_percent > 70 and data.distance_km < 50:
        speed += 15

    # Terrain & traffic adjustments
    if data.driving == "Uphill":
        speed -= 5
    elif data.driving == "Downhill":
        speed += 5
    if data.traffic == "Heavy":
        speed -= 10
    elif data.traffic == "Light":
        speed += 5
    speed = max(30, min(speed, 100))

    # AC adjustment
    ac = 22 if data.outside_temp < 25 else min(26, 20 + data.outside_temp*0.3)

    # Regen
    if data.driving == "Downhill" or data.traffic == "Heavy":
        regen = "High"
    else:
        regen = "Medium"

    # Acceleration limit
    if data.battery_percent < 30 or data.payload > 200:
        accel = "Low"
    else:
        accel = "Medium"

    return {"speed": round(speed), "ac": round(ac), "regen": regen, "acceleration_limit": accel}

import math

@app.post("/optimize")
def optimize_trip(data: TripInput):
    input_features = preprocess_input(data)
    energy_per_km_ml = float(model.predict(input_features)[0])

    # Physics-based constants
    g = 9.81
    air_density = 1.225
    C_d = 0.28
    A = 2.2
    C_r = 0.01
    m_base = 1500
    aux_kWh_per_km = 0.02

    total_mass = m_base + data.payload
    mode_speed_dict = {"Eco": 55, "Normal": 70, "Sport": 85, "Custom": 65}
    speed_kmh = mode_speed_dict.get(data.mode, 65)
    v = speed_kmh * 1000 / 3600

    slope_factor = {"Uphill": 0.05, "Downhill": -0.03, "Highway": 0, "Intracity": 0}
    slope = slope_factor.get(data.driving, 0)

    traffic_factor = {"Light": 0.95, "Medium": 1.0, "Heavy": 1.1}
    traffic_adj = traffic_factor.get(data.traffic, 1.0)

    E_rolling = C_r * total_mass * g * (1 + slope)
    E_aero = 0.5 * air_density * C_d * A * v**2

    E_physics_per_km = ((E_rolling + E_aero) * 1000 / 3600000) * traffic_adj + aux_kWh_per_km

    temp_adj = 1.0 if data.outside_temp >= 10 else 1 + (10 - data.outside_temp) * 0.02

    energy_per_km = energy_per_km_ml * 0.7 + E_physics_per_km * 0.3
    energy_per_km *= temp_adj

    battery_used = energy_per_km * data.distance_km
    battery_left = max(data.battery_percent - battery_used, 0)

    recommended_settings = intelligent_settings(data, energy_per_km, total_mass)

    # Battery vs distance chart
    battery_vs_distance = [
        {"x": i, "y": max(data.battery_percent - energy_per_km * i, 0)}
        for i in range(0, int(data.distance_km)+1, max(1, int(data.distance_km/10)))
    ]

    # Speed vs consumption chart
    speed_vs_consumption = [
        {"x": s, "y": ((0.5*air_density*C_d*A*(s*1000/3600)**2 + C_r*total_mass*g)/3600000 + aux_kWh_per_km)*temp_adj}
        for s in range(30, 101, 10)
    ]

    # Mode comparison chart
    mode_comparison = [
        {"mode": m, "range": max(data.battery_percent - energy_per_km*(data.distance_km)*(speed_kmh/mode_speed_dict[m]), 0)}
        for m in mode_speed_dict
    ]

    energy_breakdown = {
        "AC": aux_kWh_per_km*data.distance_km,
        "Speed & Drag": (E_aero*data.distance_km)/3600000,
        "Rolling Resistance": (E_rolling*data.distance_km)/3600000,
        "Payload": total_mass * 0.0001 * data.distance_km,
        "Misc": 0.01*data.distance_km
    }

    # Check if distance can be achieved
    distance_warning = battery_left <= 0

    return {
        "recommended_settings": recommended_settings,
        "predicted_range": round(data.distance_km - battery_used, 2),
        "battery_left": round(battery_left, 2),
        "distance_warning": distance_warning,
        "charts": {
            "battery_vs_distance": battery_vs_distance,
            "speed_vs_consumption": speed_vs_consumption,
            "mode_comparison": mode_comparison,
            "energy_breakdown": energy_breakdown
        },
        "enhance_button": distance_warning  # frontend can show "Enhance" if True
    }
