from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib

# Load your trained model
model = joblib.load(r"C:\Users\Aseem\Desktop\GIT\EV-Vehicle-Battery Optimization\notebooks\energy_model.pkl")

app = FastAPI()

# Enable CORS for frontend
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

# Helper to preprocess input
def preprocess_input(data: TripInput):
    df = pd.DataFrame([data.dict()])

    # One-hot encode categorical columns
    df = pd.get_dummies(df, columns=["traffic", "driving", "mode"])

    # Ensure all expected columns exist (as used in training)
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

import math

@app.post("/optimize")
def optimize_trip(data: TripInput):
    # Preprocess input for ML model
    input_features = preprocess_input(data)
    energy_per_km_ml = float(model.predict(input_features)[0])  # kWh/km from ML

    # ----- Physics-based adjustment -----
    # Constants
    g = 9.81  # m/s²
    air_density = 1.225  # kg/m³
    C_d = 0.28  # Drag coefficient typical EV
    A = 2.2  # Frontal area m²
    C_r = 0.01  # Rolling resistance coefficient
    m_base = 1500  # Base vehicle mass kg
    aux_kWh_per_km = 0.02  # AC, electronics baseline

    # Payload effect
    total_mass = m_base + data.payload

    # Speed in m/s
    mode_speed = {"Eco": 55, "Normal": 70, "Sport": 85, "Custom": 65}  # km/h
    speed_kmh = mode_speed.get(data.mode, 65)
    v = speed_kmh * 1000 / 3600  # m/s

    # Terrain factor
    slope_factor = {"Uphill": 0.05, "Downhill": -0.03, "Highway": 0, "Intracity": 0}
    slope = slope_factor.get(data.driving, 0)

    # Traffic factor: affects acceleration/deceleration
    traffic_factor = {"Light": 0.95, "Medium": 1.0, "Heavy": 1.1}
    traffic_adj = traffic_factor.get(data.traffic, 1.0)

    # Rolling resistance
    E_rolling = C_r * total_mass * g * (1 + slope)  # N*m per meter

    # Aerodynamic drag
    E_aero = 0.5 * air_density * C_d * A * v**2  # W per m/s, approximate

    # Convert to kWh/km
    E_physics_per_km = ((E_rolling + E_aero) * 1000 / 3600000) * traffic_adj + aux_kWh_per_km

    # Temperature adjustment (battery efficiency drops in cold)
    if data.outside_temp < 10:
        temp_adj = 1 + (10 - data.outside_temp) * 0.02  # 2% per °C below 10
    else:
        temp_adj = 1.0

    # Combine ML prediction and physics adjustment
    energy_per_km = energy_per_km_ml * 0.7 + E_physics_per_km * 0.3
    energy_per_km *= temp_adj

    # Battery calculation
    battery_used = energy_per_km * data.distance_km
    battery_left = max(data.battery_percent - battery_used, 0)

    # Recommended settings
    recommended_settings = {
        "speed": speed_kmh,
        "ac": 22 if data.outside_temp > 25 else 25,
        "regen": "High" if data.driving == "Downhill" else "Medium",
        "acceleration_limit": "Low" if data.mode == "Eco" else "Medium"
    }

    # Charts data
    battery_vs_distance = [
        {"x": i, "y": max(data.battery_percent - energy_per_km * i, 0)}
        for i in range(0, int(data.distance_km)+1, max(1,int(data.distance_km/10)))
    ]
    speed_vs_consumption = [
        {"x": s, "y": ((0.5*air_density*C_d*A*(s*1000/3600)**2 + C_r*total_mass*g)/3600000 + aux_kWh_per_km)*temp_adj}
        for s in range(30, 101, 10)
    ]
    mode_comparison = [
        {"mode": m, "range": max(data.battery_percent - energy_per_km*(data.distance_km)*(speed_kmh/mode_speed[m]), 0)}
        for m in mode_speed
    ]
    energy_breakdown = {
        "AC": aux_kWh_per_km*data.distance_km,
        "Speed & Drag": (E_aero*data.distance_km)/3600000,
        "Rolling Resistance": (E_rolling*data.distance_km)/3600000,
        "Payload": total_mass * 0.0001 * data.distance_km,
        "Misc": 0.01*data.distance_km
    }
    projected_range = {"value": max(data.distance_km - battery_used*5, 0)}

    return {
        "recommended_settings": recommended_settings,
        "predicted_range": round(data.distance_km - battery_used*5, 2),
        "battery_left": round(battery_left, 2),
        "charts": {
            "battery_vs_distance": battery_vs_distance,
            "speed_vs_consumption": speed_vs_consumption,
            "mode_comparison": mode_comparison,
            "energy_breakdown": energy_breakdown,
            "projected_range": projected_range
        }
    }
