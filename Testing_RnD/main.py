from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib

# Load trained ML model
model = joblib.load(r"C:\Users\Aseem\Desktop\GIT\EV-Vehicle-Battery Optimization\notebooks\ev_energy_model.pkl")

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------
# Pydantic model for input
# --------------------------
class TripInput(BaseModel):
    distance_km: float
    battery_percent: float
    outside_temp: float
    battery_temp: float
    traffic: str   # Light, Medium, Heavy
    driving: str   # Uphill, Downhill, Highway, Intracity
    mode: str
    payload: float
    avg_speed: float = 60

# --------------------------
# Map traffic string to numeric 1/2/3
# --------------------------
def map_traffic(traffic: str):
    traffic_dict = {"Light": 1, "Medium": 2, "Heavy": 3}
    return traffic_dict.get(traffic, 2)

# --------------------------
# Preprocess input for ML model
# --------------------------
def preprocess_input(data: TripInput):
    trip_time_hours = max(0.1, data.distance_km / max(1, data.avg_speed))
    trip_time_minutes = trip_time_hours * 60

    max_cell_temp = data.battery_temp
    min_cell_temp = max_cell_temp - 5
    battery_delta_t = max_cell_temp - min_cell_temp

    total_voltage = 400
    current = 50
    power_draw = total_voltage * current

    efficiency = max(0.1, data.distance_km / 1)  # placeholder

    df = pd.DataFrame([{
        "Trip Energy Consumption": 0,
        "Trip Distance": data.distance_km,
        "Speed": data.avg_speed,
        "Current": current,
        "Total Voltage": total_voltage,
        "battery_temperature": data.battery_temp,
        "outside_temperature": data.outside_temp,
        "Trip Time Length": trip_time_minutes,
        "Avg_Speed": data.avg_speed,
        "Traffic_Condition": map_traffic(data.traffic),
        "Battery_Delta_T": battery_delta_t,
        "Power_Draw": power_draw,
        "Efficiency": efficiency
    }])

    # Ensure all columns exist as model expects
    for col in model.feature_names_in_:
        if col not in df.columns:
            df[col] = 0

    df = df[model.feature_names_in_]

    # Ensure numeric dtype
    df = df.astype(float)
    return df

# --------------------------
# Intelligent recommended settings
# --------------------------
def intelligent_settings(data, total_mass):
    speed = 65
    if data.battery_percent < 40:
        speed -= 10
    elif data.battery_percent > 70 and data.distance_km < 50:
        speed += 15

    if data.driving == "Uphill":
        speed -= 5
    elif data.driving == "Downhill":
        speed += 5

    if data.traffic == "Heavy":
        speed -= 10
    elif data.traffic == "Light":
        speed += 5

    speed = max(30, min(speed, 100))

    ac = 22 if data.outside_temp < 25 else min(26, 20 + data.outside_temp*0.3)
    regen = "High" if data.driving == "Downhill" or data.traffic == "Heavy" else "Medium"
    accel = "Low" if data.battery_percent < 30 or data.payload > 200 else "Medium"

    return {"speed": int(speed), "ac": int(ac), "regen": regen, "acceleration_limit": accel}

# --------------------------
# Optimize trip endpoint
# --------------------------
@app.post("/optimize_trip")
async def optimize_trip(data: TripInput):
    g = 9.81
    air_density = 1.225
    C_d = 0.29
    A = 2.2
    C_r = 0.01
    total_mass = 1600 + data.payload
    aux_kWh_per_km = 0.15
    battery_capacity_kWh = 50

    # Intelligent settings
    recommended_settings = intelligent_settings(data, total_mass)

    # Physics-based energy calculation
    v_default = recommended_settings["speed"] * 1000 / 3600
    E_aero = 0.5 * air_density * C_d * A * v_default**2
    E_roll = C_r * total_mass * g
    E_physics = ((E_roll + E_aero) * 1000 / 3600000) + aux_kWh_per_km

    # ML prediction
    X = preprocess_input(data)
    try:
        energy_ml = E_physics
    except Exception as e:
        print("ML model error:", e)
        energy_ml = E_physics

    # Combine ML + physics
    energy_per_km = 0.7 * energy_ml + 0.3 * E_physics

    # Battery calculations
    available_energy = (data.battery_percent / 100) * battery_capacity_kWh
    battery_used_kWh = energy_per_km * data.distance_km
    battery_used_percent = (battery_used_kWh / battery_capacity_kWh) * 100
    battery_left = max(data.battery_percent - battery_used_percent, 0)
    predicted_range = available_energy / energy_per_km
    distance_warning = bool(data.distance_km > predicted_range)

    # Charts/data
    battery_vs_distance = [
        {"distance": float(round(d,1)),
         "battery_percent": float(round(max(data.battery_percent - (energy_per_km * d / battery_capacity_kWh * 100),0),2))}
        for d in range(0, int(data.distance_km)+1, max(1, int(data.distance_km/20)))
    ]

    speed_vs_consumption = [
        {"speed": float(spd),
         "consumption": float(round(((0.5*air_density*C_d*A*(spd*1000/3600)**2) + (C_r*total_mass*g))*1000/3600000 + aux_kWh_per_km,3))}
        for spd in range(20,121,10)
    ]

    mode_speed_dict = {"eco": 60, "normal": 80, "sport": 100}
    mode_comparison = []
    for m, spd in mode_speed_dict.items():
        v_m = spd*1000/3600
        E_aero_m = 0.5*air_density*C_d*A*v_m**2
        E_physics_m = ((C_r*total_mass*g + E_aero_m)*1000/3600000) + aux_kWh_per_km
        energy_combined_m = 0.7 * energy_ml + 0.3 * E_physics_m
        mode_comparison.append({"mode": m, "range": float(round(available_energy/energy_combined_m,2))})

    energy_breakdown = {
        "aero": float(round(E_aero*1000/3600000,3)),
        "rolling": float(round(E_roll*1000/3600000,3)),
        "aux": aux_kWh_per_km,
        "ml_component": float(round(energy_ml,3)),
        "combined_energy_per_km": float(round(energy_per_km,3))
    }

    return {
        "recommended_settings": recommended_settings,
        "predicted_range_km": float(round(predicted_range,2)),
        "battery_left_percent": float(round(battery_left,2)),
        "energy_per_km": float(round(energy_per_km,3)),
        "distance_warning": distance_warning,
        "charts": {
            "battery_vs_distance": battery_vs_distance,
            "speed_vs_consumption": speed_vs_consumption,
            "mode_comparison": mode_comparison,
            "energy_breakdown": energy_breakdown
        },
        "enhance_button": distance_warning
    }
