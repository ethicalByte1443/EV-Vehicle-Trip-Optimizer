from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np

# --------------------------
# Load trained ML model
# --------------------------
model = joblib.load(r"C:\Users\Aseem\Desktop\GIT\EV-Vehicle-Battery Optimization\models\rf_energy_model.pkl")

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
    speed_kmh: float
    acceleration_ms2: float
    battery_percent: float
    battery_voltage: float
    battery_temp: float
    driving_mode: int       # 1,2,3 (or your encoding)
    road_type: int          # 1,2,3
    traffic_condition: int  # 1,2,3
    slope_percent: float
    weather_condition: int  # 1,2,3
    temperature_c: float
    humidity_percent: float
    wind_speed_ms: float
    tire_pressure_psi: float
    vehicle_weight_kg: float
    distance_travelled_km: float  # <-- interpreted as planned trip distance

# --------------------------
# Preprocess input for ML
# --------------------------
def preprocess_input(data: TripInput):
    df = pd.DataFrame([{
        "Speed_kmh": data.speed_kmh,
        "Acceleration_ms2": data.acceleration_ms2,
        "Battery_State_%": data.battery_percent,
        "Battery_Voltage_V": data.battery_voltage,
        "Battery_Temperature_C": data.battery_temp,
        "Driving_Mode": int(data.driving_mode),
        "Road_Type": int(data.road_type),
        "Traffic_Condition": int(data.traffic_condition),
        "Slope_%": data.slope_percent,
        "Weather_Condition": int(data.weather_condition),
        "Temperature_C": data.temperature_c,
        "Humidity_%": data.humidity_percent,
        "Wind_Speed_ms": data.wind_speed_ms,
        "Tire_Pressure_psi": data.tire_pressure_psi,
        "Vehicle_Weight_kg": data.vehicle_weight_kg,
        "Distance_Travelled_km": data.distance_travelled_km
    }])

    expected_cols = [
        "Speed_kmh", "Acceleration_ms2", "Battery_State_%", "Battery_Voltage_V", "Battery_Temperature_C",
        "Driving_Mode", "Road_Type", "Traffic_Condition", "Slope_%", "Weather_Condition",
        "Temperature_C", "Humidity_%", "Wind_Speed_ms", "Tire_Pressure_psi",
        "Vehicle_Weight_kg", "Distance_Travelled_km"
    ]
    
    df = df[expected_cols]
    return df

# --------------------------
# Physics components & energy calculation
# --------------------------
def physics_components_per_km(data: TripInput):
    """
    Returns physics component contributions (kWh per km) as dict:
    { 'aero': ..., 'rolling': ..., 'aux': ..., 'payload': ..., 'misc': ... }
    """
    g = 9.81
    air_density = 1.225
    C_d = 0.28
    A = 2.2
    C_r = 0.01
    aux_kWh_per_km = 0.02
    misc_kWh_per_km = 0.01
    payload_factor = 0.0001  # kWh per kg per km (approx used earlier)

    total_mass = data.vehicle_weight_kg
    v = data.speed_kmh * 1000 / 3600  # m/s
    slope = data.slope_percent / 100.0
    # traffic codes: 1->Light,2->Medium,3->Heavy (adjust if your encoding differs)
    traffic_adj = {1: 0.95, 2: 1.0, 3: 1.1}.get(data.traffic_condition, 1.0)

    # Forces/energies in J per m, convert to kWh per km:
    E_rolling_J_per_m = C_r * total_mass * g * (1 + slope)  # N
    E_rolling_kWh_per_km = (E_rolling_J_per_m * 1000 / 3600000) * traffic_adj

    E_aero_J_per_m = 0.5 * air_density * C_d * A * v**2
    E_aero_kWh_per_km = (E_aero_J_per_m * 1000 / 3600000) * traffic_adj

    payload_kWh_per_km = payload_factor * total_mass

    # temperature adjustment (affects overall physics energy)
    temp_adj = 1.0 if data.temperature_c >= 10 else 1 + (10 - data.temperature_c) * 0.02

    return {
        "aero": E_aero_kWh_per_km * temp_adj,
        "rolling": E_rolling_kWh_per_km * temp_adj,
        "aux": aux_kWh_per_km * temp_adj,
        "payload": payload_kWh_per_km * temp_adj,
        "misc": misc_kWh_per_km * temp_adj
    }

def physics_energy_consumption(data: TripInput):
    comps = physics_components_per_km(data)
    # sum of physics derived components (kWh per km)
    total_physics_kWh_per_km = comps["aero"] + comps["rolling"] + comps["aux"] + comps["payload"] + comps["misc"]
    return total_physics_kWh_per_km

# --------------------------
# Helper to compute combined ML+physics for an arbitrary input dict
# --------------------------
def compute_combined_energy_per_km_from_dict(d: dict):
    # build TripInput (this validates types)
    ti = TripInput(**d)
    input_features = preprocess_input(ti)
    e_ml = float(model.predict(input_features)[0])
    e_phys = physics_energy_consumption(ti)
    return 0.7 * e_ml + 0.3 * e_phys

# --------------------------
# Intelligent recommended settings
# --------------------------
def intelligent_settings(data: TripInput, energy_per_km: float):
    # treat distance_travelled_km as planned trip distance (trip length)
    speed = data.speed_kmh
    if data.battery_percent < 40:
        speed -= 10
    elif data.battery_percent > 70 and data.distance_travelled_km < 50:
        speed += 15

    speed = max(30, min(speed, 120))
    ac = 22 if data.temperature_c < 25 else min(26, 20 + data.temperature_c * 0.3)
    regen = "High" if data.driving_mode == 3 or data.traffic_condition == 3 else "Medium"
    accel = "Low" if data.battery_percent < 30 or data.vehicle_weight_kg > 2000 else "Medium"

    return {"speed": round(speed), "ac": round(ac), "regen": regen, "acceleration_limit": accel}

# --------------------------
# Optimize trip endpoint
# --------------------------
@app.post("/optimize_trip")
def optimize_trip(data: TripInput):
    # Preprocess & base predictions (for the given planned trip scenario)
    input_features = preprocess_input(data)
    energy_per_km_ml = float(model.predict(input_features)[0])
    energy_per_km_physics = physics_energy_consumption(data)
    energy_per_km = 0.7 * energy_per_km_ml + 0.3 * energy_per_km_physics
    print(energy_per_km_ml)
    print(energy_per_km_physics)

    # Battery / range calculations (distance_travelled_km is the planned trip distance)
    battery_capacity_kWh = 50.0
    available_energy_kWh = (data.battery_percent / 100.0) * battery_capacity_kWh
    battery_used_kWh = energy_per_km * data.distance_travelled_km
    battery_left_kWh = max(available_energy_kWh - battery_used_kWh, 0.0)
    battery_left_percent = (battery_left_kWh / battery_capacity_kWh) * 100.0
    predicted_range_km = available_energy_kWh / energy_per_km if energy_per_km > 0 else float("inf")
    distance_warning = battery_left_kWh <= 0

    recommended = intelligent_settings(data, energy_per_km)

    # --------------------------
    # Charts (min 10 points) — ALL use the SAME combined ML+physics method
    # --------------------------
    trip_distance = float(data.distance_travelled_km)
    n_points = 10

    # 1) Battery vs Distance (remaining battery % along the planned trip)
    distances = np.linspace(0.0, trip_distance, n_points)
    battery_vs_distance = []
    # energy_per_km is constant under constant conditions; use it directly
    for d in distances:
        used_kWh = energy_per_km * d
        remaining_percent = max((available_energy_kWh - used_kWh) / battery_capacity_kWh * 100.0, 0.0)
        battery_vs_distance.append({"distance": float(d), "battery_percent": float(round(remaining_percent, 6))})

    # 2) Speed vs Consumption (recompute combined energy for each speed)
    speeds = np.linspace(30.0, 120.0, n_points)
    speed_vs_consumption = []
    base_dict = data.dict()
    for s in speeds:
        temp = dict(base_dict)
        temp["speed_kmh"] = float(s)
        epk = compute_combined_energy_per_km_from_dict(temp)
        speed_vs_consumption.append({"speed": float(round(s, 3)), "consumption": float(round(epk, 6))})

    # 3) Mode comparison — evaluate effective range when switching "mode speed"
    mode_speed_dict = {"Eco": 55.0, "Normal": 70.0, "Sport": 85.0, "Custom": 65.0}
    mode_comparison = []
    for m, spd in mode_speed_dict.items():
        temp = dict(base_dict)
        temp["speed_kmh"] = float(spd)
        epk = compute_combined_energy_per_km_from_dict(temp)
        available_energy_kWh = (data.battery_percent / 100.0) * battery_capacity_kWh
        predicted_range_by_mode = available_energy_kWh / epk if epk > 0 else float("inf")
        mode_comparison.append({"mode": m, "range": float(round(predicted_range_by_mode, 2))})

    # 4) Energy breakdown (physics-based components multiplied by trip length)
    comps_per_km = physics_components_per_km(data)
    energy_breakdown = [
        {"component": "AC", "energy_kWh": float(round(comps_per_km["aux"] * trip_distance, 6))},
        {"component": "Speed & Drag", "energy_kWh": float(round(comps_per_km["aero"] * trip_distance, 6))},
        {"component": "Rolling Resistance", "energy_kWh": float(round(comps_per_km["rolling"] * trip_distance, 6))},
        {"component": "Payload", "energy_kWh": float(round(comps_per_km["payload"] * trip_distance, 6))},
        {"component": "Misc", "energy_kWh": float(round(comps_per_km["misc"] * trip_distance, 6))}
    ]

    return {
        "recommended_settings": recommended,
        "predicted_range_km": float(round(predicted_range_km, 2)),
        "battery_left_percent": float(round(battery_left_percent, 2)),
        "distance_warning": bool(distance_warning),
        "charts": {
            "battery_vs_distance": battery_vs_distance,
            "speed_vs_consumption": speed_vs_consumption,
            "mode_comparison": mode_comparison,
            "energy_breakdown": energy_breakdown
        }
    }
