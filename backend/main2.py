from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib

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
    driving_mode: int       # 1,2,3
    road_type: int          # 1,2,3
    traffic_condition: int  # 1,2,3
    slope_percent: float
    weather_condition: int  # 1,2,3
    temperature_c: float
    humidity_percent: float
    wind_speed_ms: float
    tire_pressure_psi: float
    vehicle_weight_kg: float
    distance_travelled_km: float

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
        "Driving_Mode": int(data.driving_mode),        # 1,2,3
        "Road_Type": int(data.road_type),              # 1,2,3
        "Traffic_Condition": int(data.traffic_condition), # 1,2,3
        "Slope_%": data.slope_percent,
        "Weather_Condition": int(data.weather_condition), # 1,2,3
        "Temperature_C": data.temperature_c,
        "Humidity_%": data.humidity_percent,
        "Wind_Speed_ms": data.wind_speed_ms,
        "Tire_Pressure_psi": data.tire_pressure_psi,
        "Vehicle_Weight_kg": data.vehicle_weight_kg,
        "Distance_Travelled_km": data.distance_travelled_km
    }])

    # Ensure column order matches the ML model
    expected_cols = [
        "Speed_kmh", "Acceleration_ms2", "Battery_State_%", "Battery_Voltage_V", "Battery_Temperature_C",
        "Driving_Mode", "Road_Type", "Traffic_Condition", "Slope_%", "Weather_Condition",
        "Temperature_C", "Humidity_%", "Wind_Speed_ms", "Tire_Pressure_psi",
        "Vehicle_Weight_kg", "Distance_Travelled_km"
    ]
    
    df = df[expected_cols]
    return df


# --------------------------
# Physics-based energy calculation
# --------------------------
def physics_energy_consumption(data: TripInput):
    g = 9.81
    air_density = 1.225
    C_d = 0.28
    A = 2.2
    C_r = 0.01
    aux_kWh_per_km = 0.02

    total_mass = data.vehicle_weight_kg
    v = data.speed_kmh * 1000 / 3600  # m/s
    slope = data.slope_percent / 100
    traffic_adj = {"Light": 0.95, "Medium": 1.0, "Heavy": 1.1}.get(data.traffic_condition, 1.0)

    E_rolling = C_r * total_mass * g * (1 + slope)
    E_aero = 0.5 * air_density * C_d * A * v**2
    E_physics_per_km = ((E_rolling + E_aero) * 1000 / 3600000) * traffic_adj + aux_kWh_per_km

    temp_adj = 1.0 if data.temperature_c >= 10 else 1 + (10 - data.temperature_c) * 0.02
    E_physics_per_km *= temp_adj

    return E_physics_per_km

# --------------------------
# Intelligent recommended settings
# --------------------------
def intelligent_settings(data, energy_per_km):
    speed = data.speed_kmh
    if data.battery_percent < 40:
        speed -= 10
    elif data.battery_percent > 70 and data.distance_travelled_km < 50:
        speed += 15

    if data.driving_mode == "Uphill":
        speed -= 5
    elif data.driving_mode == "Downhill":
        speed += 5

    if data.traffic_condition == "Heavy":
        speed -= 10
    elif data.traffic_condition == "Light":
        speed += 5

    speed = max(30, min(speed, 120))
    ac = 22 if data.temperature_c < 25 else min(26, 20 + data.temperature_c*0.3)
    regen = "High" if data.driving_mode == "Downhill" or data.traffic_condition == "Heavy" else "Medium"
    accel = "Low" if data.battery_percent < 30 or data.vehicle_weight_kg > 2000 else "Medium"

    return {"speed": round(speed), "ac": round(ac), "regen": regen, "acceleration_limit": accel}

# --------------------------
# Optimize trip endpoint
# --------------------------
@app.post("/optimize_trip")
def optimize_trip(data: TripInput):
    input_features = preprocess_input(data)
    energy_per_km_ml = float(model.predict(input_features)[0])
    energy_per_km_physics = physics_energy_consumption(data)

    energy_per_km = 0.7 * energy_per_km_ml + 0.3 * energy_per_km_physics

    battery_capacity_kWh = 50
    available_energy = (data.battery_percent / 100) * battery_capacity_kWh
    battery_used = energy_per_km * data.distance_travelled_km
    battery_left_percent = max(0, (available_energy - battery_used) / battery_capacity_kWh * 100)
    predicted_range_km = available_energy / energy_per_km
    distance_warning = battery_left_percent <= 0

    recommended_settings = intelligent_settings(data, energy_per_km)

    return {
        "recommended_settings": recommended_settings,
        "predicted_range_km": round(predicted_range_km, 2),
        "battery_left_percent": round(battery_left_percent, 2),
        "distance_warning": distance_warning
    }

# --------------------------
# Enhance trip endpoint
# --------------------------
@app.post("/enhance_trip")
def enhance_trip(data: TripInput):
    input_features = preprocess_input(data)
    energy_per_km_ml = float(model.predict(input_features)[0])
    battery_capacity_kWh = 50

    def calculate_trip(speed, ac, regen, accel):
        v = speed * 1000 / 3600
        E_aero = 0.5 * 1.225 * 0.28 * 2.2 * v**2
        E_rolling = 0.01 * data.vehicle_weight_kg * 9.81 * (1 + data.slope_percent / 100)
        E_physics = ((E_rolling + E_aero) * 1000 / 3600000) * {"Light": 0.95, "Medium": 1.0, "Heavy": 1.1}.get(data.traffic_condition, 1.0) + 0.02

        energy_per_km = 0.5 * energy_per_km_ml + 0.5 * E_physics

        if ac == "OFF":
            energy_per_km -= 0.01
        elif isinstance(ac, int):
            energy_per_km += 0.01

        if regen == "High":
            energy_per_km *= 0.95
        if accel == "Low":
            energy_per_km *= 0.97

        available_energy = (data.battery_percent / 100) * battery_capacity_kWh
        possible_range = available_energy / energy_per_km
        energy_required = data.distance_travelled_km * energy_per_km
        required_battery_percent = (energy_required / battery_capacity_kWh) * 100
        expected_battery_percent = max(data.battery_percent - required_battery_percent - 5, 0)

        return {
            "enhanced_settings": {"mode": "Eco", "speed": speed, "ac": ac, "regen": regen, "acceleration_limit": accel},
            "possible_range_km": round(possible_range, 2),
            "expected_battery_after_trip": round(expected_battery_percent, 2),
            "required_battery_percent": round(required_battery_percent, 2),
            "energy_per_km": round(energy_per_km, 3)
        }

    candidates = []
    for speed in [45, 50, 55]:
        for ac in ["OFF", 20, 26]:
            for regen in ["High"]:
                for accel in ["Low", "Medium"]:
                    res = calculate_trip(speed, ac, regen, accel)
                    if res["possible_range_km"] >= data.distance_travelled_km:
                        candidates.append(res)

    if candidates:
        best = max(candidates, key=lambda x: x["expected_battery_after_trip"])
        best["status"] = "✅ Trip feasible with optimised settings"
        return best
    else:
        worst_case = calculate_trip(45, "OFF", "High", "Low")
        min_charge_needed_percent = (data.distance_travelled_km * worst_case["energy_per_km"]) / battery_capacity_kWh * 100
        worst_case["status"] = "⚠️ Trip not possible with current charge"
        worst_case["min_charge_needed_percent"] = round(min_charge_needed_percent, 2)
        return worst_case
