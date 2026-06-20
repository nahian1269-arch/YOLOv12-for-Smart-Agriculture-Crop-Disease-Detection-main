import hashlib
import json
import logging
import os
import pickle
import time
import uuid
from datetime import datetime, timezone

import cv2
import keras
import numpy as np
import requests
import supervision as sv
from flask import Flask, jsonify, redirect, render_template, request, send_from_directory, session, url_for
from keras import layers
from keras.saving import register_keras_serializable
from ultralytics import YOLO
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".matplotlib"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

APP_NAME = "NuroAgro"
UPLOAD_FOLDER = "static/uploads/"
DATA_FOLDER = "data"
LOCAL_DB_PATH = os.path.join(DATA_FOLDER, "nuroagro_state.json")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_PROJECT_MODEL = os.getenv("OPENAI_PROJECT_MODEL", "gpt-5.5")
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"

WEATHER_FEATURES = [
    ("max_temp", "Max Temperature", "C"),
    ("min_temp", "Min Temperature", "C"),
    ("avg_temp", "Average Temperature", "C"),
    ("humidity", "Humidity", "%"),
    ("pressure", "Pressure", "hPa"),
    ("rainfall", "Rainfall", "mm"),
]
WEATHER_INPUT_STEPS = 60
WEATHER_FORECAST_DAYS = 30
WEATHER_MODEL_PATH = "weather_prediction_transformer_model.keras"
WEATHER_SCALER_PATH = "scaler.pkl"

SENSOR_DEFINITIONS = [
    {"key": "dht11_temp", "label": "DHT11 Air Temp", "unit": "C", "ideal": "20-30"},
    {"key": "dht11_humidity", "label": "DHT11 Humidity", "unit": "%", "ideal": "55-75"},
    {"key": "dht7_temp", "label": "DHT7 Root Temp", "unit": "C", "ideal": "18-26"},
    {"key": "mq5", "label": "MQ-5 LPG/CH4", "unit": "ppm", "ideal": "<350"},
    {"key": "mq7", "label": "MQ-7 CO", "unit": "ppm", "ideal": "<80"},
    {"key": "mq135", "label": "MQ-135 Air Quality", "unit": "ppm", "ideal": "<600"},
    {"key": "lux", "label": "TEMT6000 Light", "unit": "lux", "ideal": "800-18000"},
    {"key": "rain_drop", "label": "Raindrop", "unit": "%", "ideal": "0"},
    {"key": "soil_moisture", "label": "Soil Moisture", "unit": "%", "ideal": "40-65"},
    {"key": "water_level", "label": "Water Reserve", "unit": "%", "ideal": ">35"},
    {"key": "motion", "label": "Motion", "unit": "", "ideal": "clear"},
]

HARDWARE_COMPONENTS = [
    "ESP-WROOM-32",
    "ESP camera module",
    "MQ-5 gas sensor",
    "MQ-7 carbon monoxide sensor",
    "MQ-135 air quality sensor",
    "DHT11 temperature and humidity",
    "DHT7 root-zone temperature",
    "TEMT6000 light sensor",
    "Raindrop sensor",
    "Soil moisture sensor",
    "2 water pumps",
    "Blue UV grow lights",
    "6V relay module",
    "Motion sensor",
]

FARMING_SYSTEMS = [
    "Hydroponic NFT rack",
    "Aquaponic tower",
    "Aeroponic column",
    "Hybrid hydro-aqua vertical system",
    "Traditional open-field bed",
    "Protected greenhouse bed",
]

PLANT_RECOMMENDATIONS = {
    "vertical": ["Lettuce", "Basil", "Spinach", "Mint", "Strawberry", "Pak choi"],
    "traditional": ["Rice", "Tomato", "Chili", "Eggplant", "Okra", "Cucumber"],
    "hybrid": ["Lettuce", "Coriander", "Tomato", "Spinach", "Kale", "Water spinach"],
}

FISH_RECOMMENDATIONS = ["Tilapia", "Koi", "Catfish fingerlings", "Guppy", "Molly"]

DISEASE_ACTIONS = {
    "Blight": "Remove affected leaves, improve airflow, and apply copper-based fungicide if spread continues.",
    "Brown spot": "Balance nitrogen and potassium, avoid overhead watering, and remove infected debris.",
    "Brown Spot": "Balance nitrogen and potassium, avoid overhead watering, and remove infected debris.",
    "False Smut": "Use clean seed, reduce excess nitrogen, and keep field humidity under control.",
    "Leaf Smut": "Remove infected leaves and keep foliage dry during evening hours.",
    "Rice blast": "Reduce leaf wetness, avoid dense canopy, and use a recommended blast-control fungicide.",
    "Stem Rot": "Improve drainage, remove infected residue, and avoid prolonged waterlogging.",
    "Tungro": "Control leafhopper vectors and remove infected plants quickly.",
    "Healthy": "No disease action needed. Keep monitoring environmental stress.",
}

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "nuroagro-local-dev-secret")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)


@register_keras_serializable()
class TransformerBlock(layers.Layer):
    """Custom layer required by the trained weather transformer model."""

    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "rate": self.rate,
        })
        return config


try:
    disease_model = YOLO("best.pt")
    logger.info("YOLO model loaded successfully")
    logger.info("Model class names: %s", disease_model.names)
except Exception as exc:
    disease_model = None
    logger.exception("Failed to load YOLO model: %s", exc)

try:
    with open(WEATHER_SCALER_PATH, "rb") as scaler_file:
        weather_scaler = pickle.load(scaler_file)
    weather_model = keras.models.load_model(
        WEATHER_MODEL_PATH,
        custom_objects={"TransformerBlock": TransformerBlock},
        compile=False,
    )
    logger.info("Weather transformer model loaded successfully from %s", WEATHER_MODEL_PATH)
    logger.info("Weather model input shape: %s", weather_model.input_shape)
    logger.info("Weather model output shape: %s", weather_model.output_shape)
except Exception as exc:
    weather_scaler = None
    weather_model = None
    logger.exception("Failed to load weather prediction model: %s", exc)


COLOR_MAP = {
    "Blight": sv.Color(255, 0, 0),
    "Brown Spot": sv.Color(0, 0, 255),
    "Brown spot": sv.Color(0, 0, 255),
    "False Smut": sv.Color(0, 255, 0),
    "Healthy": sv.Color(255, 255, 0),
    "Leaf Smut": sv.Color(128, 0, 128),
    "Rice blast": sv.Color(255, 165, 0),
    "Stem Rot": sv.Color(255, 0, 255),
    "Tungro": sv.Color(0, 255, 255),
    "Background": sv.Color(255, 255, 255),
}


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def hash_password(password):
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def default_sensor_reading(project_id="PRJ-001"):
    return {
        "id": f"SEN-{uuid.uuid4().hex[:8].upper()}",
        "project_id": project_id,
        "timestamp": now_iso(),
        "dht11_temp": 27.4,
        "dht11_humidity": 68,
        "dht7_temp": 23.1,
        "mq5": 148,
        "mq7": 21,
        "mq135": 412,
        "lux": 9200,
        "rain_drop": 0,
        "soil_moisture": 42,
        "water_level": 74,
        "motion": 0,
    }


def default_state():
    sensor_history = []
    for offset, moisture in enumerate([47, 44, 39, 31, 28, 45]):
        reading = default_sensor_reading()
        reading["id"] = f"SEN-DEMO-{offset + 1}"
        reading["soil_moisture"] = moisture
        reading["lux"] = 7300 + (offset * 420)
        reading["mq135"] = 390 + (offset * 18)
        reading["water_level"] = 78 - (offset * 3)
        sensor_history.append(reading)

    return {
        "admins": [
            {
                "id": "ADM-001",
                "name": "NuroAgro Admin",
                "email": "admin@nuroagro.local",
                "username": os.getenv("NUROAGRO_ADMIN_USERNAME", "admin"),
                "password_hash": hash_password(os.getenv("NUROAGRO_ADMIN_PASSWORD", "admin123")),
                "access_key_hash": hash_password(os.getenv("NUROAGRO_ADMIN_ACCESS_KEY", "NURO-ADMIN-2026")),
                "role": "admin",
                "status": "active",
            }
        ],
        "users": [
            {
                "id": "USR-001",
                "name": "Rahman Vertical Farm",
                "email": "rahman@example.com",
                "password_hash": generate_password_hash("farmer123"),
                "status": "accepted",
                "role": "farmer",
                "joined": "2026-06-01",
                "last_login": None,
                "location": {"label": "Dhaka, Bangladesh", "lat": 23.8103, "lng": 90.4125},
            },
            {
                "id": "USR-002",
                "name": "GreenRoof Hydro Lab",
                "email": "greenroof@example.com",
                "password_hash": generate_password_hash("farmer123"),
                "status": "pending",
                "role": "farmer",
                "joined": "2026-06-11",
                "last_login": None,
                "location": {"label": "Chattogram, Bangladesh", "lat": 22.3569, "lng": 91.7832},
            },
            {
                "id": "USR-003",
                "name": "North Bed Aquaponics",
                "email": "northbed@example.com",
                "password_hash": generate_password_hash("farmer123"),
                "status": "accepted",
                "role": "farmer",
                "joined": "2026-06-10",
                "last_login": None,
                "location": {"label": "Rajshahi, Bangladesh", "lat": 24.3745, "lng": 88.6042},
            },
        ],
        "visitors": [
            {"label": "Mon", "count": 42},
            {"label": "Tue", "count": 57},
            {"label": "Wed", "count": 51},
            {"label": "Thu", "count": 68},
            {"label": "Fri", "count": 77},
            {"label": "Sat", "count": 83},
            {"label": "Sun", "count": 64},
        ],
        "projects": [
            {
                "id": "PRJ-001",
                "owner_id": "USR-001",
                "name": "Nuro Tower A",
                "area": 850,
                "floors": 4,
                "lat": 23.8103,
                "lng": 90.4125,
                "weather_notes": "Humid monsoon roof site with partial morning sun",
                "goal": "hybrid vertical farming",
                "created_at": now_iso(),
            }
        ],
        "sensor_history": sensor_history,
        "controls": {
            "pump_1": "auto",
            "pump_2": "auto",
            "uv_lights": "auto",
            "camera": "active",
            "relay": "auto",
        },
        "disease_history": [],
        "activities": [],
    }


def load_state():
    if not os.path.exists(LOCAL_DB_PATH):
        state = default_state()
        save_state(state)
        return state

    with open(LOCAL_DB_PATH, "r", encoding="utf-8") as state_file:
        state = json.load(state_file)

    defaults = default_state()
    changed = False
    for key, value in defaults.items():
        if key not in state:
            state[key] = value
            changed = True

    default_admin = defaults["admins"][0]
    for admin in state.get("admins", []):
        if "access_key_hash" not in admin:
            admin["access_key_hash"] = default_admin["access_key_hash"]
            changed = True

    for user in state.get("users", []):
        if "password_hash" not in user:
            user["password_hash"] = generate_password_hash("farmer123")
            changed = True
        if "last_login" not in user:
            user["last_login"] = None
            changed = True

    if changed:
        save_state(state)
    return state


def save_state(state):
    with open(LOCAL_DB_PATH, "w", encoding="utf-8") as state_file:
        json.dump(state, state_file, indent=2)


def supabase_insert(table_name, row):
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        return False

    try:
        response = requests.post(
            f"{SUPABASE_URL}/rest/v1/{table_name}",
            headers={
                "apikey": SUPABASE_SERVICE_ROLE_KEY,
                "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
                "Content-Type": "application/json",
                "Prefer": "return=minimal",
            },
            json=row,
            timeout=5,
        )
        response.raise_for_status()
        return True
    except Exception as exc:
        logger.warning("Supabase insert failed for %s: %s", table_name, exc)
        return False


def latest_sensor(state):
    readings = state.get("sensor_history", [])
    if not readings:
        reading = default_sensor_reading()
        state.setdefault("sensor_history", []).append(reading)
        save_state(state)
        return reading
    return readings[-1]


def normalize_sensor_payload(payload):
    reading = default_sensor_reading(payload.get("project_id", "PRJ-001"))
    reading["id"] = payload.get("id") or f"SEN-{uuid.uuid4().hex[:8].upper()}"
    reading["timestamp"] = payload.get("timestamp") or now_iso()

    for sensor in SENSOR_DEFINITIONS:
        key = sensor["key"]
        if key in payload:
            try:
                reading[key] = float(payload[key])
            except (TypeError, ValueError):
                reading[key] = payload[key]

    reading["motion"] = 1 if str(reading.get("motion")).lower() in {"1", "true", "yes", "detected"} else 0
    return reading


def evaluate_automation(reading):
    alerts = []
    actions = {
        "pump_1": "hold",
        "pump_2": "hold",
        "uv_lights": "hold",
        "relay": "hold",
        "camera": "active",
    }

    soil = float(reading.get("soil_moisture", 0))
    water = float(reading.get("water_level", 0))
    lux = float(reading.get("lux", 0))
    rain = float(reading.get("rain_drop", 0))
    mq5 = float(reading.get("mq5", 0))
    mq7 = float(reading.get("mq7", 0))
    mq135 = float(reading.get("mq135", 0))

    if soil < 35 and water > 25:
        actions["pump_1"] = "auto_on"
        alerts.append("Soil is dry. Pump 1 should run until moisture reaches 45%.")
    elif soil >= 45:
        actions["pump_1"] = "auto_off"

    if soil < 35 and water <= 25:
        actions["pump_1"] = "blocked_low_water"
        alerts.append("Soil is dry but water reserve is low. Refill tank before pumping.")

    if water < 30:
        alerts.append("Water reserve is low.")

    if lux < 800:
        actions["uv_lights"] = "increase"
        alerts.append("Light intensity is low. Increase blue UV grow light output.")
    elif lux > 18000:
        actions["uv_lights"] = "reduce"
        alerts.append("Light intensity is high. Reduce grow light output or add shade.")

    if rain > 20:
        actions["pump_2"] = "off"
        alerts.append("Rain detected. Outdoor irrigation is paused.")

    if mq5 > 350 or mq7 > 80 or mq135 > 600:
        actions["relay"] = "ventilate"
        alerts.append("Gas or air quality level is high. Start ventilation and inspect farm.")

    if int(reading.get("motion", 0)) == 1:
        actions["camera"] = "record"
        alerts.append("Motion detected near the farm.")

    if not alerts:
        alerts.append("All core sensor values are within operational range.")

    return actions, alerts


def sensor_cards(reading):
    cards = []
    for sensor in SENSOR_DEFINITIONS:
        value = reading.get(sensor["key"], "-")
        status = "stable"
        if sensor["key"] == "soil_moisture" and float(value) < 35:
            status = "alert"
        elif sensor["key"] == "water_level" and float(value) < 30:
            status = "alert"
        elif sensor["key"] in {"mq5", "mq7", "mq135"} and float(value) > {"mq5": 350, "mq7": 80, "mq135": 600}[sensor["key"]]:
            status = "alert"
        elif sensor["key"] == "rain_drop" and float(value) > 20:
            status = "watch"
        elif sensor["key"] == "motion" and int(value) == 1:
            status = "alert"

        cards.append({
            "label": sensor["label"],
            "value": value,
            "unit": sensor["unit"],
            "ideal": sensor["ideal"],
            "status": status,
        })
    return cards


def build_automatic_weather_input():
    if weather_scaler is not None and hasattr(weather_scaler, "data_min_") and hasattr(weather_scaler, "data_max_"):
        seed_values = (weather_scaler.data_min_ + weather_scaler.data_max_) / 2
    else:
        seed_values = np.zeros(len(WEATHER_FEATURES), dtype=np.float32)
    return np.tile(np.array(seed_values, dtype=np.float32), (WEATHER_INPUT_STEPS, 1))


def predict_weather():
    if weather_model is None or weather_scaler is None:
        raise RuntimeError("Weather prediction model or scaler is not available")

    weather_input = build_automatic_weather_input()
    scaled_input = weather_scaler.transform(weather_input).reshape(1, WEATHER_INPUT_STEPS, len(WEATHER_FEATURES))
    prediction = weather_model.predict(scaled_input, verbose=0)
    forecast_scaled = prediction.reshape(WEATHER_FORECAST_DAYS, len(WEATHER_FEATURES))
    forecast = weather_scaler.inverse_transform(forecast_scaled)

    return [
        {
            "day": day_index + 1,
            "measurements": [
                {
                    "label": WEATHER_FEATURES[feature_index][1],
                    "unit": WEATHER_FEATURES[feature_index][2],
                    "value": round(float(forecast[day_index][feature_index]), 2),
                }
                for feature_index in range(len(WEATHER_FEATURES))
            ],
        }
        for day_index in range(WEATHER_FORECAST_DAYS)
    ]


def weather_average(forecast, label):
    values = []
    for day in forecast[:7]:
        for measurement in day["measurements"]:
            if measurement["label"] == label:
                values.append(float(measurement["value"]))
    return round(sum(values) / len(values), 2) if values else 0


def analyze_project(project, forecast):
    area = float(project.get("area", 0) or 0)
    floors = int(project.get("floors", 1) or 1)
    avg_temp = weather_average(forecast, "Average Temperature")
    avg_humidity = weather_average(forecast, "Humidity")
    avg_rainfall = weather_average(forecast, "Rainfall")

    vertical_score = 45
    traditional_score = 45
    if area <= 1200:
        vertical_score += 22
    else:
        traditional_score += 16
    if floors >= 2:
        vertical_score += 18
    if avg_rainfall > 8 or avg_humidity > 70:
        vertical_score += 12
    if avg_temp >= 20 and avg_temp <= 32:
        vertical_score += 6
        traditional_score += 6

    if vertical_score > traditional_score + 10:
        recommendation = "Vertical or hybrid farming is the best fit."
        plants = PLANT_RECOMMENDATIONS["vertical"]
    elif traditional_score > vertical_score + 10:
        recommendation = "Traditional farming is suitable, with protected beds recommended."
        plants = PLANT_RECOMMENDATIONS["traditional"]
    else:
        recommendation = "Both vertical and traditional farming can work here."
        plants = PLANT_RECOMMENDATIONS["hybrid"]

    return {
        "vertical_score": min(vertical_score, 100),
        "traditional_score": min(traditional_score, 100),
        "recommendation": recommendation,
        "plants": plants,
        "systems": FARMING_SYSTEMS,
        "fish": FISH_RECOMMENDATIONS,
        "reason": (
            f"Area {area:g} sq ft, {floors} floor option, forecast avg {avg_temp:g} C, "
            f"humidity {avg_humidity:g}%, rainfall {avg_rainfall:g} mm."
        ),
    }


def extract_response_text(response_data):
    if response_data.get("output_text"):
        return response_data["output_text"].strip()

    text_parts = []
    for output_item in response_data.get("output", []):
        for content_item in output_item.get("content", []):
            if content_item.get("type") == "output_text" and content_item.get("text"):
                text_parts.append(content_item["text"])
    return "\n".join(text_parts).strip()


def analyze_project_with_agent(project, forecast):
    local_analysis = analyze_project(project, forecast)
    local_analysis["agent_source"] = "NuroAgro local farm-fit engine"
    local_analysis["agent_report"] = (
        f"{local_analysis['recommendation']} {local_analysis['reason']} "
        f"Recommended goals include {', '.join(local_analysis['systems'][:4])}. "
        f"Suggested plants include {', '.join(local_analysis['plants'])}."
    )

    if not OPENAI_API_KEY:
        return local_analysis

    weather_context = {
        "average_temperature_c": weather_average(forecast, "Average Temperature"),
        "average_humidity_percent": weather_average(forecast, "Humidity"),
        "average_rainfall_mm": weather_average(forecast, "Rainfall"),
    }
    project_context = {
        "name": project["name"],
        "area_sq_ft": project["area"],
        "vertical_floors": project["floors"],
        "current_goal": project["goal"],
        "latitude": project["lat"],
        "longitude": project["lng"],
        "weather_forecast_summary": weather_context,
        "local_fit_analysis": local_analysis,
    }

    try:
        response = requests.post(
            OPENAI_RESPONSES_URL,
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": OPENAI_PROJECT_MODEL,
                "instructions": (
                    "You are NuroAgro's agricultural project advisor. Assess whether a site is suitable for "
                    "traditional, vertical, hydroponic, aquaponic, aeroponic, or hybrid farming. Give a concise "
                    "decision, major risks, the best farm goal, suitable crops, and suitable small fish when "
                    "aquaponics is relevant. Do not claim certainty beyond the supplied project and forecast data."
                ),
                "input": json.dumps(project_context),
                "max_output_tokens": 700,
            },
            timeout=45,
        )
        response.raise_for_status()
        agent_report = extract_response_text(response.json())
        if agent_report:
            local_analysis["agent_source"] = f"OpenAI {OPENAI_PROJECT_MODEL}"
            local_analysis["agent_report"] = agent_report
    except Exception as exc:
        logger.warning("OpenAI project analysis failed; using local fallback: %s", exc)

    return local_analysis


def build_recommendations(reading, forecast, disease_history):
    actions, alerts = evaluate_automation(reading)
    recommendations = list(alerts)

    avg_rainfall = weather_average(forecast, "Rainfall") if forecast else 0
    if avg_rainfall > 8:
        recommendations.append("High rainfall forecast. Favor protected vertical racks or covered beds.")

    if disease_history:
        latest_disease = disease_history[-1]
        if latest_disease.get("summary"):
            top_disease = next(iter(latest_disease["summary"].keys()))
            recommendations.append(DISEASE_ACTIONS.get(top_disease, "Review the latest disease image and isolate affected plants."))

    if actions.get("uv_lights") == "increase":
        recommendations.append("Set blue UV lights to the next intensity step for leafy greens.")

    return recommendations[:6]


def admin_stats(state):
    users = state.get("users", [])
    projects = state.get("projects", [])
    accepted = [user for user in users if user.get("status") == "accepted"]
    pending = [user for user in users if user.get("status") == "pending"]
    visitors = sum(day.get("count", 0) for day in state.get("visitors", []))
    return {
        "registered_users": len(users),
        "accepted_users": len(accepted),
        "pending_users": len(pending),
        "total_visitors": visitors,
        "active_projects": len(projects),
    }


def map_points(state):
    points = []
    for user in state.get("users", []):
        location = user.get("location", {})
        lat = float(location.get("lat", 0))
        lng = float(location.get("lng", 0))
        points.append({
            "label": location.get("label", user.get("name", "Unknown")),
            "status": user.get("status", "pending"),
            "x": max(4, min(96, (lng + 180) / 360 * 100)),
            "y": max(6, min(94, (90 - lat) / 180 * 100)),
        })
    return points


def record_activity(state, actor_type, actor_id, action, details=""):
    state.setdefault("activities", []).append({
        "id": f"ACT-{uuid.uuid4().hex[:8].upper()}",
        "timestamp": now_iso(),
        "actor_type": actor_type,
        "actor_id": actor_id,
        "action": action,
        "details": details,
    })
    state["activities"] = state["activities"][-500:]


def record_visitor(state):
    label = datetime.now().strftime("%a")
    for item in state.setdefault("visitors", []):
        if item.get("label") == label:
            item["count"] = int(item.get("count", 0)) + 1
            return
    state["visitors"].append({"label": label, "count": 1})


def authenticate_admin(state, username, password, access_key=""):
    password_hash = hash_password(password or "")
    access_key_hash = hash_password(access_key or "")
    for admin in state.get("admins", []):
        password_match = admin.get("username") == username and admin.get("password_hash") == password_hash
        key_match = bool(access_key) and admin.get("access_key_hash") == access_key_hash
        if password_match or key_match:
            return admin
    return None


def current_admin(state):
    admin_id = session.get("admin_id")
    for admin in state.get("admins", []):
        if admin.get("id") == admin_id:
            return admin
    return None


def authenticate_user(state, email, password):
    normalized_email = (email or "").strip().lower()
    for user in state.get("users", []):
        if user.get("email", "").lower() != normalized_email:
            continue
        if check_password_hash(user.get("password_hash", ""), password or ""):
            return user
    return None


def current_user(state):
    user_id = session.get("user_id")
    for user in state.get("users", []):
        if user.get("id") == user_id:
            return user
    return None


def build_admin_context(state, admin=None, error=None, notice=None):
    return {
        "app_name": APP_NAME,
        "admin": admin,
        "error": error,
        "notice": notice,
        "admin_stats": admin_stats(state),
        "admins": state.get("admins", []),
        "users": state.get("users", []),
        "visitors": state.get("visitors", []),
        "map_points": map_points(state),
        "projects": state.get("projects", []),
        "activities": list(reversed(state.get("activities", [])[-50:])),
        "sensor_count": len(state.get("sensor_history", [])),
        "disease_count": len(state.get("disease_history", [])),
        "supabase_enabled": bool(SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY),
    }


def create_project_from_form(form, owner_id):
    area = float(form.get("area", 0) or 0)
    if area <= 0:
        raise ValueError("Farm area must be greater than zero.")

    floors = max(1, min(4, int(form.get("floors", 1) or 1)))
    lat = form.get("lat", "").strip()
    lng = form.get("lng", "").strip()
    if not lat or not lng:
        raise ValueError("Location permission is required before creating a project.")

    latitude = float(lat)
    longitude = float(lng)
    if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
        raise ValueError("The detected geographic coordinates are invalid.")

    return {
        "id": f"PRJ-{uuid.uuid4().hex[:6].upper()}",
        "owner_id": owner_id,
        "name": form.get("project_name", "New NuroAgro Project").strip() or "New NuroAgro Project",
        "area": area,
        "floors": floors,
        "lat": latitude,
        "lng": longitude,
        "goal": form.get("goal", "hybrid vertical farming").strip(),
        "created_at": now_iso(),
    }


def process_image(image_path):
    if disease_model is None:
        return None, "Disease model is not available", {}, []

    try:
        image = cv2.imread(image_path)
        if image is None:
            return None, "Error: Could not load image", {}, []

        image = cv2.resize(image, (1280, 720))
        results = disease_model(image)[0]
        detections = sv.Detections.from_ultralytics(results)

        disease_counts = {}
        if len(detections) > 0:
            for class_id in detections.class_id:
                class_name = results.names[class_id]
                disease_counts[class_name] = disease_counts.get(class_name, 0) + 1

        annotated_image = image.copy()
        if len(detections) > 0:
            for detection_idx, xyxy in enumerate(detections.xyxy):
                class_id = detections.class_id[detection_idx]
                class_name = results.names[class_id]
                color = COLOR_MAP.get(class_name, sv.Color(128, 128, 128))
                single_detection = sv.Detections(
                    xyxy=np.array([xyxy]),
                    class_id=np.array([class_id]),
                    confidence=np.array([detections.confidence[detection_idx]]),
                )
                box_annotator = sv.BoxAnnotator(color=color)
                label_annotator = sv.LabelAnnotator(
                    color=color,
                    text_color=sv.Color(255, 255, 255),
                    text_position=sv.Position.TOP_LEFT,
                )
                annotated_image = box_annotator.annotate(scene=annotated_image, detections=single_detection)
                labels = [f"{class_name}: {single_detection.confidence[0]:.2f}"]
                annotated_image = label_annotator.annotate(
                    scene=annotated_image,
                    detections=single_detection,
                    labels=labels,
                )

        y_offset = 30
        for disease, count in disease_counts.items():
            cv2.putText(
                annotated_image,
                f"{disease}: {count}",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            y_offset += 30

        timestamp = int(time.time())
        output_filename = os.path.join(app.config["UPLOAD_FOLDER"], f"output_{timestamp}.jpg")
        cv2.imwrite(output_filename, annotated_image)
        output_path = f"uploads/output_{timestamp}.jpg"
        recommendations = [
            DISEASE_ACTIONS.get(disease, "Inspect the affected area and isolate unhealthy plants.")
            for disease in disease_counts
        ] or ["No disease detected. Keep monitoring leaf color, humidity, and airflow."]
        return output_path, None, disease_counts, recommendations

    except Exception as exc:
        logger.exception("Error processing image: %s", exc)
        return None, f"Error: {str(exc)}", {}, []


def build_context(state, user, disease_result=None, error=None, setup_result=None):
    weather_error = None
    try:
        weather_forecast = predict_weather()
    except Exception as exc:
        weather_forecast = []
        weather_error = str(exc)
        logger.error("Weather prediction failed: %s", exc)

    reading = latest_sensor(state)
    automation, alerts = evaluate_automation(reading)
    projects = [project for project in state.get("projects", []) if project.get("owner_id") == user.get("id")]
    current_project = projects[-1] if projects else {
        "id": "",
        "name": "No project created",
        "area": 0,
        "floors": 1,
        "lat": user.get("location", {}).get("lat", ""),
        "lng": user.get("location", {}).get("lng", ""),
        "goal": "hybrid vertical farming",
    }
    project_analysis = current_project.get("analysis")
    if not project_analysis and projects and weather_forecast:
        project_analysis = analyze_project(current_project, weather_forecast)
    project_analysis = project_analysis or {
        "vertical_score": 0,
        "traditional_score": 0,
        "recommendation": "Create your first project to receive a farm-fit recommendation.",
        "plants": [],
        "systems": [],
        "fish": [],
        "reason": "Project area, goal, floors, and location are required.",
        "agent_source": "",
        "agent_report": "",
    }
    project_ids = {project.get("id") for project in projects}
    disease_history = [
        record for record in state.get("disease_history", [])
        if record.get("project_id") in project_ids
    ]

    return {
        "app_name": APP_NAME,
        "user": user,
        "error": error,
        "setup_result": setup_result,
        "disease_result": disease_result,
        "hardware_components": HARDWARE_COMPONENTS,
        "weather_features": WEATHER_FEATURES,
        "weather_forecast": weather_forecast,
        "weather_error": weather_error,
        "sensor_cards": sensor_cards(reading),
        "latest_sensor": reading,
        "automation": automation,
        "alerts": alerts,
        "recommendations": build_recommendations(reading, weather_forecast, disease_history),
        "projects": projects,
        "has_projects": bool(projects),
        "user_metrics": {
            "project_count": len(projects),
            "disease_count": len(disease_history),
            "area": current_project.get("area", 0),
            "floors": current_project.get("floors", 0),
            "status": user.get("status", "pending"),
        },
        "current_project": current_project,
        "project_analysis": project_analysis,
        "controls": state.get("controls", {}),
        "disease_history": list(reversed(disease_history[-5:])),
        "supabase_enabled": bool(SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY),
        "sample_camera_image": url_for("static", filename="uploads/images_7.jpg"),
    }


@app.route("/favicon.ico")
def favicon():
    return send_from_directory(os.path.join(app.root_path, "static"), "favicon.ico", mimetype="image/vnd.microsoft.icon")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    state = load_state()
    error = None
    notice = None

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        phone = request.form.get("phone", "").strip()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not name or not email or not password:
            error = "Name, email, and password are required."
        elif any(user.get("email", "").lower() == email for user in state.get("users", [])):
            error = "An account already exists for this email."
        elif len(password) < 6:
            error = "Password must contain at least 6 characters."
        elif password != confirm_password:
            error = "Passwords do not match."
        else:
            user = {
                "id": f"USR-{uuid.uuid4().hex[:8].upper()}",
                "name": name,
                "email": email,
                "phone": phone,
                "password_hash": generate_password_hash(password),
                "status": "pending",
                "role": "farmer",
                "joined": datetime.now().date().isoformat(),
                "last_login": None,
                "location": {"label": "Not captured", "lat": 0, "lng": 0},
            }
            state.setdefault("users", []).append(user)
            record_activity(state, "user", user["id"], "signup", "Account submitted for admin approval.")
            save_state(state)
            supabase_insert("users", {key: value for key, value in user.items() if key != "password_hash"})
            notice = "Account created. An admin must approve it before you can log in."

    return render_template("signup.html", app_name=APP_NAME, error=error, notice=notice)


@app.route("/login", methods=["GET", "POST"])
def login():
    state = load_state()
    error = None

    if current_user(state):
        return redirect(url_for("index"))

    if request.method == "POST":
        user = authenticate_user(
            state,
            request.form.get("email", ""),
            request.form.get("password", ""),
        )
        if not user:
            error = "Invalid email or password."
        elif user.get("status") != "accepted":
            error = "Your account is waiting for admin approval."
        else:
            session.clear()
            session["user_id"] = user["id"]
            user["last_login"] = now_iso()
            record_visitor(state)
            record_activity(state, "user", user["id"], "login", "User logged in.")
            save_state(state)
            return redirect(url_for("index"))

    return render_template("login.html", app_name=APP_NAME, error=error)


@app.route("/logout", methods=["POST"])
def logout():
    state = load_state()
    user = current_user(state)
    if user:
        record_activity(state, "user", user["id"], "logout", "User logged out.")
        save_state(state)
    session.pop("user_id", None)
    return redirect(url_for("login"))


@app.route("/", methods=["GET", "POST"])
def index():
    state = load_state()
    user = current_user(state)
    if not user or user.get("status") != "accepted":
        session.pop("user_id", None)
        return redirect(url_for("login"))

    error = None
    setup_result = None
    disease_result = None

    if request.method == "POST":
        form_type = request.form.get("form_type", "")

        if form_type == "project_setup":
            try:
                project = create_project_from_form(request.form, user["id"])
                forecast = predict_weather()
                setup_result = analyze_project_with_agent(project, forecast)
                project["analysis"] = setup_result
                state.setdefault("projects", []).append(project)
                user["location"] = {
                    "label": f"{project['lat']:.5f}, {project['lng']:.5f}",
                    "lat": project["lat"],
                    "lng": project["lng"],
                }
                record_activity(state, "user", user["id"], "project_created", project["name"])
                save_state(state)
                supabase_insert("projects", project)
            except (TypeError, ValueError) as exc:
                error = str(exc)
            except Exception as exc:
                logger.exception("Project creation failed: %s", exc)
                error = "Project analysis failed. Please try again."

        elif form_type == "control_update":
            control_name = request.form.get("control_name")
            control_value = request.form.get("control_value")
            if control_name and control_value:
                state.setdefault("controls", {})[control_name] = control_value
                record_activity(state, "user", user["id"], "control_update", f"{control_name}={control_value}")
                save_state(state)

        elif form_type == "disease":
            file = request.files.get("file")
            if not file or file.filename == "":
                error = "No selected disease image"
            elif not allowed_file(file.filename):
                error = "Only PNG, JPG, and JPEG images are supported"
            else:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                file.save(filepath)
                output_path, processing_error, disease_counts, disease_recommendations = process_image(filepath)
                if processing_error:
                    error = processing_error
                else:
                    disease_result = {
                        "image": url_for("static", filename=output_path),
                        "summary": disease_counts,
                        "recommendations": disease_recommendations,
                    }
                    record = {
                        "id": f"DIS-{uuid.uuid4().hex[:8].upper()}",
                        "timestamp": now_iso(),
                        "project_id": next(
                            (
                                project.get("id")
                                for project in reversed(state.get("projects", []))
                                if project.get("owner_id") == user["id"]
                            ),
                            "",
                        ),
                        "image": disease_result["image"],
                        "summary": disease_counts,
                        "recommendations": disease_recommendations,
                    }
                    state.setdefault("disease_history", []).append(record)
                    record_activity(state, "user", user["id"], "disease_scan", filename)
                    save_state(state)
                    supabase_insert("disease_detections", record)

    return render_template(
        "index.html",
        **build_context(
            state,
            user,
            disease_result=disease_result,
            error=error,
            setup_result=setup_result,
        ),
    )


@app.route("/admin", methods=["GET", "POST"])
def admin_dashboard():
    state = load_state()
    error = None
    notice = None
    admin = current_admin(state)

    if request.method == "POST":
        form_type = request.form.get("form_type", "")

        if form_type == "admin_login":
            admin = authenticate_admin(
                state,
                request.form.get("username", ""),
                request.form.get("password", ""),
                request.form.get("access_key", ""),
            )
            if admin:
                session.clear()
                session["admin_id"] = admin["id"]
                notice = "Admin login successful."
                record_activity(state, "admin", admin["id"], "login", "Admin logged in.")
                save_state(state)
            else:
                error = "Invalid admin credentials or access key."

        elif form_type == "admin_logout":
            if admin:
                record_activity(state, "admin", admin["id"], "logout", "Admin logged out.")
                save_state(state)
            session.pop("admin_id", None)
            admin = None
            notice = "Admin logged out."

        elif admin and form_type == "admin_action":
            user_id = request.form.get("user_id")
            action = request.form.get("admin_action")
            users = state.get("users", [])
            if action == "accept":
                for user in users:
                    if user.get("id") == user_id:
                        user["status"] = "accepted"
                        notice = f"Accepted {user.get('name', 'user')}."
                        record_activity(state, "admin", admin["id"], "user_accepted", user.get("email", ""))
                        break
            elif action == "delete":
                deleted_user = next((user for user in users if user.get("id") == user_id), None)
                state["users"] = [user for user in users if user.get("id") != user_id]
                if deleted_user:
                    notice = f"Deleted {deleted_user.get('name', 'user')}."
                    record_activity(state, "admin", admin["id"], "user_deleted", deleted_user.get("email", ""))
            save_state(state)

        elif not admin:
            error = "Admin login is required."

    return render_template("admin.html", **build_admin_context(state, admin=admin, error=error, notice=notice))


@app.route("/api/sensors", methods=["POST"])
def api_sensors():
    state = load_state()
    payload = request.get_json(silent=True) or request.form.to_dict()
    reading = normalize_sensor_payload(payload)
    actions, alerts = evaluate_automation(reading)
    state.setdefault("sensor_history", []).append(reading)
    state["sensor_history"] = state["sensor_history"][-500:]
    save_state(state)
    supabase_insert("sensor_readings", reading)
    return jsonify({
        "ok": True,
        "stored": reading,
        "automation": actions,
        "alerts": alerts,
    })


@app.route("/api/status", methods=["GET"])
def api_status():
    state = load_state()
    reading = latest_sensor(state)
    actions, alerts = evaluate_automation(reading)
    return jsonify({
        "app": APP_NAME,
        "latest_sensor": reading,
        "automation": actions,
        "alerts": alerts,
        "supabase_enabled": bool(SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY),
    })


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
