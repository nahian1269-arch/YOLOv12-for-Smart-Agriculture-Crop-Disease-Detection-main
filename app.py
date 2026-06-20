import hashlib
import json
import logging
import os
import pickle
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from functools import wraps

import cv2
import keras
import numpy as np
import requests
import supervision as sv
from flask import Flask, Response, jsonify, redirect, render_template, request, session, url_for
from keras import layers
from keras.saving import register_keras_serializable
from ultralytics import YOLO
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".matplotlib"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_env_file(path=".env"):
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


load_env_file()

APP_NAME = "NuroAgro"
UPLOAD_FOLDER = "static/uploads/"
DATA_FOLDER = "data"
LOCAL_DB_PATH = os.path.join(DATA_FOLDER, "nuroagro_state.json")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
SUPABASE_URL = os.getenv("SUPABASE_URL", os.getenv("VITE_SUPABASE_URL", "")).rstrip("/")
SUPABASE_PUBLISHABLE_KEY = os.getenv(
    "SUPABASE_PUBLISHABLE_KEY",
    os.getenv("VITE_SUPABASE_PUBLISHABLE_KEY", ""),
)
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
SUPABASE_API_KEY = SUPABASE_SERVICE_ROLE_KEY or SUPABASE_PUBLISHABLE_KEY
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_PROJECT_MODEL = os.getenv("OPENAI_PROJECT_MODEL", "gpt-5.5")
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
WEATHER_REFRESH_MINUTES = 15
SEASONAL_ANALYSIS_DAYS = 75
SENSOR_ANALYSIS_HOURS = 2
SENSOR_ANALYSIS_MODEL = "NuroAgro Sensor Intelligence v1"

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

PLANT_HEALTHY_CLASSES = {
    "Apple leaf",
    "Bell_pepper leaf",
    "Blueberry leaf",
    "Cherry leaf",
    "Peach leaf",
    "Potato leaf",
    "Raspberry leaf",
    "Soyabean leaf",
    "Strawberry leaf",
    "Tomato leaf",
    "grape leaf",
}

PLANT_DISEASE_GUIDANCE = {
    "scab": "Remove infected leaves and fruit debris, improve airflow, and use an approved scab fungicide.",
    "rust": "Remove heavily infected leaves, avoid overhead watering, and apply an approved rust treatment.",
    "leaf spot": "Remove infected foliage, keep leaves dry, improve spacing, and use a crop-appropriate fungicide.",
    "bacterial spot": "Avoid handling wet plants, sanitize tools, remove infected tissue, and use an approved copper treatment.",
    "early blight": "Remove lower infected leaves, mulch soil, improve airflow, and apply an approved blight fungicide.",
    "late blight": "Isolate affected plants immediately, remove infected tissue, avoid leaf wetness, and apply an approved late-blight treatment.",
    "blight": "Remove affected foliage, reduce leaf wetness, improve airflow, and apply a crop-appropriate blight treatment.",
    "powdery mildew": "Improve ventilation, reduce humidity around leaves, remove infected foliage, and use an approved mildew treatment.",
    "mosaic virus": "Isolate and remove infected plants, disinfect tools, and control aphid or whitefly vectors.",
    "yellow virus": "Remove infected plants and control whiteflies to reduce virus transmission.",
    "mold": "Reduce humidity, improve airflow, remove infected leaves, and avoid overhead irrigation.",
    "spider mites": "Inspect leaf undersides, isolate affected plants, wash foliage, and apply insecticidal soap or a suitable miticide.",
    "black rot": "Remove infected leaves and fruit, sanitize tools, improve airflow, and apply an approved black-rot treatment.",
}

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "nuroagro-local-dev-secret")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
STATE_LOCK = threading.RLock()


def state_transaction(view_function):
    @wraps(view_function)
    def wrapped(*args, **kwargs):
        with STATE_LOCK:
            return view_function(*args, **kwargs)
    return wrapped


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
    rice_disease_model = YOLO("best.pt")
    logger.info("Rice disease YOLO model loaded successfully")
    logger.info("Rice model class names: %s", rice_disease_model.names)
except Exception as exc:
    rice_disease_model = None
    logger.exception("Failed to load rice disease YOLO model: %s", exc)

try:
    plant_disease_model = YOLO("plant.pt")
    logger.info("Multi-plant disease YOLO model loaded successfully")
    logger.info("Plant model class names: %s", plant_disease_model.names)
except Exception as exc:
    plant_disease_model = None
    logger.exception("Failed to load multi-plant disease YOLO model: %s", exc)

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
        "weather_snapshots": [],
        "weather_predictions": [],
        "seasonal_analyses": [],
        "sensor_analyses": [],
    }


def load_state():
    with STATE_LOCK:
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
    with STATE_LOCK:
        temp_path = f"{LOCAL_DB_PATH}.tmp"
        with open(temp_path, "w", encoding="utf-8") as state_file:
            json.dump(state, state_file, indent=2)
        os.replace(temp_path, LOCAL_DB_PATH)


def supabase_insert(table_name, row):
    if not SUPABASE_URL or not SUPABASE_API_KEY:
        return False

    try:
        response = requests.post(
            f"{SUPABASE_URL}/rest/v1/{table_name}",
            headers={
                "apikey": SUPABASE_API_KEY,
                "Authorization": f"Bearer {SUPABASE_API_KEY}",
                "Content-Type": "application/json",
                "Prefer": "return=minimal",
            },
            json=row,
            timeout=8,
        )
        response.raise_for_status()
        return True
    except Exception as exc:
        logger.warning("Supabase insert failed for %s: %s", table_name, exc)
        return False


def supabase_configured():
    return bool(SUPABASE_URL and SUPABASE_API_KEY)


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


SENSOR_ANALYSIS_KEYS = [
    "dht11_temp",
    "dht11_humidity",
    "dht7_temp",
    "mq5",
    "mq7",
    "mq135",
    "lux",
    "rain_drop",
    "soil_moisture",
    "water_level",
    "motion",
]


def sensor_window_readings(state, project_id, hours=SENSOR_ANALYSIS_HOURS):
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    readings = []
    for reading in state.get("sensor_history", []):
        if project_id and reading.get("project_id") != project_id:
            continue
        timestamp = parse_iso_datetime(reading.get("timestamp"))
        if timestamp and timestamp.astimezone(timezone.utc) >= cutoff:
            readings.append(reading)
    return readings


def numeric_average(readings, key):
    values = []
    for reading in readings:
        try:
            values.append(float(reading.get(key, 0) or 0))
        except (TypeError, ValueError):
            continue
    return round(sum(values) / len(values), 2) if values else 0


def sensor_trend(readings, key):
    if len(readings) < 2:
        return 0
    midpoint = max(1, len(readings) // 2)
    early = numeric_average(readings[:midpoint], key)
    late = numeric_average(readings[midpoint:], key)
    return round(late - early, 2)


def analyze_sensor_window(state, project_id):
    readings = sensor_window_readings(state, project_id)
    if not readings:
        return None

    averages = {key: numeric_average(readings, key) for key in SENSOR_ANALYSIS_KEYS}
    trends = {key: sensor_trend(readings, key) for key in SENSOR_ANALYSIS_KEYS}
    anomalies = []
    feedback = []
    score = 100

    checks = [
        ("soil_moisture", averages["soil_moisture"] < 35, 18, "Soil moisture stayed below 35%. Irrigation is required."),
        ("water_level", averages["water_level"] < 30, 18, "Water reserve stayed below 30%. Refill the tank."),
        ("lux_low", averages["lux"] < 800, 10, "Average light intensity is low. Increase grow-light output."),
        ("lux_high", averages["lux"] > 18000, 8, "Average light intensity is high. Reduce UV output or add shade."),
        ("mq5", averages["mq5"] > 350, 12, "MQ-5 gas level is elevated. Inspect gas sources and ventilate."),
        ("mq7", averages["mq7"] > 80, 15, "Carbon monoxide trend is unsafe. Ventilate and inspect immediately."),
        ("mq135", averages["mq135"] > 600, 12, "Air-quality level is poor. Increase ventilation."),
        ("temperature", averages["dht11_temp"] < 18 or averages["dht11_temp"] > 34, 10, "Air temperature is outside the preferred farm range."),
        ("humidity", averages["dht11_humidity"] < 45 or averages["dht11_humidity"] > 85, 8, "Humidity is outside the preferred range."),
        ("motion", averages["motion"] > 0.2, 5, "Repeated motion was detected during the analysis window."),
    ]
    for code, triggered, penalty, message in checks:
        if triggered:
            anomalies.append(code)
            feedback.append(message)
            score -= penalty

    if trends["soil_moisture"] < -8:
        anomalies.append("soil_drying_fast")
        feedback.append("Soil moisture is falling quickly. Check pump timing and irrigation flow.")
        score -= 8
    if trends["water_level"] < -12:
        anomalies.append("water_drop_fast")
        feedback.append("Water reserve is dropping quickly. Inspect for leakage or high pump usage.")
        score -= 8
    if trends["mq135"] > 120:
        anomalies.append("air_quality_worsening")
        feedback.append("Air quality is worsening across the two-hour window.")
        score -= 8

    score = max(0, min(100, score))
    risk_level = "critical" if score < 45 else "watch" if score < 75 else "stable"
    if not feedback:
        feedback.append("Two-hour sensor patterns are stable. Continue automatic monitoring.")

    generated_at = datetime.now(timezone.utc)
    record = {
        "id": f"SAN-{uuid.uuid4().hex[:10].upper()}",
        "project_id": project_id,
        "generated_at": generated_at.isoformat(),
        "next_analysis_at": (generated_at + timedelta(hours=SENSOR_ANALYSIS_HOURS)).isoformat(),
        "window_start": readings[0].get("timestamp"),
        "window_end": readings[-1].get("timestamp"),
        "sample_count": len(readings),
        "health_score": score,
        "risk_level": risk_level,
        "averages": averages,
        "trends": trends,
        "anomalies": anomalies,
        "feedback": feedback,
        "model_name": SENSOR_ANALYSIS_MODEL,
    }
    state.setdefault("sensor_analyses", []).append(record)
    supabase_insert("sensor_analyses", record)
    return record


def get_or_create_sensor_analysis(state, project_id):
    analyses = [
        item for item in state.get("sensor_analyses", [])
        if item.get("project_id") == project_id
    ]
    if analyses:
        latest = analyses[-1]
        next_analysis_at = parse_iso_datetime(latest.get("next_analysis_at"))
        if next_analysis_at and datetime.now(timezone.utc) < next_analysis_at.astimezone(timezone.utc):
            return latest
    return analyze_sensor_window(state, project_id)


def sensor_analysis_scheduler():
    while True:
        try:
            with STATE_LOCK:
                state = load_state()
                project_ids = {
                    reading.get("project_id")
                    for reading in state.get("sensor_history", [])
                    if reading.get("project_id")
                }
                project_ids.update(
                    project.get("id")
                    for project in state.get("projects", [])
                    if project.get("id")
                )
                analysis_count = len(state.get("sensor_analyses", []))
                for project_id in project_ids:
                    get_or_create_sensor_analysis(state, project_id)
                if len(state.get("sensor_analyses", [])) != analysis_count:
                    save_state(state)
        except Exception as exc:
            logger.exception("Two-hour sensor analysis scheduler failed: %s", exc)
        time.sleep(60)


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


WEATHER_CODE_LABELS = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Rime fog",
    51: "Light drizzle",
    53: "Drizzle",
    55: "Heavy drizzle",
    61: "Light rain",
    63: "Rain",
    65: "Heavy rain",
    71: "Light snow",
    73: "Snow",
    75: "Heavy snow",
    80: "Rain showers",
    81: "Rain showers",
    82: "Heavy showers",
    95: "Thunderstorm",
    96: "Thunderstorm with hail",
    99: "Severe thunderstorm",
}


def parse_iso_datetime(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def project_coordinates(user, project):
    lat = float(project.get("lat", 0) or 0)
    lng = float(project.get("lng", 0) or 0)
    if lat or lng:
        return lat, lng, "Project location"

    location = user.get("location", {})
    lat = float(location.get("lat", 0) or 0)
    lng = float(location.get("lng", 0) or 0)
    if lat or lng:
        return lat, lng, location.get("label", "Account location")

    return 23.8103, 90.4125, "Dhaka fallback"


def recent_weather_bundle(state, user_id, project_id):
    snapshots = [
        item for item in state.get("weather_snapshots", [])
        if item.get("user_id") == user_id and item.get("project_id") == project_id
    ]
    if not snapshots:
        return None

    latest = snapshots[-1]
    observed_at = parse_iso_datetime(latest.get("observed_at"))
    if observed_at and datetime.now(timezone.utc) - observed_at.astimezone(timezone.utc) < timedelta(minutes=WEATHER_REFRESH_MINUTES):
        return latest.get("bundle")
    return None


def latest_weather_bundle(state, user_id, project_id):
    snapshots = [
        item for item in state.get("weather_snapshots", [])
        if item.get("user_id") == user_id and item.get("project_id") == project_id
    ]
    return snapshots[-1].get("bundle") if snapshots else None


def aggregate_hourly_by_date(hourly):
    date_values = {}
    times = hourly.get("time", [])
    humidities = hourly.get("relative_humidity_2m", [])
    pressures = hourly.get("surface_pressure", [])
    for index, timestamp in enumerate(times):
        date_key = str(timestamp)[:10]
        bucket = date_values.setdefault(date_key, {"humidity": [], "pressure": []})
        if index < len(humidities) and humidities[index] is not None:
            bucket["humidity"].append(float(humidities[index]))
        if index < len(pressures) and pressures[index] is not None:
            bucket["pressure"].append(float(pressures[index]))

    return {
        date_key: {
            "humidity": round(sum(values["humidity"]) / len(values["humidity"]), 2) if values["humidity"] else 0,
            "pressure": round(sum(values["pressure"]) / len(values["pressure"]), 2) if values["pressure"] else 1013,
        }
        for date_key, values in date_values.items()
    }


def fetch_location_weather(latitude, longitude):
    response = requests.get(
        OPEN_METEO_FORECAST_URL,
        params={
            "latitude": latitude,
            "longitude": longitude,
            "current": ",".join([
                "temperature_2m",
                "relative_humidity_2m",
                "apparent_temperature",
                "precipitation",
                "rain",
                "weather_code",
                "cloud_cover",
                "surface_pressure",
                "wind_speed_10m",
            ]),
            "daily": ",".join([
                "weather_code",
                "temperature_2m_max",
                "temperature_2m_min",
                "temperature_2m_mean",
                "precipitation_sum",
                "sunrise",
                "sunset",
            ]),
            "hourly": "relative_humidity_2m,surface_pressure",
            "past_days": 59,
            "forecast_days": 7,
            "timezone": "auto",
        },
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json()
    current = payload.get("current", {})
    daily = payload.get("daily", {})
    hourly_by_date = aggregate_hourly_by_date(payload.get("hourly", {}))

    daily_rows = []
    daily_times = daily.get("time", [])
    for index, date_value in enumerate(daily_times):
        weather_code = int(daily.get("weather_code", [0] * len(daily_times))[index] or 0)
        hourly_values = hourly_by_date.get(date_value, {})
        daily_rows.append({
            "date": date_value,
            "condition": WEATHER_CODE_LABELS.get(weather_code, "Variable weather"),
            "weather_code": weather_code,
            "max_temp": float(daily.get("temperature_2m_max", [0] * len(daily_times))[index] or 0),
            "min_temp": float(daily.get("temperature_2m_min", [0] * len(daily_times))[index] or 0),
            "avg_temp": float(daily.get("temperature_2m_mean", [0] * len(daily_times))[index] or 0),
            "humidity": float(hourly_values.get("humidity", current.get("relative_humidity_2m", 0)) or 0),
            "pressure": float(hourly_values.get("pressure", current.get("surface_pressure", 1013)) or 1013),
            "rainfall": float(daily.get("precipitation_sum", [0] * len(daily_times))[index] or 0),
            "sunrise": daily.get("sunrise", [""] * len(daily_times))[index],
            "sunset": daily.get("sunset", [""] * len(daily_times))[index],
        })

    current_code = int(current.get("weather_code", 0) or 0)
    return {
        "latitude": float(payload.get("latitude", latitude)),
        "longitude": float(payload.get("longitude", longitude)),
        "timezone": payload.get("timezone", "auto"),
        "timezone_abbreviation": payload.get("timezone_abbreviation", ""),
        "observed_at": now_iso(),
        "current": {
            "time": current.get("time", now_iso()),
            "temperature": current.get("temperature_2m"),
            "humidity": current.get("relative_humidity_2m"),
            "apparent_temperature": current.get("apparent_temperature"),
            "precipitation": current.get("precipitation"),
            "rain": current.get("rain"),
            "weather_code": current_code,
            "condition": WEATHER_CODE_LABELS.get(current_code, "Variable weather"),
            "cloud_cover": current.get("cloud_cover"),
            "pressure": current.get("surface_pressure"),
            "wind_speed": current.get("wind_speed_10m"),
        },
        "daily": daily_rows,
        "forecast_daily": daily_rows[-7:],
    }


def store_daily_weather(state, user, project, bundle):
    weather_date = str(bundle.get("current", {}).get("time", ""))[:10] or datetime.now().date().isoformat()
    project_id = project.get("id", "")
    existing = next(
        (
            item for item in state.get("weather_snapshots", [])
            if item.get("user_id") == user["id"]
            and item.get("project_id") == project_id
            and item.get("weather_date") == weather_date
        ),
        None,
    )

    if existing:
        existing["observed_at"] = now_iso()
        existing["bundle"] = bundle
        return existing

    today_daily = next((row for row in bundle.get("daily", []) if row.get("date") == weather_date), {})
    record = {
        "id": f"WTH-{uuid.uuid4().hex[:10].upper()}",
        "user_id": user["id"],
        "project_id": project_id,
        "observed_at": now_iso(),
        "weather_date": weather_date,
        "latitude": bundle["latitude"],
        "longitude": bundle["longitude"],
        "timezone": bundle.get("timezone", ""),
        "current_weather": bundle.get("current", {}),
        "daily_weather": today_daily,
        "bundle": bundle,
    }
    state.setdefault("weather_snapshots", []).append(record)
    supabase_insert("weather_snapshots", {key: value for key, value in record.items() if key != "bundle"})
    return record


def get_location_weather(state, user, project):
    project_id = project.get("id", "")
    cached = recent_weather_bundle(state, user["id"], project_id)
    if cached:
        return cached, None

    latitude, longitude, location_label = project_coordinates(user, project)
    try:
        bundle = fetch_location_weather(latitude, longitude)
        bundle["location_label"] = location_label
        store_daily_weather(state, user, project, bundle)
        return bundle, None
    except Exception as exc:
        logger.warning("Realtime weather request failed: %s", exc)
        previous = latest_weather_bundle(state, user["id"], project_id)
        return previous, "Realtime weather service is temporarily unavailable."


def build_automatic_weather_input():
    if weather_scaler is not None and hasattr(weather_scaler, "data_min_") and hasattr(weather_scaler, "data_max_"):
        seed_values = (weather_scaler.data_min_ + weather_scaler.data_max_) / 2
    else:
        seed_values = np.zeros(len(WEATHER_FEATURES), dtype=np.float32)
    return np.tile(np.array(seed_values, dtype=np.float32), (WEATHER_INPUT_STEPS, 1))


def weather_history_input(bundle):
    daily_rows = bundle.get("daily", []) if bundle else []
    historical_rows = daily_rows[:-6] if len(daily_rows) > 6 else daily_rows
    values = [
        [
            row.get("max_temp", 0),
            row.get("min_temp", 0),
            row.get("avg_temp", 0),
            row.get("humidity", 0),
            row.get("pressure", 1013),
            row.get("rainfall", 0),
        ]
        for row in historical_rows[-WEATHER_INPUT_STEPS:]
    ]
    if not values:
        return build_automatic_weather_input()
    while len(values) < WEATHER_INPUT_STEPS:
        values.insert(0, values[0])
    return np.array(values[-WEATHER_INPUT_STEPS:], dtype=np.float32)


def predict_weather(bundle=None):
    if weather_model is None or weather_scaler is None:
        raise RuntimeError("Weather prediction model or scaler is not available")

    weather_input = weather_history_input(bundle)
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


def store_next_day_prediction(state, user, project, forecast, bundle):
    if not forecast:
        return None

    project_id = project.get("id", "")
    local_date = str(bundle.get("current", {}).get("time", datetime.now().date().isoformat()))[:10]
    target_date = (datetime.fromisoformat(local_date) + timedelta(days=1)).date().isoformat()
    existing = next(
        (
            item for item in state.get("weather_predictions", [])
            if item.get("user_id") == user["id"]
            and item.get("project_id") == project_id
            and str(item.get("target_at", ""))[:10] == target_date
        ),
        None,
    )
    if existing:
        return existing

    prediction_values = {
        WEATHER_FEATURES[index][0]: measurement["value"]
        for index, measurement in enumerate(forecast[0]["measurements"])
    }
    record = {
        "id": f"WPR-{uuid.uuid4().hex[:10].upper()}",
        "user_id": user["id"],
        "project_id": project_id,
        "predicted_at": now_iso(),
        "target_at": f"{target_date}T00:00:00",
        "latitude": bundle["latitude"],
        "longitude": bundle["longitude"],
        "model_name": os.path.basename(WEATHER_MODEL_PATH),
        "prediction": prediction_values,
    }
    state.setdefault("weather_predictions", []).append(record)
    supabase_insert("weather_predictions", record)
    return record


def weather_average(forecast, label):
    values = []
    for day in forecast[:7]:
        for measurement in day["measurements"]:
            if measurement["label"] == label:
                values.append(float(measurement["value"]))
    return round(sum(values) / len(values), 2) if values else 0


def seasonal_weather_summary(state, user_id, project_id, bundle, next_day_prediction):
    snapshots = [
        item for item in state.get("weather_snapshots", [])
        if item.get("user_id") == user_id and item.get("project_id") == project_id
    ][-75:]
    daily_rows = [item.get("daily_weather", {}) for item in snapshots if item.get("daily_weather")]
    if not daily_rows and bundle:
        daily_rows = bundle.get("daily", [])[-7:]

    def average(key, fallback=0):
        values = [float(row.get(key, fallback) or fallback) for row in daily_rows]
        return round(sum(values) / len(values), 2) if values else fallback

    return {
        "sample_days": len(daily_rows),
        "average_max_temperature_c": average("max_temp"),
        "average_min_temperature_c": average("min_temp"),
        "average_temperature_c": average("avg_temp"),
        "average_humidity_percent": average("humidity"),
        "average_pressure_hpa": average("pressure", 1013),
        "average_rainfall_mm": average("rainfall"),
        "next_day_prediction": next_day_prediction.get("prediction", {}) if next_day_prediction else {},
    }


def local_seasonal_recommendation(summary, project):
    avg_temp = float(summary.get("average_temperature_c", 0))
    humidity = float(summary.get("average_humidity_percent", 0))
    rainfall = float(summary.get("average_rainfall_mm", 0))
    floors = max(1, int(project.get("floors", 1) or 1))

    if 18 <= avg_temp <= 28 and humidity <= 80:
        plants = ["Lettuce", "Spinach", "Basil", "Pak choi", "Strawberry", "Mint"]
    elif avg_temp > 28:
        plants = ["Tomato", "Chili", "Okra", "Eggplant", "Cucumber", "Water spinach"]
    else:
        plants = ["Kale", "Coriander", "Pea shoots", "Lettuce", "Strawberry", "Broccoli"]

    if rainfall > 8:
        plants = ["Lettuce", "Basil", "Mint", "Pak choi", "Coriander", "Water spinach"]

    floor_recommendations = []
    floor_groups = [
        ("Lower floor", plants[:2], "Use for heavier crops and stable root-zone temperature."),
        ("Middle floor", plants[2:4], "Use for balanced airflow and moderate light."),
        ("Upper floor", plants[4:6], "Use for crops that benefit from stronger light and ventilation."),
    ]
    for index in range(min(floors, 3)):
        label, floor_plants, reason = floor_groups[index]
        floor_recommendations.append({
            "floor": index + 1,
            "label": label,
            "plants": floor_plants,
            "reason": reason,
        })
    if floors == 4:
        floor_recommendations.append({
            "floor": 4,
            "label": "Top floor",
            "plants": [plants[-1], plants[0]],
            "reason": "Reserve for the highest-light crops with active cooling and humidity control.",
        })

    report = (
        f"Based on {summary.get('sample_days', 0)} stored weather days, average temperature is "
        f"{avg_temp:g} C, humidity {humidity:g}%, and rainfall {rainfall:g} mm. "
        f"Prioritize {', '.join(plants[:4])}. Match crop height and light demand to the recommended floors."
    )
    return plants, floor_recommendations, report, "NuroAgro 75-day local analysis"


def seasonal_analysis_with_agent(summary, project):
    plants, floor_recommendations, report, source = local_seasonal_recommendation(summary, project)
    if not OPENAI_API_KEY:
        return plants, floor_recommendations, report, source

    context = {
        "project": {
            "name": project.get("name"),
            "area_sq_ft": project.get("area"),
            "floors": project.get("floors"),
            "goal": project.get("goal"),
            "latitude": project.get("lat"),
            "longitude": project.get("lng"),
        },
        "weather_summary": summary,
        "local_recommendation": {
            "preferred_plants": plants,
            "floor_recommendations": floor_recommendations,
        },
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
                "reasoning": {"effort": "low"},
                "instructions": (
                    "You are NuroAgro's 75-day crop planning advisor. Analyze location, stored weather averages, "
                    "next-day model prediction, farm goal, and vertical floor count. Return a concise practical "
                    "report that recommends plants and assigns suitable crops to each available floor. Mention "
                    "weather risks and environmental controls. Do not invent unavailable measurements."
                ),
                "input": json.dumps(context),
                "max_output_tokens": 800,
            },
            timeout=45,
        )
        response.raise_for_status()
        agent_report = extract_response_text(response.json())
        if agent_report:
            report = agent_report
            source = f"OpenAI {OPENAI_PROJECT_MODEL}"
    except Exception as exc:
        logger.warning("OpenAI seasonal weather analysis failed; using local fallback: %s", exc)

    return plants, floor_recommendations, report, source


def get_or_create_seasonal_analysis(state, user, project, bundle, next_day_prediction):
    project_id = project.get("id", "")
    analyses = [
        item for item in state.get("seasonal_analyses", [])
        if item.get("user_id") == user["id"] and item.get("project_id") == project_id
    ]
    if analyses:
        latest = analyses[-1]
        next_analysis_at = parse_iso_datetime(latest.get("next_analysis_at"))
        if next_analysis_at and datetime.now(timezone.utc) < next_analysis_at.astimezone(timezone.utc):
            return latest

    summary = seasonal_weather_summary(state, user["id"], project_id, bundle, next_day_prediction)
    plants, floor_recommendations, report, source = seasonal_analysis_with_agent(summary, project)
    generated_at = datetime.now(timezone.utc)
    record = {
        "id": f"SEA-{uuid.uuid4().hex[:10].upper()}",
        "user_id": user["id"],
        "project_id": project_id,
        "generated_at": generated_at.isoformat(),
        "next_analysis_at": (generated_at + timedelta(days=SEASONAL_ANALYSIS_DAYS)).isoformat(),
        "latitude": float(bundle.get("latitude", project.get("lat", 0)) or 0),
        "longitude": float(bundle.get("longitude", project.get("lng", 0)) or 0),
        "preferred_plants": plants,
        "floor_recommendations": floor_recommendations,
        "source": source,
        "report": report,
        "weather_summary": summary,
    }
    state.setdefault("seasonal_analyses", []).append(record)
    supabase_insert("seasonal_analyses", record)
    return record


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
        "supabase_enabled": supabase_configured(),
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


PLANT_COLOR_PALETTE = [
    sv.Color(46, 134, 222),
    sv.Color(16, 172, 132),
    sv.Color(245, 166, 35),
    sv.Color(155, 89, 182),
    sv.Color(231, 76, 60),
    sv.Color(0, 184, 212),
    sv.Color(121, 85, 72),
    sv.Color(63, 81, 181),
]


def detection_color(model_key, class_name, class_id):
    if model_key == "rice":
        return COLOR_MAP.get(class_name, sv.Color(128, 128, 128))
    return PLANT_COLOR_PALETTE[int(class_id) % len(PLANT_COLOR_PALETTE)]


def disease_recommendation(class_name):
    if class_name in DISEASE_ACTIONS:
        return DISEASE_ACTIONS[class_name]
    if class_name in PLANT_HEALTHY_CLASSES:
        return "Healthy leaf class detected. Continue routine monitoring, balanced nutrition, and sanitation."

    normalized_name = class_name.lower()
    for disease_term, guidance in PLANT_DISEASE_GUIDANCE.items():
        if disease_term in normalized_name:
            return guidance
    return "Inspect the affected plant, isolate suspicious foliage, sanitize tools, and confirm with a local crop specialist."


def process_image(image_path):
    model_specs = [
        ("rice", "Rice", rice_disease_model),
        ("plant", "Plant", plant_disease_model),
    ]
    available_models = [spec for spec in model_specs if spec[2] is not None]
    if not available_models:
        return None, "Disease models are not available", {}, [], []

    try:
        image = cv2.imread(image_path)
        if image is None:
            return None, "Error: Could not load image", {}, [], []

        image = cv2.resize(image, (1280, 720))
        annotated_image = image.copy()
        disease_counts = {}
        detection_records = []
        successful_models = 0

        for model_key, model_label, model in available_models:
            try:
                results = model(image, verbose=False)[0]
                detections = sv.Detections.from_ultralytics(results)
                successful_models += 1
            except Exception as exc:
                logger.exception("%s disease model inference failed: %s", model_label, exc)
                continue

            for detection_idx, xyxy in enumerate(detections.xyxy):
                class_id = detections.class_id[detection_idx]
                class_name = results.names[class_id]
                confidence = float(detections.confidence[detection_idx])
                disease_counts[class_name] = disease_counts.get(class_name, 0) + 1
                detection_records.append({
                    "model": model_key,
                    "model_label": model_label,
                    "class_id": int(class_id),
                    "class_name": class_name,
                    "confidence": round(confidence, 5),
                    "bbox": [round(float(value), 2) for value in xyxy],
                })
                color = detection_color(model_key, class_name, class_id)
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
                labels = [f"{model_label} | {class_name}: {single_detection.confidence[0]:.2f}"]
                annotated_image = label_annotator.annotate(
                    scene=annotated_image,
                    detections=single_detection,
                    labels=labels,
                )

        if successful_models == 0:
            return None, "Both disease models failed during image analysis", {}, [], []

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
        recommendations = []
        for disease in disease_counts:
            recommendation = f"{disease}: {disease_recommendation(disease)}"
            if recommendation not in recommendations:
                recommendations.append(recommendation)
        if not recommendations:
            recommendations = ["No disease detected by either model. Keep monitoring leaf color, humidity, and airflow."]
        return output_path, None, disease_counts, recommendations, detection_records

    except Exception as exc:
        logger.exception("Error processing image: %s", exc)
        return None, f"Error: {str(exc)}", {}, [], []


def build_context(state, user, disease_result=None, error=None, setup_result=None):
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

    weather_bundle, realtime_weather_error = get_location_weather(state, user, current_project)
    weather_error = realtime_weather_error
    try:
        weather_forecast = predict_weather(weather_bundle)
    except Exception as exc:
        weather_forecast = []
        weather_error = str(exc)
        logger.error("Weather prediction failed: %s", exc)

    next_day_prediction = None
    seasonal_analysis = None
    if weather_bundle and weather_forecast:
        next_day_prediction = store_next_day_prediction(state, user, current_project, weather_forecast, weather_bundle)
        if projects:
            seasonal_analysis = get_or_create_seasonal_analysis(
                state,
                user,
                current_project,
                weather_bundle,
                next_day_prediction,
            )
        save_state(state)

    reading = latest_sensor(state)
    automation, alerts = evaluate_automation(reading)
    sensor_project_id = current_project.get("id") or reading.get("project_id", "")
    sensor_analysis = get_or_create_sensor_analysis(state, sensor_project_id)
    if sensor_analysis:
        save_state(state)
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
        or record.get("user_id") == user.get("id")
    ]
    sensor_history = [
        record for record in state.get("sensor_history", [])
        if record.get("project_id") in project_ids
    ]
    weather_history = [
        record for record in state.get("weather_snapshots", [])
        if record.get("user_id") == user.get("id")
        or record.get("project_id") in project_ids
    ]
    prediction_history = [
        record for record in state.get("weather_predictions", [])
        if record.get("user_id") == user.get("id")
        or record.get("project_id") in project_ids
    ]
    seasonal_history = [
        record for record in state.get("seasonal_analyses", [])
        if record.get("user_id") == user.get("id")
        or record.get("project_id") in project_ids
    ]
    sensor_analysis_history = [
        record for record in state.get("sensor_analyses", [])
        if record.get("project_id") in project_ids
    ]
    activity_history = [
        record for record in state.get("activities", [])
        if record.get("actor_type") == "user" and record.get("actor_id") == user.get("id")
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
        "realtime_weather": weather_bundle,
        "next_day_prediction": next_day_prediction,
        "seasonal_analysis": seasonal_analysis,
        "weather_error": weather_error,
        "sensor_cards": sensor_cards(reading),
        "latest_sensor": reading,
        "automation": automation,
        "alerts": alerts,
        "sensor_analysis": sensor_analysis,
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
        "history": {
            "projects": list(reversed(projects)),
            "diseases": list(reversed(disease_history)),
            "sensors": list(reversed(sensor_history)),
            "weather": list(reversed(weather_history)),
            "predictions": list(reversed(prediction_history)),
            "seasonal": list(reversed(seasonal_history)),
            "sensor_analyses": list(reversed(sensor_analysis_history)),
            "activities": list(reversed(activity_history)),
        },
        "history_counts": {
            "all": (
                len(projects)
                + len(disease_history)
                + len(sensor_history)
                + len(weather_history)
                + len(prediction_history)
                + len(seasonal_history)
                + len(sensor_analysis_history)
                + len(activity_history)
            ),
            "diseases": len(disease_history),
            "sensors": len(sensor_history),
            "weather": len(weather_history) + len(prediction_history),
            "analyses": len(seasonal_history) + len(sensor_analysis_history),
            "activities": len(activity_history) + len(projects),
        },
        "supabase_enabled": supabase_configured(),
        "sample_camera_image": url_for("static", filename="uploads/images_7.jpg"),
    }


@app.route("/favicon.ico")
def favicon():
    favicon_svg = """
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
        <rect width="64" height="64" rx="12" fill="#176b87"/>
        <path d="M18 46V18h7l14 17V18h7v28h-7L25 29v17z" fill="#fff"/>
        <path d="M42 13c7 1 10 5 9 12-7 0-11-4-9-12z" fill="#66d1aa"/>
    </svg>
    """
    return Response(favicon_svg, mimetype="image/svg+xml")


@app.route("/signup", methods=["GET", "POST"])
@state_transaction
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
@state_transaction
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
@state_transaction
def logout():
    state = load_state()
    user = current_user(state)
    if user:
        record_activity(state, "user", user["id"], "logout", "User logged out.")
        save_state(state)
    session.pop("user_id", None)
    return redirect(url_for("login"))


@app.route("/", methods=["GET", "POST"])
@state_transaction
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
                try:
                    weather_bundle = fetch_location_weather(project["lat"], project["lng"])
                except Exception as weather_exc:
                    logger.warning("Project weather lookup failed; using model fallback: %s", weather_exc)
                    weather_bundle = None
                forecast = predict_weather(weather_bundle)
                setup_result = analyze_project_with_agent(project, forecast)
                project["analysis"] = setup_result
                state.setdefault("projects", []).append(project)
                user["location"] = {
                    "label": f"{project['lat']:.5f}, {project['lng']:.5f}",
                    "lat": project["lat"],
                    "lng": project["lng"],
                }
                if weather_bundle:
                    store_daily_weather(state, user, project, weather_bundle)
                    next_day_prediction = store_next_day_prediction(state, user, project, forecast, weather_bundle)
                    get_or_create_seasonal_analysis(state, user, project, weather_bundle, next_day_prediction)
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
                (
                    output_path,
                    processing_error,
                    disease_counts,
                    disease_recommendations,
                    detection_records,
                ) = process_image(filepath)
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
                        "detections": detection_records,
                        "recommendations": disease_recommendations,
                        "user_id": user["id"],
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
@state_transaction
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
@state_transaction
def api_sensors():
    state = load_state()
    payload = request.get_json(silent=True) or request.form.to_dict()
    reading = normalize_sensor_payload(payload)
    actions, alerts = evaluate_automation(reading)
    state.setdefault("sensor_history", []).append(reading)
    sensor_analysis = get_or_create_sensor_analysis(state, reading.get("project_id", ""))
    save_state(state)
    supabase_insert("sensor_readings", reading)
    return jsonify({
        "ok": True,
        "stored": reading,
        "automation": actions,
        "alerts": alerts,
        "two_hour_analysis": sensor_analysis,
    })


@app.route("/api/status", methods=["GET"])
@state_transaction
def api_status():
    state = load_state()
    reading = latest_sensor(state)
    actions, alerts = evaluate_automation(reading)
    sensor_analysis = get_or_create_sensor_analysis(state, reading.get("project_id", ""))
    if sensor_analysis:
        save_state(state)
    return jsonify({
        "app": APP_NAME,
        "latest_sensor": reading,
        "automation": actions,
        "alerts": alerts,
        "two_hour_analysis": sensor_analysis,
        "supabase_enabled": supabase_configured(),
    })


if __name__ == "__main__":
    threading.Thread(target=sensor_analysis_scheduler, daemon=True, name="sensor-analysis-scheduler").start()
    app.run(debug=True, use_reloader=False)
