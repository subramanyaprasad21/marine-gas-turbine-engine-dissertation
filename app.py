# app.py - robust loader that downloads missing .pkl from GitHub raw if necessary
from flask import Flask, request, jsonify
import joblib, os, sys, requests, time
import pandas as pd

app = Flask(__name__)

# ---------- CONFIG: update these if your repo/branch/username differ ----------
GITHUB_USER = "subramanyaprasad21"
GITHUB_REPO = "marine-gas-turbine-engine-dissertation"
GITHUB_BRANCH = "main"   # usually main or master
# ---------------------------------------------------------------------------

# Map logical target -> expected filename in repo root
MODEL_FILES = {
    "Fuel_flow_mf_kg/s": "rf_Fuel_flow_mf_kg_s.pkl",
    "GT_Turbine_decay_state_coefficient": "rf_GT_Turbine_decay_state_coefficient_compressed.pkl",
    "GT_Compressor_decay_state_coefficient": "rf_GT_Compressor_decay_state_coefficient.pkl"
}

MODELS = {}
missing_files = []
download_errors = []

# utility to download a raw file from GitHub
def download_from_github_raw(filename, dest_path):
    raw_url = f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/{GITHUB_BRANCH}/{filename}"
    try:
        print(f"Attempting to download {filename} from {raw_url}", file=sys.stderr)
        resp = requests.get(raw_url, timeout=30, stream=True)
        if resp.status_code == 200:
            with open(dest_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print(f"Downloaded {filename} -> {dest_path}", file=sys.stderr)
            return True
        else:
            print(f"Failed to download {filename}: HTTP {resp.status_code}", file=sys.stderr)
            return False
    except Exception as e:
        print(f"Exception downloading {filename}: {e}", file=sys.stderr)
        return False

# show what files are present (very helpful in Railway logs)
print("=== STARTUP: listing current directory files ===", file=sys.stderr)
for f in sorted(os.listdir(".")):
    print(" -", f, file=sys.stderr)
print("=== END STARTUP LIST ===", file=sys.stderr)

# ensure requests is available; Railway installs requirements, so OK
# Try to ensure each model exists locally, otherwise attempt to download from GitHub raw
for key, fname in MODEL_FILES.items():
    if os.path.exists(fname):
        print(f"Found file {fname} locally", file=sys.stderr)
    else:
        print(f"File {fname} not found locally. Attempting to fetch from GitHub.", file=sys.stderr)
        ok = download_from_github_raw(fname, fname)
        if not ok:
            download_errors.append(fname)

# Now try loading models
for key, fname in MODEL_FILES.items():
    try:
        if not os.path.exists(fname):
            missing_files.append(fname)
            continue
        MODELS[key] = joblib.load(fname)
        print(f"Loaded model for {key} from {fname}", file=sys.stderr)
    except Exception as e:
        print(f"Error loading model {fname}: {e}", file=sys.stderr)
        download_errors.append(fname)

# Feature columns - must match training
FEATURE_COLUMNS = [
    "Lever_position",
    "Ship_speed_v",
    "Gas_Turbine_GT_shaft_torque_GTT_kN_m",
    "GT_rate_of_revolutions_GTn_rpm",
    "Gas_Generator_rate_of_revolutions_GGn_rpm",
    "Starboard_Propeller_Torque_Ts_kN",
    "Port_Propeller_Torque_Tp_kN",
    "Hight_Pressure_HP_Turbine_exit_temperature_T48_C",
    "GT_Compressor_inlet_air_temperature_T1_C",
    "GT_Compressor_outlet_air_temperature_T2_C",
    "HP_Turbine_exit_pressure_P48_bar",
    "GT_Compressor_inlet_air_pressure_P1_bar",
    "GT_Compressor_outlet_air_pressure_P2_bar",
    "GT_exhaust_gas_pressure_Pexh_bar",
    "Turbine_Injecton_Control_TIC_"
]

@app.route("/")
def index():
    status = {
        "models_loaded": list(MODELS.keys()),
        "missing_files": missing_files,
        "download_errors": download_errors
    }
    code = 200 if not missing_files and not download_errors else 500
    return jsonify(status), code

@app.route("/predict", methods=["POST"])
def predict():
    if not MODELS:
        return jsonify({"error": "No models loaded", "missing": missing_files, "download_errors": download_errors}), 500
    data = request.get_json(force=True)
    if isinstance(data, dict):
        records = [data]
    else:
        records = data
    df = pd.DataFrame(records)
    missing_cols = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing_cols:
        return jsonify({"error": "Missing input columns", "missing_columns": missing_cols}), 400
    X = df[FEATURE_COLUMNS]
    outputs = {}
    for target, model in MODELS.items():
        preds = model.predict(X).tolist()
        outputs[target] = preds if len(preds) > 1 else preds[0]
    return jsonify(outputs)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
