from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os
import sys

app = Flask(__name__)

# Print out the files at startup for debugging
print("=== STARTUP: Listing files in current directory ===", file=sys.stderr)
for f in sorted(os.listdir(".")):
    print(" -", f, file=sys.stderr)
print("===================================================", file=sys.stderr)

# ✅ Update this dictionary to match your uploaded filenames
MODEL_FILES = {
    "Fuel_flow_mf_kg/s": "rf_Fuel_flow_mf_kg_s.pkl",
    "GT_Turbine_decay_state_coefficient": "rf_GT_Turbine_decay_state_coefficient_compressed.pkl",  # <- compressed file
    "GT_Compressor_decay_state_coefficient": "rf_GT_Compressor_decay_state_coefficient.pkl"
}

MODELS = {}
missing_files = []

# Load models safely
for key, filename in MODEL_FILES.items():
    if os.path.exists(filename):
        try:
            MODELS[key] = joblib.load(filename)
            print(f"✅ Loaded model for {key} from {filename}", file=sys.stderr)
        except Exception as e:
            print(f"❌ Failed to load {filename}: {e}", file=sys.stderr)
            missing_files.append(filename)
    else:
        print(f"⚠️ Missing model file: {filename}", file=sys.stderr)
        missing_files.append(filename)

# Feature columns must exactly match those used during training
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

@app.route('/')
def home():
    """Root route to show server status."""
    if missing_files:
        return jsonify({
            "status": "error",
            "message": "Some model files are missing.",
            "missing_files": missing_files
        }), 500
    return jsonify({
        "status": "ok",
        "message": "API is running and all models are loaded successfully."
    }), 200


@app.route('/predict', methods=['POST'])
def predict():
    """POST endpoint to get predictions for turbine data."""
    try:
        if not MODELS:
            return jsonify({
                "error": "Models not loaded",
                "details": "Check the logs for missing files"
            }), 500

        data = request.get_json(force=True)

        if isinstance(data, dict):
            records = [data]
        else:
            records = data

        df = pd.DataFrame(records)
        missing_cols = [c for c in FEATURE_COLUMNS if c not in df.columns]
        if missing_cols:
            return jsonify({
                "error": "Missing required input columns",
                "missing_columns": missing_cols
            }), 400

        X = df[FEATURE_COLUMNS]
        output = {}
        for target, model in MODELS.items():
            preds = model.predict(X).tolist()
            output[target] = preds if len(preds) > 1 else preds[0]

        return jsonify(output)

    except Exception as e:
        print(f"❌ Prediction error: {e}", file=sys.stderr)
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
