from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

MODEL_FILES = {"Fuel_flow_mf_kg/s": "xgb_Fuel_flow_mf_kg_s.pkl", "GT_Turbine_decay_state_coefficient": "xgb_GT_Turbine_decay_state_coefficient.pkl", "GT_Compressor_decay_state_coefficient": "xgb_GT_Compressor_decay_state_coefficient.pkl"}
MODELS = {}
for k, fname in MODEL_FILES.items():
    MODELS[k] = joblib.load(os.path.join(os.path.dirname(__file__), fname))

FEATURE_COLUMNS = ['index', 'Lever_position', 'Ship_speed_v', 'Gas_Turbine_GT_shaft_torque_GTT_kN_m', 'GT_rate_of_revolutions_GTn_rpm', 'Gas_Generator_rate_of_revolutions_GGn_rpm', 'Starboard_Propeller_Torque_Ts_kN', 'Port_Propeller_Torque_Tp_kN', 'Hight_Pressure_HP_Turbine_exit_temperature_T48_C', 'GT_Compressor_inlet_air_temperature_T1_C', 'GT_Compressor_outlet_air_temperature_T2_C', 'HP_Turbine_exit_pressure_P48_bar', 'GT_Compressor_inlet_air_pressure_P1_bar', 'GT_Compressor_outlet_air_pressure_P2_bar', 'GT_exhaust_gas_pressure_Pexh_bar', 'Turbine_Injecton_Control_TIC_']

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    if isinstance(data, dict):
        records = [data]
    else:
        records = data
    df = pd.DataFrame(records)
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        return jsonify({'error': 'Missing columns', 'missing': missing}), 400
    X = df[FEATURE_COLUMNS]
    outputs = {}
    for target, model in MODELS.items():
        preds = model.predict(X).tolist()
        outputs[target] = preds if len(preds) > 1 else preds[0]
    return jsonify(outputs)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
