import requests, json
url = 'http://localhost:5000/predict'
example = {
    "index": 10172.0,
    "Lever_position": 3.144,
    "Ship_speed_v": 9.0,
    "Gas_Turbine_GT_shaft_torque_GTT_kN_m": 8375.164,
    "GT_rate_of_revolutions_GTn_rpm": 1386.735,
    "Gas_Generator_rate_of_revolutions_GGn_rpm": 7045.84,
    "Starboard_Propeller_Torque_Ts_kN": 60.311,
    "Port_Propeller_Torque_Tp_kN": 60.311,
    "Hight_Pressure_HP_Turbine_exit_temperature_T48_C": 576.879,
    "GT_Compressor_inlet_air_temperature_T1_C": 288.0,
    "GT_Compressor_outlet_air_temperature_T2_C": 576.548,
    "HP_Turbine_exit_pressure_P48_bar": 1.392,
    "GT_Compressor_inlet_air_pressure_P1_bar": 0.998,
    "GT_Compressor_outlet_air_pressure_P2_bar": 7.506,
    "GT_exhaust_gas_pressure_Pexh_bar": 1.021,
    "Turbine_Injecton_Control_TIC_": 11.961
}
resp = requests.post(url, json=example)
print('Status', resp.status_code)
print(resp.json())
