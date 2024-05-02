from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import traceback

app = Flask(__name__)

# Mock data for devices
devices = [
    {"ID": 1, "battery_power": 1043, "blue": 1, "clock_speed": 1.8, "dual_sim": 1,
     "fc": 14, "four_g": 0, "int_memory": 5, "m_dep": 0.1, "mobile_wt": 193,
     "n_cores": 3, "pc": 16, "px_height": 226, "px_width": 1412, "ram": 3476,
     "sc_h": 12, "sc_w": 7, "talk_time": 2, "three_g": 0, "touch_screen": 1, "wifi": 0},
    {"ID": 2, "battery_power": 841, "blue": 1, "clock_speed": 0.5, "dual_sim": 1,
     "fc": 4, "four_g": 1, "int_memory": 61, "m_dep": 0.8, "mobile_wt": 191,
     "n_cores": 5, "pc": 12, "px_height": 746, "px_width": 857, "ram": 3895,
     "sc_h": 6, "sc_w": 0, "talk_time": 7, "three_g": 1, "touch_screen": 0, "wifi": 0},
    {"ID": 3, "battery_power": 1807, "blue": 1, "clock_speed": 2.8, "dual_sim": 0,
     "fc": 1, "four_g": 1, "int_memory": 27, "m_dep": 0.9, "mobile_wt": 186,
     "n_cores": 3, "pc": 4, "px_height": 1270, "px_width": 1366, "ram": 2396,
     "sc_h": 17, "sc_w": 10, "talk_time": 10, "three_g": 0, "touch_screen": 1, "wifi": 1}
]

# Endpoints
@app.route('/api/devices', methods=['GET'])
def get_all_devices():
    return jsonify(devices)

@app.route('/api/devices/<int:id>', methods=['GET'])
def get_device_by_id(id):
    device = next((device for device in devices if device['id'] == id), None)
    if device:
        return jsonify(device)
    else:
        return jsonify({"error": "Device not found"}), 404

@app.route('/api/devices', methods=['POST'])
def add_device():
    new_device = request.json
    devices.append(new_device)
    return jsonify({"message": "Device added successfully"}), 201

@app.route('/api/predict/', methods=['POST'])
def predict_device():
    model = joblib.load("model.pkl")

    if model:
        try:
            json_ = request.json
          #   print(json_)
            query = pd.DataFrame(json_)
            # query = query.reindex(columns=model_columns, fill_value=0)

            prediction = list(model.predict(query))

            return jsonify({'prediction': str(prediction)})

        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')

if __name__ == '__main__':
    app.run(debug=True)
