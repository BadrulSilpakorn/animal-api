from flask import Flask, request, jsonify
import numpy as np
import tflite_runtime.interpreter as tflite

# โหลดโมเดล TFLite
interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

app = Flask(__name__)

@app.route("/")
def home():
    return "✅ TFLite API Server is running"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # ESP32 ส่ง JSON เช่น {"data": [0.1, 0.2, 0.3, ...]}
        data = request.json.get("data", [])
        arr = np.array(data, dtype=np.float32).reshape(input_details[0]['shape'])

        # run inference
        interpreter.set_tensor(input_details[0]['index'], arr)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])

        return jsonify({"prediction": output.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
