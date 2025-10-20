from flask import Flask, request, jsonify
import numpy as np, io, base64, requests, os
from PIL import Image
from datetime import datetime, timedelta

app = Flask(__name__)

# ===== โหลด labels =====
def load_labels(path="labels.txt"):
    if os.path.exists(path):
        with open(path) as f:
            return [line.strip() for line in f if line.strip()]
    return ["nottarget", "cow", "goat", "sheep"]

labels = load_labels()
print("Labels:", labels)

# ===== โหลดโมดูล TensorFlow Lite =====
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow as tf
    tflite = tf.lite

# ===== โหลดโมเดล =====
class Model:
    def __init__(self):
        self.interpreter = None
        self.input = None
        self.output = None
        self.loaded = False

    def load(self, path="animal_model_float32.tflite"):
        if not os.path.exists(path):
            print("Model not found.")
            return False
        self.interpreter = tflite.Interpreter(model_path=path)
        self.interpreter.allocate_tensors()
        self.input = self.interpreter.get_input_details()[0]
        self.output = self.interpreter.get_output_details()[0]
        self.loaded = True
        print("Model loaded:", path)
        return True

    def predict(self, arr):
        self.interpreter.set_tensor(self.input["index"], arr)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output["index"])[0]

model = Model()
model.load()

# ===== Telegram Config =====
TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
TG_URL = f"https://api.telegram.org/bot{TOKEN}/sendPhoto"

def now():
    return (datetime.utcnow() + timedelta(hours=7)).strftime("%H:%M:%S")

def send_photo(img, caption, silent=False):
    if not TOKEN or not CHAT_ID: 
        return
    try:
        r = requests.post(TG_URL, files={'photo': ('img.jpg', img, 'image/jpeg')},
                          data={'chat_id': CHAT_ID, 'caption': caption,
                                'disable_notification': silent}, timeout=10)
        print("Telegram:", r.status_code)
    except Exception as e:
        print("Telegram error:", e)

# ===== Routes =====
@app.route("/")
def home():
    return jsonify({
        "status": "running",
        "model_loaded": model.loaded,
        "labels": labels,
        "time": now()
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if not model.loaded:
            if not model.load(): 
                return jsonify({"error": "model not loaded"}), 500

        img_b64 = request.json.get("image")
        if not img_b64:
            return jsonify({"error": "no image"}), 400

        img = Image.open(io.BytesIO(base64.b64decode(img_b64))).convert("RGB")
        h, w = model.input["shape"][1:3]
        arr = np.expand_dims(np.array(img.resize((w, h)), np.float32), 0)

        pred = model.predict(arr)
        probs = np.exp(pred - np.max(pred))
        probs /= probs.sum()
        idx = np.argmax(probs)
        label, conf = labels[idx], float(probs[idx] * 100)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        silent = label == "nottarget"
        cap = f"{'✅ Clear area' if silent else '🚨 ALERT: ' + label.upper()} ({conf:.1f}%)"
        send_photo(buf.getvalue(), cap, silent)

        return jsonify({"prediction": label, "confidence": round(conf, 1), "time": now()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=False)

