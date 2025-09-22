from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import io
import base64
import requests
import os
from datetime import datetime, timedelta
import tflite_runtime.interpreter as tflite   # ✅ float32-only

app = Flask(__name__)

# ===== Telegram Config =====
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

def send_telegram_message(text, image_bytes=None):
    """ส่งข้อความ + รูปไป Telegram"""
    if not TELEGRAM_TOKEN or not CHAT_ID:
        print("⚠️ Telegram not configured")
        return False
    try:
        # ส่งข้อความ
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        data = {"chat_id": CHAT_ID, "text": text, "parse_mode": "HTML"}
        requests.post(url, data=data, timeout=10)

        # ส่งภาพ (ถ้ามี)
        if image_bytes:
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
            files = {"photo": ("image.jpg", image_bytes, "image/jpeg")}
            data = {"chat_id": CHAT_ID}
            requests.post(url, data=data, files=files, timeout=15)

        return True
    except Exception as e:
        print("❌ Telegram error:", e)
        return False

def get_thai_time():
    return (datetime.utcnow() + timedelta(hours=7)).strftime("%H:%M:%S")

# ===== Model Loader =====
class ModelLoader:
    def __init__(self):
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.model_file = None
        self.loaded = False

    def load_model(self, model_path="animal_model_float32_v1.tflite"):
        try:
            print(f"🔄 Loading: {model_path}")
            self.interpreter = tflite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.model_file = model_path
            self.loaded = True
            print(f"✅ Loaded model: {model_path}")
            return True
        except Exception as e:
            print("❌ Load error:", e)
            return False

    def predict(self, img_array):
        if not self.loaded:
            raise Exception("No model loaded")
        self.interpreter.set_tensor(self.input_details[0]["index"], img_array)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_details[0]["index"])[0]

# สร้าง model loader
model = ModelLoader()
model.load_model()

# ===== Routes =====
@app.route("/")
def home():
    tflite_files = [f for f in os.listdir('.') if f.endswith('.tflite')]
    return jsonify({
        "status": "running",
        "model_loaded": model.loaded,
        "model_file": model.model_file,
        "available_models": tflite_files,
        "time": get_thai_time(),
        "telegram_ready": bool(TELEGRAM_TOKEN and CHAT_ID)
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if not model.loaded:
            return jsonify({"error": "Model not loaded"}), 500

        if not request.json or "image" not in request.json:
            return jsonify({"error": "No image provided"}), 400

        # Decode image
        img_base64 = request.json["image"]
        original_bytes = base64.b64decode(img_base64)
        original_img = Image.open(io.BytesIO(original_bytes)).convert("RGB")

        # Resize for model
        target_size = model.input_details[0]['shape'][1:3]
        model_img = original_img.resize((target_size[1], target_size[0]))
        img_array = np.array(model_img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        output = model.predict(img_array)
        exp_out = np.exp(output - np.max(output))
        probs = exp_out / np.sum(exp_out)

        labels = ["nottarget", "cow", "goat", "sheep"]
        pred_idx = int(np.argmax(probs))
        pred_label = labels[pred_idx]
        confidence = float(probs[pred_idx] * 100)

        print(f"🎯 Prediction: {pred_label} ({confidence:.1f}%)")

        # เตรียมภาพสำหรับส่ง Telegram
        img_buffer = io.BytesIO()
        original_img.save(img_buffer, format="JPEG", quality=85, optimize=True)
        clean_img = img_buffer.getvalue()

        # ส่งไป Telegram
        thai_time = get_thai_time()
        msg = f"🤖 <b>Animal Detection</b>\n📌 Prediction: <b>{pred_label}</b>\n📊 Confidence: {confidence:.1f}%\n⏰ Time: {thai_time}"
        send_telegram_message(msg, clean_img)

        return jsonify({
            "status": "success",
            "prediction": pred_label,
            "confidence": round(confidence, 1),
            "all_predictions": {
                labels[i]: round(float(probs[i] * 100), 1)
                for i in range(len(labels))
            },
            "time": thai_time,
            "model_info": {
                "file": model.model_file,
                "type": "float32"
            }
        })

    except Exception as e:
        error_msg = str(e)
        print("❌ Error:", error_msg)
        send_telegram_message(f"❌ ERROR\n{error_msg}\n⏰ {get_thai_time()}")
        return jsonify({"status": "error", "error": error_msg}), 500

# ===== Run Server =====
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
