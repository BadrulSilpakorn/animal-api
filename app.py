from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import io, base64, requests, os
from datetime import datetime, timedelta

app = Flask(__name__)

# ==== Load labels from labels.txt ====
def load_labels(file_path="labels.txt"):
    if not os.path.exists(file_path):
        print("⚠️ labels.txt not found, using fallback labels")
        return ["nottarget", "cow", "goat", "sheep"]  # fallback
    with open(file_path, "r") as f:
        return [line.strip() for line in f.readlines() if line.strip()]

labels = load_labels()
print(f"📄 Loaded labels: {labels}")

# ==== Multi-TensorFlow Loader ====
def load_tflite():
    try:
        import tflite_runtime.interpreter as tflite
        return tflite, "tflite-runtime"
    except ImportError:
        try:
            import tensorflow as tf
            return tf.lite, "tensorflow"
        except ImportError:
            return None, "none"

tflite_module, tf_type = load_tflite()
print(f"🧠 Using: {tf_type}")

# ==== Smart Model Loader ====
class SmartModelLoader:
    def __init__(self):
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.model_type = None
        self.model_file = None
        self.loaded = False
        
    def try_load_model(self, model_path):
        try:
            print(f"🔄 Trying: {model_path}")
            if not os.path.exists(model_path):
                print(f"📄 Not found: {model_path}")
                return False
            self.interpreter = tflite_module.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            input_dtype = self.input_details[0]['dtype']
            self.model_type = "int8" if input_dtype == np.uint8 else "float32"
            self.model_file = model_path
            self.loaded = True
            print(f"✅ Loaded: {model_path} ({self.model_type})")
            print(f"📐 Input: {self.input_details[0]['shape']}")
            return True
        except Exception as e:
            print(f"❌ Failed {model_path}: {e}")
            return False
    
    def load_any_model(self):
        candidates = ["animal_model_float32_v1.tflite", "animal_model_float32.tflite"]
        for path in candidates:
            if self.try_load_model(path):
                return True
        print("❌ No compatible model found!")
        return False
    
    def predict(self, image_array):
        if not self.loaded:
            raise Exception("No model loaded")
        self.interpreter.set_tensor(self.input_details[0]['index'], image_array)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_details[0]['index'])[0]

model = SmartModelLoader()

# ==== Telegram config ====
TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def get_thai_time():
    return (datetime.utcnow() + timedelta(hours=7)).strftime('%H:%M:%S')

def send_message(text):
    if not TOKEN or not CHAT_ID:
        return False
    try:
        url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
        data = {"chat_id": CHAT_ID, "text": text, "parse_mode": "HTML"}
        return requests.post(url, data=data, timeout=10).status_code == 200
    except:
        return False

def send_photo(image_bytes, caption=""):
    if not TOKEN or not CHAT_ID:
        return False
    try:
        url = f"https://api.telegram.org/bot{TOKEN}/sendPhoto"
        files = {'photo': ('img.jpg', image_bytes, 'image/jpeg')}
        data = {'chat_id': CHAT_ID, 'caption': caption, 'parse_mode': 'HTML'}
        return requests.post(url, files=files, data=data, timeout=15).status_code == 200
    except:
        return False

# ==== Routes ====
@app.route("/")
def home():
    tflite_files = [f for f in os.listdir('.') if f.endswith('.tflite')]
    return jsonify({
        "status": "running",
        "tensorflow": tf_type,
        "model_loaded": model.loaded,
        "model_file": model.model_file if model.loaded else None,
        "model_type": model.model_type if model.loaded else None,
        "available_models": tflite_files,
        "labels": labels,
        "time": get_thai_time(),
        "telegram_ready": bool(TOKEN and CHAT_ID)
    })

@app.route("/load-model")
def load_model():
    success = model.load_any_model()
    return jsonify({
        "success": success,
        "model_file": model.model_file if success else None,
        "model_type": model.model_type if success else None
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if not model.loaded:
            if not model.load_any_model():
                return jsonify({"error": "No model"}), 500
        if not request.json or "image" not in request.json:
            return jsonify({"error": "No image"}), 400

        img_base64 = request.json["image"]
        original_bytes = base64.b64decode(img_base64)
        original_img = Image.open(io.BytesIO(original_bytes)).convert("RGB")

        target_size = model.input_details[0]['shape'][1:3]
        model_img = original_img.resize((target_size[1], target_size[0]))
        if model.model_type == "float32":
            img_array = np.array(model_img, dtype=np.float32) / 255.0
        else:
            img_array = np.array(model_img, dtype=np.uint8)
        img_array = np.expand_dims(img_array, axis=0)

        output = model.predict(img_array)
        exp_out = np.exp(output - np.max(output))
        probs = exp_out / np.sum(exp_out)

        pred_idx = np.argmax(probs)
        pred_label = labels[pred_idx]
        confidence = float(probs[pred_idx] * 100)

        # ส่งไป Telegram
        img_buffer = io.BytesIO()
        original_img.save(img_buffer, format='JPEG', quality=85)
        img_data = img_buffer.getvalue()

        if pred_label != "nottarget" and confidence > 70:
            caption = f"🚨 ALERT: {pred_label.upper()} {confidence:.1f}%"
            send_photo(img_data, caption)
        else:
            caption = f"✅ Clear: {pred_label} {confidence:.1f}%"
            send_photo(img_data, caption)

        return jsonify({
            "prediction": pred_label,
            "confidence": round(confidence, 1),
            "all_predictions": {
                labels[i]: round(float(probs[i]*100), 1) for i in range(len(labels))
            },
            "time": get_thai_time()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

print("🚀 Starting server...")
model.load_any_model()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

