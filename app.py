# =====================================================
# 🧠 YOLOv8 Float32 Inference API (3-class + nottarget)
# =====================================================
from flask import Flask, request, jsonify
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io, base64, requests, os
from datetime import datetime, timedelta

app = Flask(__name__)

# ==== Load labels (3-class only) ====
def load_labels(file_path="labels.txt"):
    if not os.path.exists(file_path):
        print("⚠️ labels.txt not found, using default 3-class labels")
        return ["cow", "goat", "sheep"]
    with open(file_path, "r") as f:
        return [line.strip() for line in f if line.strip()]

labels = load_labels()
print(f"📄 Loaded labels: {labels}")

# ==== Load TensorFlow Lite ====
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
print(f"🧠 Using backend: {tf_type}")

# ==== Model Loader ====
class SmartModelLoader:
    def __init__(self):
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.loaded = False
        self.model_file = None

    def try_load_model(self, model_path):
        try:
            print(f"🔄 Loading model: {model_path}")
            if not os.path.exists(model_path):
                print(f"❌ Not found: {model_path}")
                return False
            self.interpreter = tflite_module.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.model_file = model_path
            self.loaded = True
            print(f"✅ Model loaded: {model_path}")
            print(f"📐 Input shape: {self.input_details[0]['shape']}")
            print(f"📤 Output shape: {self.output_details[0]['shape']}")
            return True
        except Exception as e:
            print(f"❌ Load failed: {e}")
            return False

model = SmartModelLoader()

# ==== Telegram Config ====
TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def get_thai_time():
    return (datetime.utcnow() + timedelta(hours=7)).strftime('%H:%M:%S')

def send_photo(image_bytes, caption=""):
    if not TOKEN or not CHAT_ID:
        print("⚠️ Telegram not configured")
        return False
    try:
        url = f"https://api.telegram.org/bot{TOKEN}/sendPhoto"
        files = {'photo': ('image.jpg', image_bytes, 'image/jpeg')}
        data = {'chat_id': CHAT_ID, 'caption': caption, 'parse_mode': 'HTML'}
        requests.post(url, files=files, data=data, timeout=15)
        return True
    except Exception as e:
        print("❌ Telegram error:", e)
        return False

# ==== Draw Bounding Boxes ====
def draw_boxes(image, detections, color=(255, 0, 0)):
    draw = ImageDraw.Draw(image)
    W, H = image.size
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()

    for det in detections:
        x, y, w, h = det["bbox"]
        conf = det["confidence"]
        label = det["label"]

        # แปลงจากสัดส่วน YOLO → พิกเซลจริง
        x1 = int((x - w / 2) * W)
        y1 = int((y - h / 2) * H)
        x2 = int((x + w / 2) * W)
        y2 = int((y + h / 2) * H)

        # ไม่วาดกรอบถ้า label = nottarget
        if label == "nottarget":
            continue

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        caption = f"{label} {conf:.1f}%"

        # ✅ รองรับ Pillow 10+
        bbox = draw.textbbox((x1, y1), caption, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        draw.rectangle([x1, y1 - text_h, x1 + text_w + 4, y1], fill=color)
        draw.text((x1 + 2, y1 - text_h), caption, fill="white", font=font)
    return image

# ==== Routes ====
@app.route("/")
def home():
    available = [f for f in os.listdir('.') if f.endswith('.tflite')]
    return jsonify({
        "status": "running",
        "tensorflow": tf_type,
        "model_loaded": model.loaded,
        "model_file": model.model_file,
        "labels": labels,
        "time": get_thai_time(),
        "available_models": available,
        "telegram_ready": bool(TOKEN and CHAT_ID)
    })

@app.route("/load-model")
def load_model():
    success = model.try_load_model("best_float32.tflite")
    return jsonify({
        "success": success,
        "model_file": model.model_file if success else None
    })

# ==== YOLOv8 Prediction ====
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if not model.loaded:
            if not model.try_load_model("best_float32.tflite"):
                return jsonify({"error": "No model loaded"}), 500

        if not request.json or "image" not in request.json:
            return jsonify({"error": "No image provided"}), 400

        # Decode image
        img_base64 = request.json["image"]
        img_bytes = base64.b64decode(img_base64)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # ✅ Resize ให้ตรงกับโมเดล (320×320)
        IMG_SIZE = 320
        img_resized = img.resize((IMG_SIZE, IMG_SIZE))
        input_data = np.expand_dims(np.array(img_resized, dtype=np.float32) / 255.0, axis=0)

        # Inference
        model.interpreter.set_tensor(model.input_details[0]['index'], input_data)
        model.interpreter.invoke()
        output_data = model.interpreter.get_tensor(model.output_details[0]['index'])

        # ✅ บังคับให้ output เป็น 2D array เสมอ
        output = np.array(output_data)
        if output.ndim == 3:
            output = np.squeeze(output, axis=0)
        elif output.ndim == 1:
            output = np.expand_dims(output, axis=0)

        detections = []

        # Case 1: (x, y, w, h, conf, cls)
        if output.shape[-1] == 6:
            for det in output:
                if len(det) != 6:
                    continue
                x, y, w, h, conf, cls = det
                if conf > 0.3:
                    label = labels[min(int(cls), len(labels) - 1)]
                    detections.append({
                        "bbox": [float(x), float(y), float(w), float(h)],
                        "confidence": round(float(conf) * 100, 1),
                        "label": label
                    })

        # Case 2: raw (x, y, w, h, class_scores...)
        elif output.shape[-1] > 6:
            for det in output:
                if len(det) < 5:
                    continue
                x, y, w, h = det[:4]
                class_scores = 1 / (1 + np.exp(-det[4:]))  # sigmoid
                cls = int(np.argmax(class_scores))
                conf = float(class_scores[cls])

                if cls >= len(labels) or conf < 0.3:
                    label = "nottarget"
                else:
                    label = labels[cls]

                detections.append({
                    "bbox": [float(x), float(y), float(w), float(h)],
                    "confidence": round(conf * 100, 1),
                    "label": label
                })
        else:
            return jsonify({"error": f"Unsupported output shape: {output.shape}"}), 500

        # Draw boxes
        img_drawn = img.copy()
        if detections:
            img_drawn = draw_boxes(img_drawn, detections)

        # Convert to bytes
        img_buffer = io.BytesIO()
        img_drawn.save(img_buffer, format='JPEG', quality=85)
        photo_bytes = img_buffer.getvalue()

        # Telegram
        if detections:
            best = max(detections, key=lambda x: x["confidence"])
            if best['label'] == "nottarget":
                caption = f"✅ Clear area ({best['confidence']}%)"
            else:
                caption = f"🚨 Detected: {best['label'].upper()} ({best['confidence']}%)"
        else:
            caption = "✅ No animals detected"

        send_photo(photo_bytes, caption)

        return jsonify({
            "detections": detections,
            "count": len(detections),
            "time": get_thai_time()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==== Start ====
print("🚀 Starting YOLOv8 Detection API server...")
model.try_load_model("best_float32.tflite")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
