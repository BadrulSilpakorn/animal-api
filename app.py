from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import io, base64, os, requests
from datetime import datetime, timedelta

# ===== Backend: tflite-runtime (float32-only) =====
try:
    import tflite_runtime.interpreter as tflite
    TF_BACKEND = "tflite-runtime"
except ImportError:
    # fallback ถ้าไม่มี tflite-runtime (ไม่แนะนำบน Render)
    import tensorflow as tf
    tflite = tf.lite
    TF_BACKEND = "tensorflow"

app = Flask(__name__)

# ===== Env & Config =====
MODEL_PATH   = os.getenv("MODEL_PATH", "animal_model_float32_v1.tflite")
LABELS_PATH  = os.getenv("LABELS_PATH", "labels.txt")
# ถ้าโมเดลของคุณ "ไม่มี" Rescaling(1/255) ในกราฟ ให้ตั้ง PRE_DIV255=1
PRE_DIV255   = os.getenv("PRE_DIV255", "0").lower() in ("1", "true", "yes")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT  = os.getenv("TELEGRAM_CHAT_ID") or os.getenv("CHAT_ID") or os.getenv("TELEGRAM_CHAT")

def now_th():
    return (datetime.utcnow() + timedelta(hours=7)).strftime("%Y-%m-%d %H:%M:%S")

def load_labels(path=LABELS_PATH):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            labs = [ln.strip() for ln in f if ln.strip()]
        if labs:
            return labs
    # Fallback ถ้าไม่มี labels.txt
    return ["nottarget", "cow", "goat", "sheep"]

LABELS = load_labels()

def send_telegram(text, image_bytes=None):
    if not (TELEGRAM_TOKEN and TELEGRAM_CHAT):
        return False
    try:
        # message
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            data={"chat_id": TELEGRAM_CHAT, "text": text, "parse_mode": "HTML"},
            timeout=10,
        )
        # photo (optional)
        if image_bytes:
            requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto",
                data={"chat_id": TELEGRAM_CHAT},
                files={"photo": ("img.jpg", image_bytes, "image/jpeg")},
                timeout=15,
            )
        return True
    except Exception as e:
        print("❌ Telegram error:", e)
        return False

class TFLiteModel:
    def __init__(self, path):
        self.path = path
        self.interpreter = None
        self.inp = None
        self.out = None
        self.loaded = False

    def load(self):
        try:
            if not os.path.exists(self.path):
                print(f"❌ Model not found: {self.path}")
                return False
            print(f"🔄 Loading model: {self.path}")
            self.interpreter = tflite.Interpreter(model_path=self.path)
            self.interpreter.allocate_tensors()
            self.inp = self.interpreter.get_input_details()
            self.out = self.interpreter.get_output_details()
            self.loaded = True
            print("✅ Model loaded")
            print("📐 Input:", self.inp[0]["shape"], self.inp[0]["dtype"])
            print("📤 Output:", self.out[0]["shape"], self.out[0]["dtype"])
            return True
        except Exception as e:
            print("❌ Load error:", e)
            return False

    def predict(self, x):
        if not self.loaded:
            raise RuntimeError("Model not loaded")
        self.interpreter.set_tensor(self.inp[0]["index"], x)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.out[0]["index"])[0]

model = TFLiteModel(MODEL_PATH)
model.load()

def to_probs(vec):
    """รับเวกเตอร์เอาต์พุตจาก TFLite: 
       - ถ้าผลรวม ~1 และทุกค่า >=0 → ถือว่าเป็น probs แล้ว (ไม่ softmax ซ้ำ)
       - ไม่งั้นค่อย softmax."""
    v = np.asarray(vec, dtype=np.float32).flatten()
    s = float(np.sum(v))
    if np.all(v >= 0.0) and 0.98 <= s <= 1.02:
        return v / (s if s != 0 else 1.0)
    # logits → softmax
    v = v - np.max(v)
    e = np.exp(v, dtype=np.float32)
    return e / np.sum(e)

def preprocess_pil(img: Image.Image, want_h, want_w, dtype):
    # resize แบบยืดให้พอดี (ถ้าอยากคงสัดส่วน ให้ไป letterbox เอง)
    resized = img.resize((want_w, want_h), Image.Resampling.LANCZOS)
    arr = np.array(resized)
    # จัด dtype & scale ตามโมเดล
    if np.issubdtype(dtype, np.floating):
        # ค่าเริ่มต้น: "ไม่หาร 255" เพราะโมเดลมี Rescaling(1/255) แล้ว
        x = arr.astype(np.float32)
        if PRE_DIV255:
            x = x / 255.0
    elif np.issubdtype(dtype, np.integer):
        # ไม่ควรเกิดกับโมเดล float32 ของคุณ แต่รองรับไว้
        x = arr.astype(np.uint8)
    else:
        x = arr.astype(np.float32)
    x = np.expand_dims(x, axis=0)
    return x

@app.route("/")
def root():
    tflite_files = [f for f in os.listdir(".") if f.endswith(".tflite")]
    return jsonify({
        "status": "running",
        "time_th": now_th(),
        "backend": TF_BACKEND,
        "model_loaded": model.loaded,
        "model_file": model.path,
        "input_shape": (model.inp[0]["shape"].tolist() if model.loaded else None),
        "input_dtype": (str(model.inp[0]["dtype"]) if model.loaded else None),
        "labels": LABELS,
        "pre_div255": PRE_DIV255,
        "available_models": tflite_files,
        "telegram_ready": bool(TELEGRAM_TOKEN and TELEGRAM_CHAT),
    })

@app.route("/test-telegram")
def test_telegram():
    ok = send_telegram(f"🛰️ Render OK at {now_th()} (labels={LABELS})")
    return jsonify({"sent": ok})

@app.route("/predict", methods=["POST"])
def predict_route():
    try:
        if not model.loaded:
            return jsonify({"error": "Model not loaded"}), 500
        if not request.is_json or "image" not in request.json:
            return jsonify({"error": "Send JSON with field 'image' (base64)"}), 400

        # Decode base64 → PIL
        b64 = request.json["image"]
        try:
            img_bytes = base64.b64decode(b64, validate=True)
            pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception as e:
            return jsonify({"error": f"Invalid image/base64: {e}"}), 400

        # Preprocess
        h, w = int(model.inp[0]["shape"][1]), int(model.inp[0]["shape"][2])
        x = preprocess_pil(pil, h, w, model.inp[0]["dtype"])

        # Predict
        raw = model.predict(x)
        probs = to_probs(raw)

        # Labels safety
        n_class = min(len(LABELS), probs.shape[0])
        probs = probs[:n_class]
        labels = LABELS[:n_class]

        top_idx = int(np.argmax(probs))
        top_label = labels[top_idx]
        top_conf = float(probs[top_idx] * 100.0)

        # top-3
        order = np.argsort(-probs)
        top3 = [{"label": labels[i], "conf": round(float(probs[i]*100), 1)} for i in order[:3]]

        # Telegram (resize/บีบอัดก่อนส่ง)
        buf = io.BytesIO()
        send_preview = pil.copy()
        if max(send_preview.size) > 960:
            send_preview.thumbnail((960, 960), Image.Resampling.LANCZOS)
        send_preview.save(buf, format="JPEG", quality=85, optimize=True)
        caption = (
            f"🤖 <b>Animal Detection</b>\n"
            f"🎯 <b>{top_label}</b> ({top_conf:.1f}%)\n"
            f"📊 Top-3: " + ", ".join([f"{t['label']} {t['conf']}%" for t in top3]) + "\n"
            f"⏰ {now_th()}"
        )
        send_telegram(caption, buf.getvalue())

        return jsonify({
            "status": "success",
            "time_th": now_th(),
            "prediction": top_label,
            "confidence": round(top_conf, 1),
            "top3": top3,
            "all": {labels[i]: round(float(probs[i]*100), 1) for i in range(n_class)},
            "model_info": {
                "file": model.path,
                "backend": TF_BACKEND,
                "input_shape": model.inp[0]["shape"].tolist(),
                "input_dtype": str(model.inp[0]["dtype"]),
                "pre_div255": PRE_DIV255
            }
        })

    except Exception as e:
        msg = f"❌ ERROR: {e}\n⏰ {now_th()}"
        print(msg)
        send_telegram(msg)
        return jsonify({"status": "error", "error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
