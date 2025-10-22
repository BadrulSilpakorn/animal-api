# =====================================================
# 🧠 YOLOv8 Float32 Inference API (แก้ไข output parsing)
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
        self.input_size = 320

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
            
            input_shape = self.input_details[0]['shape']
            self.input_size = input_shape[1]
            
            self.loaded = True
            print(f"✅ Model loaded: {model_path}")
            print(f"📐 Input shape: {input_shape}")
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

# ==== NMS (Non-Maximum Suppression) ====
def nms(boxes, scores, iou_threshold=0.5):
    """Simple NMS implementation"""
    if len(boxes) == 0:
        return []
    
    boxes = np.array(boxes)
    scores = np.array(scores)
    
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    
    return keep

# ==== Draw Bounding Boxes ====
def draw_boxes(image, detections, color=(255, 0, 0)):
    draw = ImageDraw.Draw(image)
    W, H = image.size
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except:
        font = ImageFont.load_default()

    for det in detections:
        x, y, w, h = det["bbox"]
        conf = det["confidence"]
        label = det["label"]

        x1 = int((x - w / 2) * W)
        y1 = int((y - h / 2) * H)
        x2 = int((x + w / 2) * W)
        y2 = int((y + h / 2) * H)

        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        caption = f"{label} {conf:.1f}%"

        bbox = draw.textbbox((x1, y1), caption, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        draw.rectangle([x1, y1 - text_h - 2, x1 + text_w + 4, y1], fill=color)
        draw.text((x1 + 2, y1 - text_h - 2), caption, fill="white", font=font)
    
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
        "input_size": model.input_size if model.loaded else None,
        "labels": labels,
        "time": get_thai_time(),
        "available_models": available,
        "telegram_ready": bool(TOKEN and CHAT_ID)
    })

@app.route("/load-model")
def load_model_route():
    success = model.try_load_model("best_float32.tflite")
    return jsonify({
        "success": success,
        "model_file": model.model_file if success else None
    })

# ==== YOLOv8 Prediction (แก้ไขตรงนี้) ====
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
        img_original = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        
        original_size = img_original.size
        print(f"📷 Original: {original_size[0]}×{original_size[1]}")

        # Resize
        IMG_SIZE = model.input_size
        img_resized = img_original.resize((IMG_SIZE, IMG_SIZE))
        input_data = np.expand_dims(np.array(img_resized, dtype=np.float32) / 255.0, axis=0)

        # Inference
        model.interpreter.set_tensor(model.input_details[0]['index'], input_data)
        model.interpreter.invoke()
        output_data = model.interpreter.get_tensor(model.output_details[0]['index'])

        print(f"📊 Raw output shape: {output_data.shape}")

        # ✅ แก้ไข: รองรับ YOLOv8 output [1, 8400, 7] หรือ [1, 7, 8400]
        output = np.squeeze(output_data)  # ลบ batch dimension
        
        # ถ้า shape เป็น [7, 8400] → transpose เป็น [8400, 7]
        if output.shape[0] < output.shape[1]:
            output = output.T
        
        print(f"📊 Processed shape: {output.shape}")

        num_classes = len(labels)
        detections = []
        boxes = []
        scores = []
        class_ids = []

        # ✅ Parse YOLOv8 output: [x, y, w, h, class1_score, class2_score, class3_score]
        for det in output:
            if len(det) < 4 + num_classes:
                continue
            
            x, y, w, h = det[:4]
            class_scores = det[4:4+num_classes]
            
            # Sigmoid activation
            class_scores = 1 / (1 + np.exp(-class_scores))
            
            cls_idx = int(np.argmax(class_scores))
            conf = float(class_scores[cls_idx])
            
            # ✅ กรองตาม threshold และขนาดกรอบ
            if conf > 0.5 and w > 0.01 and h > 0.01:  # เพิ่ม threshold
                boxes.append([x, y, w, h])
                scores.append(conf)
                class_ids.append(cls_idx)

        print(f"🔍 Before NMS: {len(boxes)} boxes")

        # ✅ Apply NMS
        if len(boxes) > 0:
            keep_indices = nms(boxes, scores, iou_threshold=0.45)
            
            for idx in keep_indices:
                detections.append({
                    "bbox": [float(v) for v in boxes[idx]],
                    "confidence": round(scores[idx] * 100, 1),
                    "label": labels[class_ids[idx]]
                })

        print(f"✅ After NMS: {len(detections)} detections")

        # Draw boxes on resized image
        img_drawn = img_resized.copy()
        if detections:
            img_drawn = draw_boxes(img_drawn, detections)

        # Convert to bytes
        img_buffer = io.BytesIO()
        img_drawn.save(img_buffer, format='JPEG', quality=90)
        photo_bytes = img_buffer.getvalue()

        # Telegram
        if detections:
            best = max(detections, key=lambda x: x["confidence"])
            caption = (
                f"🚨 <b>Detected: {best['label'].upper()}</b>\n"
                f"📊 Confidence: {best['confidence']}%\n"
                f"🔢 Total: {len(detections)}\n"
                f"📐 {original_size[0]}×{original_size[1]} → {IMG_SIZE}×{IMG_SIZE}\n"
                f"⏰ {get_thai_time()}"
            )
        else:
            caption = (
                f"✅ <b>No animals detected</b>\n"
                f"📐 {original_size[0]}×{original_size[1]} → {IMG_SIZE}×{IMG_SIZE}\n"
                f"⏰ {get_thai_time()}"
            )

        send_photo(photo_bytes, caption)

        return jsonify({
            "detections": detections,
            "count": len(detections),
            "original_size": list(original_size),
            "model_input_size": IMG_SIZE,
            "time": get_thai_time()
        })

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


# ==== Start ====
print("🚀 Starting YOLOv8 Detection API server...")
model.try_load_model("best_float32.tflite")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
