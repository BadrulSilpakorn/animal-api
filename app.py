# =====================================================
# 🧠 YOLOv8 Float32 (Fixed - proper output parsing)
# =====================================================
from flask import Flask, request, jsonify
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io, base64, requests, os
from datetime import datetime, timedelta

app = Flask(__name__)

# ==== Config ====
CONFIDENCE_THRESHOLD = 0.6  # ✅ เพิ่มเป็น 60%
IOU_THRESHOLD = 0.45
MAX_DETECTIONS = 20  # ✅ จำกัดจำนวน detection สูงสุด

# ==== Load labels ====
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
            self.input_size = int(input_shape[1])
            
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

# ==== NMS ====
def nms(boxes, scores, iou_threshold=0.45):
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

# ==== Draw Boxes ====
def draw_boxes(image, detections):
    draw = ImageDraw.Draw(image)
    W, H = image.size
    
    colors = {
        'cow': (255, 0, 0),      # แดง
        'goat': (0, 255, 0),     # เขียว
        'sheep': (0, 0, 255)     # น้ำเงิน
    }
    
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except:
        font = ImageFont.load_default()

    for det in detections:
        x, y, w, h = det["bbox"]
        conf = det["confidence"]
        label = det["label"]
        
        color = colors.get(label, (255, 255, 0))

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
        "confidence_threshold": CONFIDENCE_THRESHOLD,
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

# ==== Prediction ====
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if not model.loaded:
            if not model.try_load_model("best_float32.tflite"):
                return jsonify({"error": "No model loaded"}), 500

        if not request.json or "image" not in request.json:
            return jsonify({"error": "No image provided"}), 400

        img_base64 = request.json["image"]
        img_bytes = base64.b64decode(img_base64)
        img_original = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        
        original_size = img_original.size
        print(f"\n📷 Original: {original_size[0]}×{original_size[1]}")

        IMG_SIZE = model.input_size
        img_resized = img_original.resize((IMG_SIZE, IMG_SIZE))
        input_data = np.expand_dims(np.array(img_resized, dtype=np.float32) / 255.0, axis=0)

        # Inference
        model.interpreter.set_tensor(model.input_details[0]['index'], input_data)
        model.interpreter.invoke()
        output_data = model.interpreter.get_tensor(model.output_details[0]['index'])

        print(f"📊 Raw output shape: {output_data.shape}")
        print(f"📊 Output dtype: {output_data.dtype}")
        print(f"📊 Output range: [{output_data.min():.3f}, {output_data.max():.3f}]")

        # ✅ Process output
        output = np.squeeze(output_data)
        
        # ถ้า shape เป็น [C, N] → transpose เป็น [N, C]
        if len(output.shape) == 2 and output.shape[0] < output.shape[1]:
            output = output.T
            print(f"📊 Transposed to: {output.shape}")
        
        num_classes = len(labels)
        expected_features = 4 + num_classes  # x,y,w,h + 3 classes
        
        print(f"📊 Expected features: {expected_features}, Got: {output.shape[-1]}")

        detections = []
        boxes = []
        scores = []
        class_ids = []

        # ✅ Parse detections
        for i, det in enumerate(output):
            if len(det) < expected_features:
                continue
            
            # YOLOv8 format: [x, y, w, h, class1, class2, class3]
            x, y, w, h = det[0], det[1], det[2], det[3]
            class_scores = det[4:4+num_classes]
            
            # ✅ Sigmoid activation
            class_scores = 1 / (1 + np.exp(-class_scores))
            
            cls_idx = int(np.argmax(class_scores))
            conf = float(class_scores[cls_idx])
            
            # ✅ กรองด้วย threshold + ขนาดกรอบที่สมเหตุสมผล
            if (conf > CONFIDENCE_THRESHOLD and 
                0.02 < w < 0.95 and  # กรอบต้องมีความกว้าง 2-95%
                0.02 < h < 0.95 and  # กรอบต้องมีความสูง 2-95%
                0 < x < 1 and 
                0 < y < 1):
                
                boxes.append([float(x), float(y), float(w), float(h)])
                scores.append(float(conf))
                class_ids.append(int(cls_idx))

        print(f"🔍 Before NMS: {len(boxes)} boxes")

        # ✅ Apply NMS
        if len(boxes) > 0:
            keep_indices = nms(boxes, scores, iou_threshold=IOU_THRESHOLD)
            keep_indices = keep_indices[:MAX_DETECTIONS]  # จำกัดจำนวน
            
            for idx in keep_indices:
                detections.append({
                    "bbox": boxes[idx],
                    "confidence": round(scores[idx] * 100, 1),
                    "label": labels[class_ids[idx]]
                })

        print(f"✅ After NMS: {len(detections)} detections")
        
        # แสดง detection ที่ผ่าน
        for det in detections:
            print(f"  → {det['label']}: {det['confidence']}% at {det['bbox']}")

        # Draw boxes
        img_drawn = img_resized.copy()
        if detections:
            img_drawn = draw_boxes(img_drawn, detections)

        img_buffer = io.BytesIO()
        img_drawn.save(img_buffer, format='JPEG', quality=90)
        photo_bytes = img_buffer.getvalue()

        # Telegram
        if detections:
            best = max(detections, key=lambda x: x["confidence"])
            count_by_class = {}
            for det in detections:
                count_by_class[det['label']] = count_by_class.get(det['label'], 0) + 1
            
            summary = ", ".join([f"{k}:{v}" for k, v in count_by_class.items()])
            
            caption = (
                f"🚨 <b>Detected: {best['label'].upper()}</b>\n"
                f"📊 Confidence: {best['confidence']}%\n"
                f"🔢 Count: {summary}\n"
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
            "original_size": [int(original_size[0]), int(original_size[1])],
            "model_input_size": int(IMG_SIZE),
            "time": get_thai_time()
        })

    except Exception as e:
        import traceback
        print("❌ Error details:")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

print("🚀 Starting YOLOv8 Detection API server...")
model.try_load_model("best_float32.tflite")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
