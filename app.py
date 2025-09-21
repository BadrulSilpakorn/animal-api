from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import io
import base64
import tflite_runtime.interpreter as tflite
import requests
import os
from datetime import datetime, timedelta

app = Flask(__name__)

# โหลดโมเดล TFLite
interpreter = None
try:
    interpreter = tflite.Interpreter(model_path="animal_model_int8.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("✅ Model loaded")
except:
    print("❌ Model loading failed")

# Telegram config
TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def get_thai_time():
    """เวลาไทย UTC+7"""
    return (datetime.utcnow() + timedelta(hours=7)).strftime('%H:%M:%S')

def send_message(text):
    """ส่งข้อความ Telegram"""
    try:
        url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
        data = {"chat_id": CHAT_ID, "text": text, "parse_mode": "HTML"}
        response = requests.post(url, data=data, timeout=5)
        return response.status_code == 200
    except:
        return False

def send_photo(image_bytes, caption=""):
    """ส่งรูป Telegram"""
    try:
        url = f"https://api.telegram.org/bot{TOKEN}/sendPhoto"
        files = {'photo': ('img.jpg', image_bytes, 'image/jpeg')}
        data = {'chat_id': CHAT_ID, 'caption': caption}
        response = requests.post(url, files=files, data=data, timeout=10)
        return response.status_code == 200
    except:
        return False

@app.route("/")
def home():
    return jsonify({
        "status": "running",
        "model": interpreter is not None,
        "time": get_thai_time()
    })

@app.route("/test")
def test():
    """ทดสอบ Telegram"""
    result = send_message(f"✅ Bot working! Time: {get_thai_time()}")
    return jsonify({"sent": result})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if not interpreter:
            return jsonify({"error": "Model not loaded"}), 500
            
        if not request.json or "image" not in request.json:
            return jsonify({"error": "No image"}), 400

        # Decode image
        img_base64 = request.json["image"]
        img_bytes = base64.b64decode(img_base64)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        
        # Resize for model
        target_size = input_details[0]['shape'][1:3]
        img_resized = img.resize((target_size[1], target_size[0]))
        img_array = np.expand_dims(np.array(img_resized, dtype=np.uint8), axis=0)
        
        # Predict
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])[0]
        
        # Process results
        labels = ["safe", "cow", "goat", "sheep"]
        
        # Simple softmax
        exp_out = np.exp(output - np.max(output))
        probs = exp_out / np.sum(exp_out)
        
        pred_idx = np.argmax(probs)
        pred_label = labels[pred_idx]
        confidence = float(probs[pred_idx] * 100)
        
        print(f"🎯 {pred_label}: {confidence:.1f}%")
        
        # Prepare clean image
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='JPEG', quality=85)
        clean_img = img_buffer.getvalue()
        
        # Send to Telegram
        thai_time = get_thai_time()
        
        if pred_label != "safe" and confidence > 70:
            # Alert!
            send_photo(clean_img, "🚨 Animal Detected!")
            
            alert = f"🚨 <b>ALERT!</b>\n"
            alert += f"🐄 Animal: <b>{pred_label.upper()}</b>\n"
            alert += f"📊 Confidence: <b>{confidence:.1f}%</b>\n"
            alert += f"⏰ Time: <b>{thai_time}</b>"
            
            msg_sent = send_message(alert)
        else:
            # Safe
            send_photo(clean_img, "✅ Area Clear")
            
            safe_msg = f"✅ <b>All Clear</b>\n"
            safe_msg += f"📊 Confidence: <b>{confidence:.1f}%</b>\n"
            safe_msg += f"⏰ Time: <b>{thai_time}</b>"
            
            msg_sent = send_message(safe_msg)
        
        return jsonify({
            "prediction": pred_label,
            "confidence": round(confidence, 1),
            "time": thai_time,
            "alert": pred_label != "safe" and confidence > 70,
            "sent": msg_sent
        })
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
