from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import io
import base64
import tflite_runtime.interpreter as tflite
import requests
import os
import pytz
from datetime import datetime

app = Flask(__name__)

# โหลดโมเดล TFLite
try:
    interpreter = tflite.Interpreter(model_path="animal_model_int8.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    interpreter = None

# Telegram config
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

print(f"🔑 TELEGRAM_TOKEN: {'✅ Set' if TELEGRAM_TOKEN else '❌ Missing'}")
print(f"🔑 TELEGRAM_CHAT_ID: {'✅ Set' if TELEGRAM_CHAT_ID else '❌ Missing'}")

def get_thailand_time():
    """ได้เวลาไทยที่ถูกต้อง"""
    try:
        thailand_tz = pytz.timezone('Asia/Bangkok')
        now = datetime.now(thailand_tz)
        return now.strftime('%Y-%m-%d %H:%M:%S')
    except:
        # fallback ถ้าไม่มี pytz
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def send_telegram_message(text):
    """ส่งข้อความไป Telegram"""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("❌ Telegram credentials missing")
        return {"error": "Missing TELEGRAM_TOKEN or TELEGRAM_CHAT_ID"}

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML"
    }

    try:
        response = requests.post(url, data=data, timeout=10)
        print(f"📡 Message API Response: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ Message sent successfully")
            return {"success": True, "data": response.json()}
        else:
            print(f"❌ Failed to send message: {response.status_code}")
            return {"success": False, "error": f"HTTP {response.status_code}"}
            
    except Exception as e:
        print(f"❌ Message request failed: {e}")
        return {"success": False, "error": str(e)}

def send_telegram_photo(image_bytes, caption=""):
    """ส่งรูปภาพไป Telegram"""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("❌ Telegram credentials missing")
        return {"error": "Missing TELEGRAM_TOKEN or TELEGRAM_CHAT_ID"}

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
    
    files = {
        'photo': ('detection.jpg', image_bytes, 'image/jpeg')
    }
    
    data = {
        'chat_id': TELEGRAM_CHAT_ID,
        'caption': caption,
        'parse_mode': 'HTML'
    }

    try:
        response = requests.post(url, files=files, data=data, timeout=30)
        print(f"📸 Photo API Response: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ Photo sent successfully")
            return {"success": True, "data": response.json()}
        else:
            print(f"❌ Failed to send photo: {response.status_code}")
            return {"success": False, "error": f"HTTP {response.status_code}"}
            
    except Exception as e:
        print(f"❌ Photo request failed: {e}")
        return {"success": False, "error": str(e)}

def softmax(x):
    """คำนวณ softmax เพื่อแปลง logits เป็น probability"""
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

@app.route("/")
def home():
    return jsonify({
        "status": "running",
        "message": "✅ Animal Detection API with Telegram Alert",
        "model_loaded": interpreter is not None,
        "telegram_configured": bool(TELEGRAM_TOKEN and TELEGRAM_CHAT_ID),
        "thailand_time": get_thailand_time()
    })

@app.route("/testbot")
def testbot():
    """ทดสอบส่งข้อความ Telegram"""
    try:
        test_msg = f"✅ Animal Detection Bot is working!\n📸 Photo sending ready\n⏰ Thailand Time: {get_thailand_time()}"
        print(f"📢 Sending test message: {test_msg}")
        
        result = send_telegram_message(test_msg)
        
        if result.get("success"):
            return jsonify({"status": "success", "message": "Test message sent!"})
        else:
            return jsonify({"status": "error", "error": result.get("error")}), 500
            
    except Exception as e:
        print(f"❌ Test bot error: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route("/predict", methods=["POST"])
def predict():
    try:
        print("📸 Received prediction request")
        
        # ตรวจสอบโมเดล
        if interpreter is None:
            error_msg = "❌ Model not loaded"
            print(error_msg)
            return jsonify({"error": "Model not loaded"}), 500

        # รับภาพจาก request
        if not request.json or "image" not in request.json:
            error_msg = "❌ No image data in request"
            print(error_msg)
            return jsonify({"error": "No image data"}), 400

        img_base64 = request.json["image"]
        print(f"📷 Received image data (length: {len(img_base64)})")

        # decode base64 → Image
        try:
            img_bytes = base64.b64decode(img_base64)
            original_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            print(f"📷 Image decoded successfully: {original_img.size}")
        except Exception as e:
            error_msg = f"❌ Failed to decode image: {e}"
            print(error_msg)
            return jsonify({"error": error_msg}), 400

        # resize ตาม input model
        target_shape = input_details[0]['shape'][1:3]  # (height, width)
        img_resized = original_img.resize((target_shape[1], target_shape[0]))
        img_resized_array = np.array(img_resized)
        print(f"🔄 Image resized to: {img_resized_array.shape}")

        # เตรียมข้อมูลสำหรับโมเดล
        img_input = np.expand_dims(img_resized_array.astype(np.uint8), axis=0)

        # run inference
        print("🤖 Running inference...")
        interpreter.set_tensor(input_details[0]['index'], img_input)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])[0]  # ⭐ เอา [0] เพื่อลบ batch dimension

        print(f"🔍 Raw output shape: {output.shape}")
        print(f"🔍 Raw output values: {output}")

        # ประมวลผลผลลัพธ์
        labels = ["nottarget", "cow", "goat", "sheep"]
        
        # แปลง logits เป็น probabilities ด้วย softmax
        probabilities = softmax(output)
        print(f"🔍 Probabilities: {probabilities}")
        
        pred_idx = int(np.argmax(probabilities))
        pred_label = labels[pred_idx]
        confidence = float(probabilities[pred_idx])  # ⭐ ใช้ probability แทน raw output
        
        print(f"🎯 Prediction: {pred_label} (confidence: {confidence:.4f} = {confidence*100:.2f}%)")

        # แปลงรูปต้นฉบับเป็น bytes (ไม่มี overlay)
        img_buffer = io.BytesIO()
        original_img.save(img_buffer, format='JPEG', quality=90)
        clean_img_bytes = img_buffer.getvalue()

        # สร้างข้อความแจ้งเตือน
        thailand_time = get_thailand_time()
        
        if pred_label != "nottarget" and confidence > 0.5:  # เพิ่มเงื่อนไข confidence threshold
            # ส่งรูปภาพก่อน
            photo_result = send_telegram_photo(clean_img_bytes, "📸 Detected Image")
            
            # ส่งข้อความแยกต่างหาก
            alert_msg = f"🚨 <b>Animal Intrusion Alert!</b>\n\n"
            alert_msg += f"🐄 <b>Animal:</b> {pred_label.upper()}\n"
            alert_msg += f"📊 <b>Confidence:</b> {confidence*100:.1f}%\n"
            alert_msg += f"⏰ <b>Detection Time:</b> {thailand_time}\n"
            alert_msg += f"📍 <b>Location:</b> Farm Camera\n\n"
            alert_msg += f"⚠️ Please check the farm immediately!"
            
            message_result = send_telegram_message(alert_msg)
            
        else:
            # ส่งรูปภาพก่อน
            photo_result = send_telegram_photo(clean_img_bytes, "📸 Scan Result")
            
            # ส่งข้อความแยกต่างหาก
            safe_msg = f"✅ <b>Area Scan Complete</b>\n\n"
            safe_msg += f"🔍 <b>Result:</b> No animals detected\n"
            safe_msg += f"📊 <b>Confidence:</b> {confidence*100:.1f}%\n"
            safe_msg += f"⏰ <b>Scan Time:</b> {thailand_time}\n"
            safe_msg += f"📍 <b>Location:</b> Farm Camera\n\n"
            safe_msg += f"🛡️ Farm area is secure"
            
            message_result = send_telegram_message(safe_msg)

        # แสดงความมั่นใจของแต่ละ class
        confidence_breakdown = {}
        for i, label in enumerate(labels):
            confidence_breakdown[label] = float(probabilities[i] * 100)
        
        # ส่งผลลัพธ์กลับ
        response_data = {
            "prediction": pred_label,
            "confidence": round(confidence * 100, 2),  # แปลงเป็น %
            "confidence_breakdown": confidence_breakdown,
            "photo_sent": photo_result.get("success", False),
            "message_sent": message_result.get("success", False),
            "thailand_time": thailand_time,
            "is_alert": pred_label != "nottarget" and confidence > 0.5
        }
        
        if not photo_result.get("success"):
            response_data["photo_error"] = photo_result.get("error")
        if not message_result.get("success"):
            response_data["message_error"] = message_result.get("error")

        print(f"✅ Prediction completed: {response_data}")
        return jsonify(response_data)

    except Exception as e:
        error_msg = f"❌ Prediction error: {str(e)}"
        print(error_msg)
        
        # ส่งข้อความ error ไป Telegram
        try:
            error_alert = f"❌ <b>System Error</b>\n\n"
            error_alert += f"🔧 <b>Error:</b> {str(e)}\n"
            error_alert += f"⏰ <b>Time:</b> {get_thailand_time()}\n"
            error_alert += f"🔄 Please check the system"
            send_telegram_message(error_alert)
        except:
            pass
            
        return jsonify({"error": str(e)}), 500

@app.route("/health")
def health():
    """ตรวจสอบสถานะระบบ"""
    return jsonify({
        "status": "healthy",
        "model_loaded": interpreter is not None,
        "telegram_configured": bool(TELEGRAM_TOKEN and TELEGRAM_CHAT_ID),
        "features": ["clean_image_sending", "separate_text_alerts", "correct_confidence", "thailand_timezone"],
        "endpoints": ["/", "/predict", "/testbot", "/health"],
        "thailand_time": get_thailand_time()
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
