from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import io
import base64
import tflite_runtime.interpreter as tflite
import requests  # ใช้ requests แทน telegram library
import os

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

def send_telegram_message(text):
    """ส่งข้อความไป Telegram ด้วย requests"""
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
        print(f"📡 Telegram API Response: {response.status_code}")
        print(f"📡 Response body: {response.text}")
        
        if response.status_code == 200:
            print("✅ Message sent successfully")
            return {"success": True, "data": response.json()}
        else:
            print(f"❌ Failed to send message: {response.status_code}")
            return {"success": False, "error": f"HTTP {response.status_code}"}
            
    except Exception as e:
        print(f"❌ Telegram request failed: {e}")
        return {"success": False, "error": str(e)}

@app.route("/")
def home():
    return jsonify({
        "status": "running",
        "message": "✅ TFLite Inference API with Telegram Alert is running",
        "model_loaded": interpreter is not None,
        "telegram_configured": bool(TELEGRAM_TOKEN and TELEGRAM_CHAT_ID)
    })

@app.route("/testbot")
def testbot():
    """ทดสอบส่งข้อความ Telegram"""
    try:
        test_msg = "✅ Render Bot is working! 🚀"
        print(f"📢 Sending test message: {test_msg}")
        
        result = send_telegram_message(test_msg)
        
        if result.get("success"):
            return jsonify({"status": "success", "message": "Message sent to Telegram!"})
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
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            print(f"📷 Image decoded successfully: {img.size}")
        except Exception as e:
            error_msg = f"❌ Failed to decode image: {e}"
            print(error_msg)
            return jsonify({"error": error_msg}), 400

        # resize ตาม input model
        target_shape = input_details[0]['shape'][1:3]  # (height, width)
        img_resized = img.resize((target_shape[1], target_shape[0]))
        img_resized = np.array(img_resized)
        print(f"🔄 Image resized to: {img_resized.shape}")

        # เตรียมข้อมูลสำหรับโมเดล
        img_input = np.expand_dims(img_resized.astype(np.uint8), axis=0)

        # run inference
        print("🤖 Running inference...")
        interpreter.set_tensor(input_details[0]['index'], img_input)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])

        # ประมวลผลผลลัพธ์
        labels = ["nottarget", "cow", "goat", "sheep"]
        pred_idx = int(np.argmax(output))
        pred_label = labels[pred_idx]
        confidence = float(np.max(output))
        
        print(f"🎯 Prediction: {pred_label} (confidence: {confidence:.2f})")

        # สร้างข้อความแจ้งเตือน
        if pred_label != "nottarget":
            message = f"🚨 <b>Intrusion Detected!</b>\n🐄 Animal: <b>{pred_label}</b>\n📊 Confidence: <b>{confidence:.2f}</b>"
            telegram_result = send_telegram_message(message)
        else:
            message = f"✅ <b>No animal detected</b>\n📊 Confidence: <b>{confidence:.2f}</b>"
            telegram_result = send_telegram_message(message)

        # ส่งผลลัพธ์กลับ
        response_data = {
            "prediction": pred_label,
            "confidence": confidence,
            "telegram_sent": telegram_result.get("success", False)
        }
        
        if not telegram_result.get("success"):
            response_data["telegram_error"] = telegram_result.get("error")

        print(f"✅ Prediction completed: {response_data}")
        return jsonify(response_data)

    except Exception as e:
        error_msg = f"❌ Prediction error: {str(e)}"
        print(error_msg)
        
        # ส่งข้อความ error ไป Telegram
        try:
            send_telegram_message(f"❌ <b>API Error:</b>\n{str(e)}")
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
        "endpoints": ["/", "/predict", "/testbot", "/health"]
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
