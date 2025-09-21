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

# โหลดโมเดล TFLite Float32 🔄
interpreter = None
try:
    # 🔄 เปลี่ยนเป็น float32 model
    interpreter = tflite.Interpreter(model_path="animal_model_float32.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # ตรวจสอบ input details
    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']
    print(f"✅ Float32 Model loaded")
    print(f"📐 Input shape: {input_shape}")
    print(f"🔢 Input dtype: {input_dtype}")
    
except Exception as e:
    print(f"❌ Model loading failed: {e}")

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
        response = requests.post(url, data=data, timeout=8)
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Send message error: {e}")
        return False

def send_photo(image_bytes, caption=""):
    """ส่งรูปต้นฉบับ Telegram"""
    try:
        url = f"https://api.telegram.org/bot{TOKEN}/sendPhoto"
        files = {'photo': ('original.jpg', image_bytes, 'image/jpeg')}
        data = {'chat_id': CHAT_ID, 'caption': caption, 'parse_mode': 'HTML'}
        response = requests.post(url, files=files, data=data, timeout=15)
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Send photo error: {e}")
        return False

@app.route("/")
def home():
    model_info = "Not loaded"
    if interpreter:
        try:
            input_shape = input_details[0]['shape']
            input_dtype = str(input_details[0]['dtype'])
            model_info = f"Float32 {input_shape}"
        except:
            model_info = "Loaded"
    
    return jsonify({
        "status": "running",
        "model": model_info,
        "model_type": "float32",
        "time": get_thai_time(),
        "telegram_configured": TOKEN is not None and CHAT_ID is not None
    })

@app.route("/test")
def test():
    """ทดสอบ Telegram Bot"""
    thai_time = get_thai_time()
    test_msg = f"🤖 <b>Float32 Model Test</b>\n⏰ Time: <b>{thai_time}</b>\n✅ Bot is working!"
    result = send_message(test_msg)
    return jsonify({
        "sent": result,
        "time": thai_time,
        "message": "Test message sent" if result else "Failed to send"
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if not interpreter:
            return jsonify({"error": "Float32 model not loaded"}), 500
            
        if not request.json or "image" not in request.json:
            return jsonify({"error": "No image data provided"}), 400

        # 🔄 Decode และเก็บรูปต้นฉบับ
        img_base64 = request.json["image"]
        original_img_bytes = base64.b64decode(img_base64)
        original_img = Image.open(io.BytesIO(original_img_bytes)).convert("RGB")
        
        print(f"📸 Original image size: {original_img.size}")
        
        # 🔄 เตรียมรูปสำหรับโมเดล Float32 (224x224)
        target_size = input_details[0]['shape'][1:3]  # [224, 224]
        model_img = original_img.resize((target_size[1], target_size[0]))
        
        # 🔄 แปลงเป็น float32 array [0.0-1.0]
        img_array = np.array(model_img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        print(f"🤖 Model input shape: {img_array.shape}, dtype: {img_array.dtype}")
        print(f"📊 Input range: [{img_array.min():.3f}, {img_array.max():.3f}]")
        
        # 🔄 ทำนายด้วย Float32 model
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])[0]
        
        # Process results
        labels = ["safe", "cow", "goat", "sheep"]
        
        # 🔄 Softmax สำหรับ float32 output
        if output_details[0]['dtype'] == np.float32:
            # Output เป็น logits แล้ว
            exp_out = np.exp(output - np.max(output))
            probs = exp_out / np.sum(exp_out)
        else:
            # ถ้าเป็น quantized output
            probs = output.astype(np.float32) / 255.0
        
        pred_idx = np.argmax(probs)
        pred_label = labels[pred_idx]
        confidence = float(probs[pred_idx] * 100)
        
        print(f"🎯 Prediction: {pred_label} ({confidence:.1f}%)")
        print(f"📊 All probabilities: {[f'{labels[i]}:{probs[i]*100:.1f}%' for i in range(len(labels))]}")
        
        # 🔄 เตรียมรูปต้นฉบับสำหรับส่ง Telegram
        img_buffer = io.BytesIO()
        # ปรับขนาดให้เหมาะสมกับ Telegram (ไม่ใหญ่เกินไป)
        if original_img.size[0] > 1280 or original_img.size[1] > 1280:
            # Resize ถ้าใหญ่เกินไป แต่รักษาอัตราส่วน
            original_img.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
        
        original_img.save(img_buffer, format='JPEG', quality=90, optimize=True)
        clean_img = img_buffer.getvalue()
        
        print(f"📱 Telegram image size: {len(clean_img)/1024:.1f} KB")
        
        # Send to Telegram
        thai_time = get_thai_time()
        
        # 🔄 ปรับเกณฑ์การแจ้งเตือน
        ALERT_THRESHOLD = 75.0  # เพิ่มเป็น 75% สำหรับความแม่นยำสูง
        
        if pred_label != "safe" and confidence > ALERT_THRESHOLD:
            # 🚨 Alert case
            photo_caption = f"🚨 <b>ANIMAL DETECTED!</b>\n🐄 {pred_label.upper()}: {confidence:.1f}%"
            photo_sent = send_photo(clean_img, photo_caption)
            
            alert_msg = f"🚨 <b>HIGH PRIORITY ALERT!</b>\n\n"
            alert_msg += f"🐄 <b>Animal Type:</b> {pred_label.upper()}\n"
            alert_msg += f"📊 <b>Confidence:</b> {confidence:.1f}%\n"
            alert_msg += f"📍 <b>Location:</b> Farm Camera\n"
            alert_msg += f"⏰ <b>Time:</b> {thai_time}\n\n"
            alert_msg += f"🔍 <b>All Predictions:</b>\n"
            for i, label in enumerate(labels):
                alert_msg += f"   • {label}: {probs[i]*100:.1f}%\n"
            
            msg_sent = send_message(alert_msg)
            
        else:
            # ✅ Safe case
            photo_caption = f"✅ <b>AREA SECURE</b>\n📊 Safe: {confidence:.1f}%"
            photo_sent = send_photo(clean_img, photo_caption)
            
            safe_msg = f"✅ <b>All Clear - Area Secure</b>\n\n"
            safe_msg += f"📊 <b>Safety Confidence:</b> {confidence:.1f}%\n"
            safe_msg += f"📍 <b>Location:</b> Farm Camera\n"
            safe_msg += f"⏰ <b>Time:</b> {thai_time}\n\n"
            
            # แสดงรายละเอียดเฉพาะเมื่อมีสัตว์แต่ confidence ต่ำ
            if pred_label != "safe":
                safe_msg += f"⚠️ <b>Low confidence detection:</b>\n"
                safe_msg += f"   • {pred_label}: {confidence:.1f}%\n"
            
            msg_sent = send_message(safe_msg)
        
        # Return API response
        return jsonify({
            "status": "success",
            "prediction": pred_label,
            "confidence": round(confidence, 2),
            "all_predictions": {
                labels[i]: round(float(probs[i] * 100), 2) 
                for i in range(len(labels))
            },
            "alert": pred_label != "safe" and confidence > ALERT_THRESHOLD,
            "threshold": ALERT_THRESHOLD,
            "time": thai_time,
            "telegram": {
                "photo_sent": photo_sent if 'photo_sent' in locals() else False,
                "message_sent": msg_sent if 'msg_sent' in locals() else False
            },
            "model_info": {
                "type": "float32",
                "input_size": target_size.tolist(),
                "original_image_size": list(original_img.size)
            }
        })
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Prediction error: {error_msg}")
        
        # ส่งแจ้ง error ไป Telegram
        try:
            error_alert = f"❌ <b>SYSTEM ERROR</b>\n\n"
            error_alert += f"🚨 <b>Error:</b> {error_msg}\n"
            error_alert += f"⏰ <b>Time:</b> {get_thai_time()}\n"
            error_alert += f"🔧 <b>Please check system!</b>"
            send_message(error_alert)
        except:
            pass
        
        return jsonify({
            "status": "error",
            "error": error_msg,
            "time": get_thai_time()
        }), 500

@app.route("/status")
def status():
    """ตรวจสอบสถานะระบบโดยละเอียด"""
    try:
        system_status = {
            "server": "running",
            "time": get_thai_time(),
            "model": {
                "loaded": interpreter is not None,
                "type": "float32",
                "path": "animal_model_float32.tflite"
            },
            "telegram": {
                "token_configured": TOKEN is not None,
                "chat_id_configured": CHAT_ID is not None,
                "bot_ready": TOKEN is not None and CHAT_ID is not None
            }
        }
        
        if interpreter:
            try:
                system_status["model"]["input_shape"] = input_details[0]['shape'].tolist()
                system_status["model"]["input_dtype"] = str(input_details[0]['dtype'])
                system_status["model"]["output_shape"] = output_details[0]['shape'].tolist()
                system_status["model"]["output_dtype"] = str(output_details[0]['dtype'])
            except:
                pass
        
        return jsonify(system_status)
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e),
            "time": get_thai_time()
        }), 500

if __name__ == "__main__":
    print("🚀 Starting Float32 Animal Detection Server...")
    print(f"📊 Model: animal_model_float32.tflite")
    print(f"🤖 Telegram Bot: {'✅ Ready' if TOKEN and CHAT_ID else '❌ Not configured'}")
    
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
