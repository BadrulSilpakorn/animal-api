from flask import Flask, request, jsonify
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import tflite_runtime.interpreter as tflite
import requests
import os
from datetime import datetime  # ⭐ เพิ่มบรรทัดนี้

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

def get_current_time():
    """ได้เวลาปัจจุบัน"""
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

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
        
        if response.status_code == 200:
            print("✅ Message sent successfully")
            return {"success": True, "data": response.json()}
        else:
            print(f"❌ Failed to send message: {response.status_code}")
            return {"success": False, "error": f"HTTP {response.status_code}"}
            
    except Exception as e:
        print(f"❌ Telegram request failed: {e}")
        return {"success": False, "error": str(e)}

def send_telegram_photo(image_bytes, caption=""):
    """ส่งรูปภาพไป Telegram"""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("❌ Telegram credentials missing")
        return {"error": "Missing TELEGRAM_TOKEN or TELEGRAM_CHAT_ID"}

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
    
    files = {
        'photo': ('image.jpg', image_bytes, 'image/jpeg')
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

def add_prediction_overlay(image, prediction, confidence):
    """เพิ่ม overlay ผลการวิเคราะห์บนรูปภาพ"""
    try:
        # สร้างสำเนาของรูปภาพ
        img_copy = image.copy()
        draw = ImageDraw.Draw(img_copy)
        
        # กำหนดสี
        colors = {
            "cow": "#FF4444",      # แดง
            "goat": "#44FF44",     # เขียว  
            "sheep": "#4444FF",    # น้ำเงิน
            "nottarget": "#888888" # เทา
        }
        
        color = colors.get(prediction, "#FFFFFF")
        
        # กำหนดขนาดฟอนต์ตามขนาดรูป
        font_size = max(20, min(img_copy.width, img_copy.height) // 20)
        
        try:
            # ลองใช้ฟอนต์ default
            font = ImageFont.load_default()
        except:
            font = None
        
        # สร้างข้อความ
        if prediction != "nottarget":
            text = f"🚨 {prediction.upper()}"
            status_text = "DETECTED"
            emoji = "🐄" if prediction == "cow" else "🐐" if prediction == "goat" else "🐑"
        else:
            text = "✅ NO ANIMAL"
            status_text = "SAFE"
            emoji = "✅"
        
        confidence_text = f"Confidence: {confidence:.1%}"
        time_text = f"Time: {get_current_time()}"  # ⭐ เปลี่ยนจาก pd เป็น get_current_time()
        
        # คำนวณตำแหน่งข้อความ
        img_width, img_height = img_copy.size
        
        # วาดพื้นหลังสำหรับข้อความ
        overlay_height = font_size * 5  # เพิ่มพื้นที่สำหรับเวลา
        overlay = Image.new('RGBA', (img_width, overlay_height), (0, 0, 0, 180))
        img_copy.paste(overlay, (0, 0), overlay)
        
        # วาดข้อความหลัก
        y_pos = 5
        draw.text((10, y_pos), f"{emoji} {text}", fill=color, font=font)
        
        y_pos += font_size + 5
        draw.text((10, y_pos), confidence_text, fill="#FFFFFF", font=font)
        
        y_pos += font_size + 5
        draw.text((10, y_pos), f"Status: {status_text}", fill=color, font=font)
        
        y_pos += font_size + 5
        draw.text((10, y_pos), time_text, fill="#FFFFFF", font=font)
        
        # เพิ่มกรอบ
        border_width = 5
        draw.rectangle([0, 0, img_width-1, img_height-1], 
                      outline=color, width=border_width)
        
        return img_copy
        
    except Exception as e:
        print(f"❌ Overlay error: {e}")
        return image  # ส่งรูปต้นฉบับถ้าเกิดข้อผิดพลาด

@app.route("/")
def home():
    return jsonify({
        "status": "running",
        "message": "✅ TFLite Inference API with Telegram Alert & Photo is running",
        "model_loaded": interpreter is not None,
        "telegram_configured": bool(TELEGRAM_TOKEN and TELEGRAM_CHAT_ID),
        "current_time": get_current_time()
    })

@app.route("/testbot")
def testbot():
    """ทดสอบส่งข้อความ Telegram"""
    try:
        test_msg = f"✅ Render Bot is working! 🚀\n📸 Photo sending feature enabled\n⏰ Test time: {get_current_time()}"
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
        output = interpreter.get_tensor(output_details[0]['index'])

        # ประมวลผลผลลัพธ์
        labels = ["nottarget", "cow", "goat", "sheep"]
        pred_idx = int(np.argmax(output))
        pred_label = labels[pred_idx]
        confidence = float(np.max(output))
        
        print(f"🎯 Prediction: {pred_label} (confidence: {confidence:.2f})")

        # สร้างรูปภาพที่มี overlay ผลการวิเคราะห์
        print("🎨 Adding prediction overlay...")
        result_img = add_prediction_overlay(original_img, pred_label, confidence)
        
        # แปลงรูปภาพเป็น bytes
        img_buffer = io.BytesIO()
        result_img.save(img_buffer, format='JPEG', quality=85)
        img_bytes_with_overlay = img_buffer.getvalue()

        # สร้างข้อความแจ้งเตือน
        current_time = get_current_time()  # ⭐ ใช้ get_current_time()
        
        if pred_label != "nottarget":
            caption = f"🚨 <b>Intrusion Alert!</b>\n"
            caption += f"🐄 Animal: <b>{pred_label.upper()}</b>\n"
            caption += f"📊 Confidence: <b>{confidence:.1%}</b>\n"
            caption += f"⏰ Detection Time: {current_time}"
            
            # ส่งรูปภาพไป Telegram
            photo_result = send_telegram_photo(img_bytes_with_overlay, caption)
        else:
            caption = f"✅ <b>No Animal Detected</b>\n"
            caption += f"📊 Confidence: <b>{confidence:.1%}</b>\n"
            caption += f"⏰ Scan Time: {current_time}"
            
            # ส่งรูปภาพไป Telegram (แม้ไม่เจอสัตว์)
            photo_result = send_telegram_photo(img_bytes_with_overlay, caption)

        # ส่งผลลัพธ์กลับ
        response_data = {
            "prediction": pred_label,
            "confidence": confidence,
            "photo_sent": photo_result.get("success", False),
            "timestamp": current_time
        }
        
        if not photo_result.get("success"):
            response_data["photo_error"] = photo_result.get("error")

        print(f"✅ Prediction completed: {response_data}")
        return jsonify(response_data)

    except Exception as e:
        error_msg = f"❌ Prediction error: {str(e)}"
        print(error_msg)
        
        # ส่งข้อความ error ไป Telegram
        try:
            send_telegram_message(f"❌ <b>API Error:</b>\n{str(e)}\n⏰ {get_current_time()}")
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
        "features": ["text_alerts", "photo_sending", "prediction_overlay", "timestamp"],
        "endpoints": ["/", "/predict", "/testbot", "/health"],
        "current_time": get_current_time()
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
