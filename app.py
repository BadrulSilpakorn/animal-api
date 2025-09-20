from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import io
import base64
import tflite_runtime.interpreter as tflite
from telegram import Bot
import os

# โหลดโมเดล TFLite
interpreter = tflite.Interpreter(model_path="animal_model_int8.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# โหลด Telegram Bot
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
bot = Bot(token=TELEGRAM_TOKEN)

app = Flask(__name__)

@app.route("/")
def home():
    return "✅ TFLite Inference API with Telegram Alert is running"

# ✅ Route สำหรับทดสอบ Telegram โดยตรง
@app.route("/testbot")
def testbot():
    try:
        test_msg = "✅ Render Bot is working!"
        print("📢 Sending Telegram (test):", test_msg)
        bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=test_msg)
        return "Message sent to Telegram!"
    except Exception as e:
        return str(e)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # ESP32 ส่งภาพเป็น Base64
        img_base64 = request.json.get("image", None)
        if img_base64 is None:
            return jsonify({"error": "No image data"}), 400

        # decode base64 → Image
        img_bytes = base64.b64decode(img_base64)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # resize ตาม input model
        target_shape = input_details[0]['shape'][1:3]  # (height, width)
        img_resized = img.resize((target_shape[1], target_shape[0]))
        img_resized = np.array(img_resized)

        # ใช้ uint8 ตรงกับโมเดล INT8
        img_input = np.expand_dims(img_resized.astype(np.uint8), axis=0)

        # run inference
        interpreter.set_tensor(input_details[0]['index'], img_input)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])

        # labels
        labels = ["nottarget", "cow", "goat", "sheep"]
        pred_idx = int(np.argmax(output))
        pred_label = labels[pred_idx]
        confidence = float(np.max(output))

        # ✅ ส่งแจ้งเตือน Telegram พร้อม log
        if pred_label != "nottarget":
            message = f"🚨 Intrusion Detected: {pred_label} (confidence {confidence:.2f})"
        else:
            message = f"✅ No animal detected (confidence {confidence:.2f})"

        print("📢 Sending Telegram (predict):", message)
        bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)

        return jsonify({
            "prediction": pred_label,
            "confidence": confidence
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
