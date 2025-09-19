from flask import Flask, request, jsonify
import numpy as np
from PIL import Image   # 👈 แทนที่ cv2
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
TELEGRAM_TOKEN = os.getenv("8435446530:AAFVEb_kZsF1Xr1HuU5Zl9aurVbMNfU_etU")
TELEGRAM_CHAT_ID = os.getenv("6024710139")
bot = Bot(token=TELEGRAM_TOKEN)

app = Flask(__name__)

@app.route("/")
def home():
    return "✅ TFLite Inference API with Telegram Alert is running"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # ESP32 ส่งภาพเป็น Base64
        img_base64 = request.json.get("image", None)
        if img_base64 is None:
            return jsonify({"error": "No image data"}), 400

        # decode base64 → Image (Pillow)
        img_bytes = base64.b64decode(img_base64)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # resize ตาม input model
        target_shape = input_details[0]['shape'][1:3]  # (height, width)
        img_resized = img.resize((target_shape[1], target_shape[0]))
        img_resized = np.array(img_resized)
        img_normalized = img_resized.astype(np.float32) / 255.0
        img_input = np.expand_dims(img_normalized, axis=0)

        # run inference
        interpreter.set_tensor(input_details[0]['index'], img_input)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])

        # สมมติ label คือ [nottarget, cow, goat, sheep]
        labels = ["nottarget", "cow", "goat", "sheep"]
        pred_idx = int(np.argmax(output))
        pred_label = labels[pred_idx]
        confidence = float(np.max(output))

        # ส่งแจ้งเตือนถ้าไม่ใช่ nottarget
        if pred_label != "nottarget":
            message = f"🚨 Intrusion Detected: {pred_label} (confidence {confidence:.2f})"
            bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)

        return jsonify({
            "prediction": pred_label,
            "confidence": confidence
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
