from flask import Flask
import os
import requests

app = Flask(__name__)

# โหลด Environment Variables
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

print("🚀 TELEGRAM_TOKEN:", TELEGRAM_TOKEN)
print("🚀 TELEGRAM_CHAT_ID:", TELEGRAM_CHAT_ID)

def send_telegram_message(text):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return {"error": "Missing TELEGRAM_TOKEN or TELEGRAM_CHAT_ID"}

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": text}

    try:
        r = requests.post(url, data=data)
        print("📡 Telegram Response:", r.text)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

@app.route("/")
def home():
    return "✅ Telegram Test API is running"

@app.route("/testsend")
def testsend():
    return send_telegram_message("📢 Hello! This is a test message from Render 🚀")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
