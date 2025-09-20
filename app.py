from flask import Flask
from telegram import Bot
import os

# โหลด Telegram Bot
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
bot = Bot(token=TELEGRAM_TOKEN)

app = Flask(__name__)

@app.route("/")
def home():
    return "✅ Telegram Test API is running"

@app.route("/testsend")
def testsend():
    try:
        message = "📢 Hello! This is a test message from app.py 🚀"
        print("📢 Sending Telegram:", message)
        bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
        return "Message sent to Telegram!"
    except Exception as e:
        return str(e), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
