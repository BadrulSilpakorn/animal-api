from flask import Flask, jsonify
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
        return {"error": "Missing TELEGRAM_TOKEN or TELEGRAM_CHAT_ID", "success": False}

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {
        "chat_id": TELEGRAM_CHAT_ID, 
        "text": text,
        "parse_mode": "HTML"  # เพิ่มการรองรับ HTML formatting
    }

    try:
        response = requests.post(url, data=data, timeout=10)
        print("📡 Telegram Response:", response.text)
        
        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        else:
            return {"success": False, "error": f"HTTP {response.status_code}", "details": response.text}
            
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": f"Request failed: {str(e)}"}
    except Exception as e:
        return {"success": False, "error": f"Unexpected error: {str(e)}"}

@app.route("/")
def home():
    return jsonify({
        "status": "running",
        "message": "✅ Telegram Test API is running",
        "endpoints": {
            "/": "Home page",
            "/testsend": "Send test message to Telegram",
            "/health": "Health check"
        }
    })

@app.route("/testsend")
def testsend():
    result = send_telegram_message("📢 Hello! This is a test message from Render 🚀")
    return jsonify(result)

@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "telegram_configured": bool(TELEGRAM_TOKEN and TELEGRAM_CHAT_ID)
    })

# สำหรับ Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
