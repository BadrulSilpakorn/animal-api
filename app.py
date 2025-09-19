{\rtf1\ansi\ansicpg1252\cocoartf2513
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;\f1\fnil\fcharset222 Thonburi;\f2\fnil\fcharset0 AppleColorEmoji;
}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww10800\viewh8400\viewkind0
\pard\tx566\tx1133\tx1700\tx2267\tx2834\tx3401\tx3968\tx4535\tx5102\tx5669\tx6236\tx6803\pardirnatural\partightenfactor0

\f0\fs24 \cf0 from flask import Flask, request, jsonify\
import numpy as np, cv2, tensorflow as tf\
\
app = Flask(__name__)\
\
# 
\f1 \'e2\'cb\'c5\'b4\'e2\'c1\'e0\'b4\'c5
\f0  TFLite\
interpreter = tf.lite.Interpreter(model_path="animal_model.tflite")\
interpreter.allocate_tensors()\
input_details = interpreter.get_input_details()\
output_details = interpreter.get_output_details()\
labels = ["background", "cow", "goat", "sheep"]  # 
\f1 \'bb\'c3\'d1\'ba\'b5\'d2\'c1\'e2\'c1\'e0\'b4\'c5\'a2\'cd\'a7\'a4\'d8\'b3
\f0 \
\
@app.route("/predict", methods=["POST"])\
def predict():\
    file = request.files['image']\
    img_bytes = file.read()\
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)\
\
    # resize 
\f1 \'e3\'cb\'e9\'b5\'c3\'a7
\f0  input model\
    img = cv2.resize(img, (96, 96)).astype(np.float32) / 255.0\
    img = np.expand_dims(img, axis=0)\
\
    interpreter.set_tensor(input_details[0]['index'], img)\
    interpreter.invoke()\
    output = interpreter.get_tensor(output_details[0]['index'])[0]\
\
    pred_idx = int(np.argmax(output))\
    confidence = float(output[pred_idx])\
    return jsonify(\{"animal": labels[pred_idx], "confidence": confidence\})\
\
@app.route("/")\
def home():\
    return "
\f2 \uc0\u9989 
\f0  Animal API is running!"\
}