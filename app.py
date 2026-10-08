"""ParkiTrace Flask app.

Run:
    python app.py

Then open http://127.0.0.1:5000
"""
import base64
import gc
import io
import os
import threading
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, render_template, request, send_file
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image as RLImage, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from parkitrace_model import CLASSES, ParkiTraceNet, gradcam, overlay, preprocess

# Restrict PyTorch thread count to limit RAM allocation per request
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

app = Flask(__name__)
WEIGHTS = os.path.join("model", "parkitrace_fp16.pt")
net = None
lock = threading.Lock()
results = {}


def load_model():
    global net
    if net is None:
        if not os.path.isfile(WEIGHTS):
            raise FileNotFoundError(
                "No trained model found. Run: python train.py --data dataset --epochs 15"
            )
        checkpoint = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
        if checkpoint.get("classes") != CLASSES:
            raise RuntimeError("The saved model uses incompatible class labels.")
        
        raw_net = ParkiTraceNet(pretrained=False)
        raw_net.load_state_dict(checkpoint["state"])
        raw_net.eval()
        
        for param in raw_net.parameters():
            param.requires_grad = False

        # Dynamically quantize Linear layers to qint8 to drop memory footprint by ~65%
        net = torch.ao.quantization.quantize_dynamic(
            raw_net, {torch.nn.Linear}, dtype=torch.qint8
        )
    return net


def png_b64(rgb):
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise ValueError("Could not encode the image.")
    return base64.b64encode(buf).decode("ascii")


def risk_level(p_pd):
    if p_pd < 0.40:
        return "LOW"
    if p_pd < 0.70:
        return "MEDIUM"
    return "HIGH"


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/predict")
def predict():
    uploaded = request.files.get("image")
    if not uploaded or not uploaded.filename:
        return jsonify(error="Please choose a handwriting image."), 400

    raw = uploaded.read()
    if len(raw) > 10 * 1024 * 1024:
        return jsonify(error="Image is too large. Please use an image under 10 MB."), 413

    bgr = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        return jsonify(error="Unreadable image. Please use a PNG or JPG photo."), 400

    try:
        model = load_model()
        x, rgb = preprocess(bgr)
        
        with lock:
            with torch.inference_mode():
                out = model(x)
                probs = torch.softmax(out, dim
