"""ParkiTrace Flask app using ONNX Runtime.

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
from flask import Flask, jsonify, render_template, request, send_file
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image as RLImage, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from parkitrace_model import CLASSES, gradcam, overlay, predict_onnx, preprocess

app = Flask(__name__)
lock = threading.Lock()
results = {}


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
        x, rgb = preprocess(bgr)
        
        with lock:
            probs = predict_onnx(x)
            cls = int(np.argmax(probs))
            cam = gradcam(x, cls)

        del x, bgr
        gc.collect()

        p_pd = float(probs[CLASSES.index("parkinson")])
        patient_id = f"PT-{uuid.uuid4().hex[:8].upper()}"
        is_pd = CLASSES[cls] == "parkinson"
        result = {
            "id": patient_id,
            "prediction": "PARKINSON'S DISEASE" if is_pd else "HEALTHY",
            "confidence": round(float(probs[cls]) * 100, 1),
            "risk": risk_level(p_pd),
            "recommendation": (
                "This screening result should be reviewed by a qualified neurologist or physician."
                if p_pd >= 0.40
                else "No Parkinson's-pattern signal was detected by this model. Re-screening does not replace medical evaluation."
            ),
            "time": datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%d %b %Y, %H:%M IST"),
            "original": png_b64(rgb),
            "heatmap": png_b64(overlay(rgb, cam)),
        }
        results[patient_id] = result
        return jsonify(result)

    except FileNotFoundError as exc:
        return jsonify(error=str(exc)), 503
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify(error="Analysis failed. Check the server logs for details."), 500


@app.get("/report/<patient_id>")
def report(patient_id):
    result = results.get(patient_id)
    if not result:
        return "Report expired or patient ID was not found. Run the analysis again.", 404

    buffer = io.BytesIO()
    styles = getSampleStyleSheet()
    normal = styles["Normal"]

    def report_image(key):
        return RLImage(
            io.BytesIO(base64.b64decode(result[key])),
            width=6.3 * cm,
            height=6.3 * cm,
        )

    table = Table(
        [
            ["Patient ID", result["id"]],
            ["Prediction", result["prediction"]],
            ["Detection confidence", f'{result["confidence"]}%'],
            ["Risk level", result["risk"]],
            ["Date and time", result["time"]],
            ["Recommendation", Paragraph(result["recommendation"], normal)],
        ],
        colWidths=[5 * cm, 11 * cm],
    )
    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
    ]))

    images = Table([
        [report_image("original"), report_image("heatmap")],
        ["Submitted handwriting", "Grad-CAM heatmap"],
    ], colWidths=[8 * cm, 8 * cm])
    images.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))

    SimpleDocTemplate(buffer, pagesize=A4, title="ParkiTrace Screening Report").build([
        Paragraph("ParkiTrace Screening Report", styles["Title"]),
        Paragraph("AI-assisted handwriting screening research prototype", styles["Italic"]),
        Spacer(1, 14),
        table,
        Spacer(1, 18),
        images,
        Spacer(1, 18),
        Paragraph(
            "Important: This is a research screening prototype, not a medical diagnosis. "
            "The result must not be used alone to diagnose or rule out Parkinson's disease.",
            styles["Italic"],
        ),
    ])
    buffer.seek(0)
    return send_file(
        buffer,
        mimetype="application/pdf",
        as_attachment=True,
        download_name=f"ParkiTrace_{patient_id}.pdf",
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)