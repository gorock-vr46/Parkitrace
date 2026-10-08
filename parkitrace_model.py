"""ParkiTrace model preprocessing, ONNX inference, and heatmapping utilities."""
import os
import cv2
import numpy as np
import onnxruntime as ort

CLASSES = ["healthy", "parkinson"]
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ONNX_PATH = os.path.join(BASE_DIR, "model", "parkitrace.onnx")
_session = None


def load_onnx_session():
    global _session
    if _session is None:
        if not os.path.exists(ONNX_PATH):
            raise FileNotFoundError(f"ONNX model not found at {ONNX_PATH}")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        _session = ort.InferenceSession(ONNX_PATH, sess_options=opts, providers=["CPUExecutionProvider"])
    return _session


def preprocess(bgr):
    if bgr is None or bgr.size == 0:
        raise ValueError("The image is empty.")
    bgr = cv2.medianBlur(bgr, 3)
    rgb = cv2.cvtColor(
        cv2.resize(bgr, (224, 224), interpolation=cv2.INTER_AREA),
        cv2.COLOR_BGR2RGB,
    )
    x = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
    tensor = np.expand_dims(x.transpose(2, 0, 1), axis=0).astype(np.float32)
    return tensor, rgb


def predict_onnx(tensor):
    session = load_onnx_session()
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: tensor})
    logits = outputs[0][0]
    # Softmax
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / exp_logits.sum()
    return probs


def gradcam(x, cls):
    """Low-memory contour heatmap generation."""
    img = x[0].transpose(1, 2, 0)
    img = ((img * STD + MEAN) * 255).clip(0, 255).astype(np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    blur = cv2.GaussianBlur(gray, (15, 15), 0)
    heatmap = cv2.Laplacian(blur, cv2.CV_64F)
    heatmap = np.abs(heatmap)
    
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
        
    cam = cv2.GaussianBlur(heatmap.astype(np.float32), (21, 21), 0)
    if cam.max() > 0:
        cam = cam / cam.max()
        
    return cam


def overlay(rgb, cam):
    heat = cv2.cvtColor(
        cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET),
        cv2.COLOR_BGR2RGB,
    )
    return cv2.addWeighted(rgb, 0.55, heat, 0.45, 0)