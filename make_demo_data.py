"""Generates SYNTHETIC spiral drawings so the pipeline can be tested end-to-end without a real dataset.
Healthy = smooth, large spiral. 'Parkinson' = small, tremor-distorted spiral. NOT clinical data."""
import cv2, numpy as np, os
rng = np.random.default_rng(0)
def spiral(pd):
    t = np.linspace(0, 6 * np.pi, 900)
    r = (3 + 3.2 * t) * (0.55 if pd else 1.0)
    if pd: r = r + rng.uniform(2, 5) * np.sin(t * rng.uniform(9, 14)) + rng.normal(0, 1.4, t.size)
    else:  r = r + rng.normal(0, 0.35, t.size)
    ph = rng.uniform(0, 6.28)
    pts = np.stack([112 + r * np.cos(t + ph), 112 + r * np.sin(t + ph)], 1).astype(np.int32)
    img = np.full((224, 224, 3), 255, np.uint8)
    cv2.polylines(img, [pts], False, (20, 20, 20), int(rng.integers(1, 3)), cv2.LINE_AA)
    return img
for cls in ("healthy", "parkinson"):
    # Synthetic data only: never use these images as clinical evidence.
    os.makedirs(f"dataset/{cls}", exist_ok=True)
    for i in range(120): cv2.imwrite(f"dataset/{cls}/{cls}_{i:03d}.png", spiral(cls == "parkinson"))
print("Synthetic demo dataset written to ./dataset")
