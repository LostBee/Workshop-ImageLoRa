# prep.py

import cv2
import numpy as np
import pathlib
import csv
from insightface.app import FaceAnalysis

# 1) Corrected paths to match your layout
RAW_DIR  = pathlib.Path(r"H:\Python\LoRA_project\raw_photos")
OUT_DIR  = pathlib.Path(r"H:\Python\LoRA_project\training_data\20_uniquename_person")
OUT_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST = pathlib.Path(r"H:\Python\LoRA_project\manifest.csv")

# 2) Init the face detector
face_app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider"])
face_app.prepare(ctx_id=0, det_size=(640, 640))

def strip_blurred_bars(img, thresh_ratio=0.25, win=5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    col_var = np.var(lap, axis=0)
    med = np.median(col_var)
    mask = col_var > (med * thresh_ratio)
    mask = np.convolve(mask.astype(np.uint8), np.ones(win, dtype=np.uint8), mode='same') > 0
    cols = np.where(mask)[0]
    if len(cols) < win:
        return img
    return img[:, cols[0]:cols[-1]+1]

def square_pad(img, size=1024):
    h, w = img.shape[:2]
    scale = size / max(h, w)
    img = cv2.resize(img, (int(w*scale), int(h*scale)), cv2.INTER_LANCZOS4)
    h, w = img.shape[:2]
    top = (size - h) // 2
    left = (size - w) // 2
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    canvas[top:top+h, left:left+w] = img
    return canvas

# 3) Process every image, regardless of original size
with MANIFEST.open("w", newline="") as f_csv:
    writer = csv.writer(f_csv)
    writer.writerow(["filename", "height", "width"])
    
    for file in sorted(RAW_DIR.iterdir()):
        if file.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue
        
        img = cv2.imread(str(file))
        if img is None:
            print(f"[!] Cannot read {file.name}")
            continue

        # 3a) Strip blurred side bars
        img = strip_blurred_bars(img)

        # 3b) Face-centric crop if a face is detected
        faces = face_app.get(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if faces:
            x1, y1, x2, y2 = faces[0].bbox.astype(int)
            cx, cy = (x1 + x2)//2, (y1 + y2)//2
            half = max(x2-x1, y2-y1) // 2
            x1, x2 = max(0, cx-half), cx+half
            y1, y2 = max(0, cy-half), cy+half
            img = img[y1:y2, x1:x2]
        
        # 3c) Pad & resize everything to exactly 1024×1024
        img = square_pad(img, 1024)

        # 3d) Save PNG and log to manifest
        out_path = OUT_DIR / f"{file.stem}.png"
        cv2.imwrite(str(out_path), img)
        writer.writerow([out_path.name, 1024, 1024])

print("✅ Prep complete. Processed images are in:", OUT_DIR)
