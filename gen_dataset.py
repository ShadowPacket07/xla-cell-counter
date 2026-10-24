"""Sinh bộ dữ liệu thử nghiệm 12 ảnh (có đáp án) -> dataset/ + ground_truth.csv.
Nên bổ sung thêm ảnh thật (vd. BBBC005/BBBC006 Broad Institute, ảnh hạt gạo/đậu/hạt cườm chụp bằng điện thoại)."""
import os, csv
import cv2, numpy as np

os.makedirs("dataset", exist_ok=True)
rng = np.random.RandomState(7)
rows = []
for i in range(1, 13):
    W, H = 640, 480
    style = "fluo" if i % 3 else "stain"      # fluorescence (sáng/nền tối) hoặc nhuộm (tối/nền sáng)
    n_target = int(rng.randint(25, 95))
    rmin, rmax = (8, 14) if i > 6 else (12, 22)
    pts = []
    tries = 0
    while len(pts) < n_target and tries < 20000:
        tries += 1
        r = rng.randint(rmin, rmax + 1)
        x, y = rng.randint(r + 2, W - r - 2), rng.randint(r + 2, H - r - 2)
        if all(np.hypot(x - a, y - b) >= 0.85 * (r + c) for a, b, c in pts):  # cho phép chạm/dính nhẹ
            pts.append((x, y, r))
    layer = np.zeros((H, W), np.float32)
    for x, y, r in pts:
        cv2.ellipse(layer, (x, y), (r, max(4, int(r * rng.uniform(0.75, 1))))
                    , rng.randint(0, 180), 0, 360, float(rng.uniform(0.7, 1.0)), -1)
    layer = cv2.GaussianBlur(layer, (0, 0), 2)
    bg = cv2.GaussianBlur(rng.rand(H, W).astype(np.float32), (0, 0), 40) * 0.25   # nền không đều
    noise = rng.normal(0, 0.03, (H, W)).astype(np.float32)
    if style == "fluo":
        g = np.clip(layer * 0.9 + bg * 0.5 + noise, 0, 1)
        img = cv2.merge([g * 0.35, g * 1.0, g * 0.45])
    else:
        g = np.clip(0.92 - layer * 0.55 - bg * 0.3 + noise, 0, 1)
        img = cv2.merge([g * 0.85, g * 0.7, g * 1.0])
    name = f"cell_{i:02d}_{style}.png"
    cv2.imwrite(os.path.join("dataset", name), (img * 255).astype(np.uint8))
    rows.append((name, len(pts)))
with open("dataset/ground_truth.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["file", "count"]); w.writerows(rows)
print("Đã tạo", len(rows), "ảnh")
