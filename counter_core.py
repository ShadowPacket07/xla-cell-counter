"""Lõi xử lý: đếm tế bào/hạt bằng Distance Transform + Watershed."""
import csv, os
import cv2
import numpy as np

DEFAULTS = dict(blur=5, method="Otsu", thresh=127, block=35, C=5,
                polarity="Tự động", kernel=3, open_iter=2, bg_iter=3,
                dist_ratio=0.45, min_area=40, max_area=50000)


def _odd(v, lo=1):
    v = max(int(v), lo)
    return v if v % 2 == 1 else v + 1


def _binarize(blur, p):
    pol = p["polarity"]
    # Xác định cực: "Sáng" = đối tượng sáng trên nền tối; "Tối" = ngược lại
    if pol == "Tự động":
        t, b = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        inv = (b > 0).mean() > 0.5   # nếu "đối tượng" chiếm > 50% -> thực ra là nền
    else:
        inv = (pol == "Tối")
    mode = cv2.THRESH_BINARY_INV if inv else cv2.THRESH_BINARY
    if p["method"] == "Otsu":
        _, b = cv2.threshold(blur, 0, 255, mode + cv2.THRESH_OTSU)
    elif p["method"] == "Adaptive":
        b = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, mode,
                                  _odd(p["block"], 3), int(p["C"]))
    else:  # Global
        _, b = cv2.threshold(blur, int(p["thresh"]), 255, mode)
    return b


def segment(bgr, params=None):
    p = dict(DEFAULTS)
    if params:
        p.update(params)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    k = _odd(p["blur"])
    blur = cv2.GaussianBlur(gray, (k, k), 0) if k > 1 else gray
    binary = _binarize(blur, p)

    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(p["kernel"], 1),) * 2)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, ker, iterations=int(p["open_iter"]))
    sure_bg = cv2.dilate(opening, ker, iterations=int(p["bg_iter"]))

    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    dmax = dist.max() if dist.max() > 0 else 1.0
    sure_fg = (dist > p["dist_ratio"] * dmax).astype(np.uint8) * 255
    unknown = cv2.subtract(sure_bg, sure_fg)

    n_fg, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    ws = cv2.watershed(bgr.copy(), markers.astype(np.int32))

    # Thống kê từng vùng (nhãn >= 2), lọc theo diện tích
    h, w = ws.shape
    flat = ws.ravel()
    valid = flat >= 2
    cnt = np.bincount(flat[valid], minlength=n_fg + 2)
    yy, xx = np.indices((h, w))
    sx = np.bincount(flat[valid], weights=xx.ravel()[valid], minlength=n_fg + 2)
    sy = np.bincount(flat[valid], weights=yy.ravel()[valid], minlength=n_fg + 2)
    stats, keep = [], []
    for lab in range(2, n_fg + 1):
        a = int(cnt[lab])
        if p["min_area"] <= a <= p["max_area"]:
            keep.append(lab)
            stats.append(dict(id=len(keep), area=a, cx=round(sx[lab] / a, 1),
                              cy=round(sy[lab] / a, 1),
                              diameter=round(2 * np.sqrt(a / np.pi), 1)))

    result = bgr.copy()
    keep_mask = np.isin(ws, keep)
    result[(ws == -1) & keep_mask.__or__(cv2.dilate(keep_mask.astype(np.uint8), None).astype(bool))] = (0, 0, 255)
    for s in stats:
        cv2.putText(result, str(s["id"]), (int(s["cx"]) - 6, int(s["cy"]) + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(result, f"Count: {len(stats)}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

    dist_vis = cv2.applyColorMap(cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
                                 cv2.COLORMAP_JET)
    mk = np.zeros((h, w, 3), np.uint8)
    rng = np.random.RandomState(1)
    colors = rng.randint(60, 255, (n_fg + 2, 3))
    m = ws.copy()
    mk[m >= 2] = colors[m[m >= 2] % (n_fg + 2)]
    mk[ws == -1] = (255, 255, 255)
    fg_vis = cv2.cvtColor(sure_fg, cv2.COLOR_GRAY2BGR)

    return dict(count=len(stats), stats=stats, result=result,
                views={"Kết quả đếm": result,
                       "Ảnh xám (làm mờ)": cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR),
                       "Nhị phân": cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR),
                       "Sau Morphology (Open)": cv2.cvtColor(opening, cv2.COLOR_GRAY2BGR),
                       "Distance Transform": dist_vis,
                       "Sure foreground (markers)": fg_vis,
                       "Vùng Watershed": mk})


def save_stats_csv(stats, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=["id", "area", "cx", "cy", "diameter"])
        wr.writeheader()
        wr.writerows(stats)


def evaluate_folder(folder, gt_csv, params=None):
    """gt_csv: file,count. Trả về danh sách kết quả + MAE, MAPE, độ chính xác trung bình."""
    gt = {}
    with open(gt_csv, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            gt[r["file"]] = int(r["count"])
    rows = []
    for name, true in gt.items():
        img = cv2.imread(os.path.join(folder, name))
        if img is None:
            continue
        auto = segment(img, params)["count"]
        err = auto - true
        rows.append(dict(file=name, truth=true, auto=auto, error=err,
                         abs_pct=round(abs(err) / max(true, 1) * 100, 2)))
    if not rows:
        return rows, {}
    mae = float(np.mean([abs(r["error"]) for r in rows]))
    mape = float(np.mean([r["abs_pct"] for r in rows]))
    return rows, dict(MAE=round(mae, 2), MAPE=round(mape, 2), Accuracy=round(100 - mape, 2))
