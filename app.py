"""GUI đếm tế bào/hạt (Distance Transform + Watershed). Chạy: python app.py"""
import os
import csv
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import counter_core as core

PANEL_W, PANEL_H = 560, 420


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Đếm tế bào/hạt - Distance Transform + Watershed")
        self.geometry("1280x700")
        self.src = None          # ảnh gốc BGR
        self.out = None          # kết quả segment
        self.path = None
        self.cap = None
        self.playing = False
        self._job = None
        self._deb = None
        self.vars = {}
        self._build()

    # ---------- UI ----------
    def _build(self):
        left = ttk.Frame(self, padding=8)
        left.pack(side="left", fill="y")
        right = ttk.Frame(self, padding=8)
        right.pack(side="left", fill="both", expand=True)

        bf = ttk.Frame(left)
        bf.pack(fill="x")
        bf.columnconfigure((0, 1), weight=1)
        btns = [("Mở ảnh", self.open_image), ("Mở video", self.open_video),
                ("Dừng/tiếp tục video", self.toggle_play), ("Lưu kết quả", self.save),
                ("Đánh giá thư mục", self.batch_eval), ("Đặt lại tham số", self.reset)]
        for i, (t, c) in enumerate(btns):
            ttk.Button(bf, text=t, command=c).grid(
                row=i // 2, column=i % 2, sticky="ew", padx=1, pady=1)

        box = ttk.LabelFrame(left, text="Tham số", padding=6)
        box.pack(fill="x", pady=6)
        d = core.DEFAULTS
        self._combo(box, "polarity", "Đối tượng", [
                    "Tự động", "Sáng", "Tối"], d["polarity"])
        self._combo(box, "method", "Ngưỡng hóa", [
                    "Otsu", "Adaptive", "Global"], d["method"])
        self._scale(box, "blur", "Làm mờ Gaussian (kernel)", 1, 21, d["blur"])
        self._scale(box, "thresh", "Ngưỡng Global", 0, 255, d["thresh"])
        self._scale(box, "block", "Block size (Adaptive)", 3, 101, d["block"])
        self._scale(box, "C", "Hằng số C (Adaptive)", -20, 30, d["C"])
        self._scale(box, "kernel", "Kernel Morphology", 1, 15, d["kernel"])
        self._scale(box, "open_iter", "Số lần Open", 0, 8, d["open_iter"])
        self._scale(box, "bg_iter", "Số lần Dilate (sure bg)",
                    1, 10, d["bg_iter"])
        self._scale(box, "dist_ratio", "Ngưỡng Distance (× max)",
                    0.05, 0.95, d["dist_ratio"], res=0.01)
        self._scale(box, "min_area", "Diện tích tối thiểu",
                    0, 2000, d["min_area"])
        self._scale(box, "max_area", "Diện tích tối đa",
                    100, 100000, d["max_area"])

        gt = ttk.LabelFrame(left, text="Đánh giá kết quả", padding=6)
        gt.pack(fill="x", pady=6)
        row = ttk.Frame(gt)
        row.pack(fill="x")
        ttk.Label(row, text="Đếm tay:").pack(side="left")
        self.gt_var = tk.StringVar()
        e = ttk.Entry(row, textvariable=self.gt_var, width=8)
        e.pack(side="left", padx=4)
        e.bind("<KeyRelease>", lambda ev: self.update_eval())
        self.eval_lbl = ttk.Label(gt, text="", wraplength=260, justify="left")
        self.eval_lbl.pack(fill="x", pady=2)

        top = ttk.Frame(right)
        top.pack(fill="x")
        ttk.Label(top, text="Xem ảnh 'Sau':").pack(side="left")
        self.view_var = tk.StringVar(value="Kết quả đếm")
        self.view_cb = ttk.Combobox(top, textvariable=self.view_var, state="readonly", width=28,
                                    values=["Kết quả đếm", "Ảnh xám (làm mờ)", "Nhị phân", "Sau Morphology (Open)",
                                            "Distance Transform", "Sure foreground (markers)", "Vùng Watershed"])
        self.view_cb.pack(side="left", padx=6)
        self.view_cb.bind("<<ComboboxSelected>>", lambda e: self.refresh())
        self.count_lbl = ttk.Label(
            top, text="Số lượng: -", font=("Segoe UI", 16, "bold"))
        self.count_lbl.pack(side="right")

        imgs = ttk.Frame(right)
        imgs.pack(fill="both", expand=True, pady=6)
        for title, attr in (("Trước (ảnh gốc)", "lbl_before"), ("Sau", "lbl_after")):
            f = ttk.LabelFrame(imgs, text=title)
            f.pack(side="left", fill="both", expand=True, padx=3)
            lb = ttk.Label(f)
            lb.pack(fill="both", expand=True)
            setattr(self, attr, lb)

        cols = ("id", "area", "cx", "cy", "diameter")
        self.tree = ttk.Treeview(
            right, columns=cols, show="headings", height=6)
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=90, anchor="center")
        self.tree.pack(fill="x")
        self.stat_lbl = ttk.Label(right, text="")
        self.stat_lbl.pack(fill="x")

    def _scale(self, parent, key, text, lo, hi, val, res=1):
        row = ttk.Frame(parent)
        row.pack(fill="x")
        row.columnconfigure(1, weight=1)
        ttk.Label(row, text=text, width=22).grid(row=0, column=0, sticky="w")
        v = tk.DoubleVar(value=val)
        self.vars[key] = v
        ttk.Scale(row, from_=lo, to=hi, variable=v,
                  command=lambda _=None: self._debounce()).grid(row=0, column=1, sticky="ew")
        lbl = ttk.Label(row, width=5, anchor="e")
        lbl.grid(row=0, column=2)
        fmt = (lambda x: f"{x:.2f}") if res < 1 else (
            lambda x: str(int(round(x))))
        v.trace_add("write", lambda *a: lbl.config(text=fmt(v.get())))
        lbl.config(text=fmt(val))

    def _combo(self, parent, key, text, values, val):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=1)
        row.columnconfigure(1, weight=1)
        ttk.Label(row, text=text, width=22).grid(row=0, column=0, sticky="w")
        v = tk.StringVar(value=val)
        self.vars[key] = v
        cb = ttk.Combobox(row, textvariable=v, values=values,
                          state="readonly", width=12)
        cb.grid(row=0, column=1, sticky="ew")
        cb.bind("<<ComboboxSelected>>", lambda e: self.process())

    def params(self):
        p = {}
        for k, v in self.vars.items():
            val = v.get()
            if isinstance(val, float):
                val = round(val, 2) if k == "dist_ratio" else int(round(val))
            p[k] = val
        return p

    def reset(self):
        for k, v in self.vars.items():
            v.set(core.DEFAULTS[k])
        self.process()

    # ---------- nguồn ảnh ----------
    def open_image(self):
        self._stop_video()
        path = filedialog.askopenfilename(
            filetypes=[("Ảnh", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")])
        if not path:
            return
        img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            messagebox.showerror("Lỗi", "Không đọc được ảnh")
            return
        self.path, self.src = path, img
        self.process()

    def open_video(self):
        self._stop_video()
        path = filedialog.askopenfilename(
            filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv")])
        if not path:
            return
        self.cap = cv2.VideoCapture(path)
        self.path = path
        self.playing = True
        self._next_frame()

    def _next_frame(self):
        if not (self.cap and self.playing):
            return
        ok, frame = self.cap.read()
        if not ok:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
            if not ok:
                return
        if frame.shape[1] > 640:
            s = 640 / frame.shape[1]
            frame = cv2.resize(frame, None, fx=s, fy=s)
        self.src = frame
        self.process()
        self._job = self.after(30, self._next_frame)

    def toggle_play(self):
        if self.cap:
            self.playing = not self.playing
            if self.playing:
                self._next_frame()

    def _stop_video(self):
        self.playing = False
        if self._job:
            self.after_cancel(self._job)
            self._job = None
        if self.cap:
            self.cap.release()
            self.cap = None

    # ---------- xử lý & hiển thị ----------
    def _debounce(self):
        if self._deb is not None:
            self.after_cancel(self._deb)
        self._deb = self.after(150, self._process_debounced)

    def _process_debounced(self):
        self._deb = None
        self.process()

    def process(self):
        if self.src is None:
            return
        self.out = core.segment(self.src, self.params())
        self.refresh()

    def _show(self, label, bgr):
        h, w = bgr.shape[:2]
        s = min(PANEL_W / w, PANEL_H / h)
        im = cv2.resize(bgr, (int(w * s), int(h * s)),
                        interpolation=cv2.INTER_AREA)
        tk_img = ImageTk.PhotoImage(Image.fromarray(
            cv2.cvtColor(im, cv2.COLOR_BGR2RGB)))
        label.configure(image=tk_img)
        label.image = tk_img

    def refresh(self):
        if self.src is None or self.out is None:
            return
        self._show(self.lbl_before, self.src)
        self._show(self.lbl_after, self.out["views"][self.view_var.get()])
        self.count_lbl.config(text=f"Số lượng: {self.out['count']}")
        self.tree.delete(*self.tree.get_children())
        for s in self.out["stats"][:300]:
            self.tree.insert("", "end", values=(
                s["id"], s["area"], s["cx"], s["cy"], s["diameter"]))
        a = [s["area"] for s in self.out["stats"]]
        if a:
            self.stat_lbl.config(
                text=f"Diện tích: TB={np.mean(a):.1f} | Min={min(a)} | Max={max(a)} | Độ lệch chuẩn={np.std(a):.1f}")
        self.update_eval()

    def update_eval(self):
        if self.out is None:
            return
        try:
            t = int(self.gt_var.get())
        except ValueError:
            self.eval_lbl.config(text="")
            return
        auto = self.out["count"]
        err = auto - t
        self.eval_lbl.config(text=f"Tự động={auto} | Đếm tay={t} | Sai số={err:+d} "
                             f"({abs(err) / max(t, 1) * 100:.1f}%) | Chính xác={100 - abs(err) / max(t, 1) * 100:.1f}%")

    # ---------- đánh giá lô & lưu ----------
    def batch_eval(self):
        gt = filedialog.askopenfilename(
            title="Chọn ground_truth.csv (file,count)", filetypes=[("CSV", "*.csv")])
        if not gt:
            return
        rows, s = core.evaluate_folder(os.path.dirname(gt), gt, self.params())
        if not rows:
            messagebox.showwarning(
                "Thông báo", "Không có ảnh hợp lệ cạnh file csv")
            return
        out = os.path.join(os.path.dirname(gt), "evaluation_result.csv")
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        messagebox.showinfo("Kết quả đánh giá",
                            f"{len(rows)} ảnh\nMAE = {s['MAE']}\nMAPE = {s['MAPE']}%\n"
                            f"Độ chính xác TB = {s['Accuracy']}%\n\nĐã lưu: {out}")

    def save(self):
        if self.out is None:
            messagebox.showwarning("Thông báo", "Chưa có kết quả")
            return
        base = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")],
                                            initialfile="ket_qua.png")
        if not base:
            return
        stem = os.path.splitext(base)[0]
        cv2.imencode(".png", self.out["result"])[1].tofile(base)
        both = np.hstack([self.src, self.out["result"]])
        cv2.imencode(".png", both)[1].tofile(stem + "_truoc_sau.png")
        core.save_stats_csv(self.out["stats"], stem + "_thongke.csv")
        messagebox.showinfo(
            "Đã lưu", f"{base}\n{stem}_truoc_sau.png\n{stem}_thongke.csv")


if __name__ == "__main__":
    App().mainloop()
