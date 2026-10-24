"""Công cụ đếm tay: click chuột trái vào từng hạt, chuột phải (hoặc phím Z) để hoàn tác.
Bấm 'Lưu & ảnh tiếp' (hoặc Enter) -> tự ghi vào ground_truth.csv. Chạy: python count_helper.py"""
import os, csv
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2, numpy as np
from PIL import Image, ImageTk

EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
MAXW, MAXH = 1100, 640


class Helper(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Đếm tay để làm đáp án (ground truth)")
        folder = filedialog.askdirectory(title="Chọn thư mục dataset", initialdir=os.path.abspath("dataset"))
        if not folder:
            self.destroy(); return
        self.folder = folder
        self.csv_path = os.path.join(folder, "ground_truth.csv")
        done = set()
        if os.path.exists(self.csv_path):
            with open(self.csv_path, encoding="utf-8") as f:
                done = {r["file"] for r in csv.DictReader(f)}
        else:
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["file", "count"])
        self.queue = [n for n in sorted(os.listdir(folder)) if n.lower().endswith(EXT) and n not in done]
        if not self.queue:
            messagebox.showinfo("Xong", "Mọi ảnh trong thư mục đã có đáp án."); self.destroy(); return
        self.pts = []
        top = tk.Frame(self); top.pack(fill="x")
        self.info = tk.Label(top, font=("Segoe UI", 13, "bold")); self.info.pack(side="left", padx=8)
        tk.Button(top, text="Bỏ qua ảnh", command=self.skip).pack(side="right", padx=4)
        tk.Button(top, text="Lưu & ảnh tiếp (Enter)", command=self.save_next).pack(side="right", padx=4)
        tk.Button(top, text="Hoàn tác (Z)", command=self.undo).pack(side="right", padx=4)
        self.cv = tk.Canvas(self, width=MAXW, height=MAXH, bg="gray20"); self.cv.pack()
        self.cv.bind("<Button-1>", self.add); self.cv.bind("<Button-3>", lambda e: self.undo())
        self.bind("z", lambda e: self.undo()); self.bind("<Return>", lambda e: self.save_next())
        self.load()

    def load(self):
        if not self.queue:
            messagebox.showinfo("Xong", "Đã đếm hết các ảnh. File: " + self.csv_path); self.destroy(); return
        self.name = self.queue[0]; self.pts = []
        img = cv2.imdecode(np.fromfile(os.path.join(self.folder, self.name), np.uint8), cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        self.s = min(MAXW / w, MAXH / h)
        im = cv2.resize(img, (int(w * self.s), int(h * self.s)), interpolation=cv2.INTER_AREA)
        self.tkimg = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)))
        self.cv.config(width=im.shape[1], height=im.shape[0])
        self.redraw()

    def redraw(self):
        self.cv.delete("all")
        self.cv.create_image(0, 0, anchor="nw", image=self.tkimg)
        for i, (x, y) in enumerate(self.pts, 1):
            self.cv.create_oval(x - 4, y - 4, x + 4, y + 4, fill="red", outline="yellow")
        self.info.config(text=f"{self.name}  |  Đã đếm: {len(self.pts)}  |  Còn lại: {len(self.queue) - 1} ảnh sau ảnh này")

    def add(self, e):
        self.pts.append((e.x, e.y)); self.redraw()

    def undo(self):
        if self.pts:
            self.pts.pop(); self.redraw()

    def save_next(self):
        if not self.pts and not messagebox.askyesno("Xác nhận", "Chưa click hạt nào, lưu số 0?"):
            return
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([self.name, len(self.pts)])
        self.queue.pop(0); self.load()

    def skip(self):
        self.queue.pop(0); self.load()


if __name__ == "__main__":
    h = Helper()
    if h.winfo_exists():
        h.mainloop()
