# main.py
import os
import time
import threading
import cv2
from PIL import Image
import customtkinter as ctk
from tkinter import filedialog, messagebox
from detector import CrowdDetector
from utils import LOG_FILE, ABNORMAL_DIR, HEATMAP_DIR

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

APP_TITLE = "ZaBir  •  Crowd Behavior Analytics"


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1320x780")
        self.minsize(1100, 680)

        self.detector = None
        self.video_path = None
        self.cap = None
        self.running = False
        self.stop_flag = threading.Event()
        self.worker = None

        self._build_sidebar()
        self._build_main_area()

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---------- UI: Sidebar ----------
    def _build_sidebar(self):
        self.sidebar = ctk.CTkFrame(self, width=260, corner_radius=0)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        title = ctk.CTkLabel(
            self.sidebar, text="🛡️  Crowd Sentinel",
            font=ctk.CTkFont(size=22, weight="bold"),
        )
        title.pack(pady=(24, 4), padx=20, anchor="w")

        sub = ctk.CTkLabel(
            self.sidebar, text="Real-time Abnormal Activity Detection",
            font=ctk.CTkFont(size=12), text_color="gray70",
        )
        sub.pack(padx=20, anchor="w", pady=(0, 24))

        # --- Model status ---
        self.model_status = ctk.CTkLabel(
            self.sidebar, text="● Model not loaded",
            font=ctk.CTkFont(size=13), text_color="#ff6b6b",
        )
        self.model_status.pack(padx=20, pady=(0, 20), anchor="w")

        # --- Buttons ---
        self._button("📂  Load YOLO Model", self.load_model).pack(padx=20, pady=6, fill="x")
        self._button("🎬  Select Video", self.select_video).pack(padx=20, pady=6, fill="x")
        self.btn_start = self._button("▶  Start Detection", self.start, accent=True)
        self.btn_start.pack(padx=20, pady=(20, 6), fill="x")
        self.btn_stop = self._button("⏹  Stop", self.stop, danger=True)
        self.btn_stop.pack(padx=20, pady=6, fill="x")
        self.btn_stop.configure(state="disabled")

        # --- Status card ---
        stat_card = ctk.CTkFrame(self.sidebar, fg_color="#1e2330", corner_radius=12)
        stat_card.pack(padx=20, pady=(28, 12), fill="x")

        ctk.CTkLabel(stat_card, text="LIVE STATS",
                     font=ctk.CTkFont(size=12, weight="bold"),
                     text_color="gray60").pack(anchor="w", padx=14, pady=(10, 4))

        self.lbl_humans = self._stat_row(stat_card, "Humans", "0")
        self.lbl_density = self._stat_row(stat_card, "Density", "0.00")
        self.lbl_status = self._stat_row(stat_card, "Status", "IDLE")
        self.lbl_fps = self._stat_row(stat_card, "FPS", "0")

        # --- Quick open ---
        ctk.CTkLabel(self.sidebar, text="QUICK ACCESS",
                     font=ctk.CTkFont(size=12, weight="bold"),
                     text_color="gray60").pack(padx=20, pady=(20, 4), anchor="w")
        self._button("🗂️  Open Abnormal Frames", lambda: self._open_folder(ABNORMAL_DIR)).pack(padx=20, pady=4, fill="x")
        self._button("🔥  Open Heatmaps", lambda: self._open_folder(HEATMAP_DIR)).pack(padx=20, pady=4, fill="x")
        self._button("📜  View Logs", self.view_logs).pack(padx=20, pady=4, fill="x")

    def _button(self, text, cmd, accent=False, danger=False):
        color = "#2563eb" if accent else ("#dc2626" if danger else "#2b3244")
        hover = "#1d4ed8" if accent else ("#b91c1c" if danger else "#3a4258")
        return ctk.CTkButton(
            self.sidebar, text=text, command=cmd, height=40,
            corner_radius=10, fg_color=color, hover_color=hover,
            font=ctk.CTkFont(size=13, weight="bold"),
        )

    def _stat_row(self, parent, label, initial):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=14, pady=3)
        ctk.CTkLabel(row, text=label, font=ctk.CTkFont(size=12),
                     text_color="gray70").pack(side="left")
        val = ctk.CTkLabel(row, text=initial,
                           font=ctk.CTkFont(size=13, weight="bold"))
        val.pack(side="right")
        return val

    # ---------- UI: Main area ----------
    def _build_main_area(self):
        main = ctk.CTkFrame(self, corner_radius=0, fg_color="#0f1218")
        main.pack(side="right", fill="both", expand=True)

        header = ctk.CTkFrame(main, fg_color="transparent")
        header.pack(fill="x", padx=24, pady=(18, 6))
        ctk.CTkLabel(header, text="Live Feed",
                     font=ctk.CTkFont(size=20, weight="bold")).pack(side="left")
        self.video_name = ctk.CTkLabel(header, text="No video loaded",
                                       text_color="gray60",
                                       font=ctk.CTkFont(size=12))
        self.video_name.pack(side="right")

        self.video_label = ctk.CTkLabel(
            main, text="📺\n\nLoad a YOLO model and select a video to begin",
            font=ctk.CTkFont(size=16), text_color="gray50",
            fg_color="#161a24", corner_radius=12,
        )
        self.video_label.pack(padx=24, pady=(6, 20), fill="both", expand=True)

        # Alert banner
        self.banner = ctk.CTkLabel(
            main, text="", height=34, corner_radius=10,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#161a24", text_color="gray60",
        )
        self.banner.pack(padx=24, pady=(0, 18), fill="x")

    # ---------- Actions ----------
    def load_model(self):
        path = filedialog.askopenfilename(
            title="Select YOLO weights",
            filetypes=[("PyTorch weights", "*.pt"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            self.model_status.configure(text="⏳ Loading model...", text_color="#facc15")
            self.update_idletasks()
            self.detector = CrowdDetector(model_path=path)
            self.model_status.configure(
                text=f"● Model: {os.path.basename(path)}",
                text_color="#22c55e",
            )
        except Exception as e:
            self.model_status.configure(text="● Load failed", text_color="#ff6b6b")
            messagebox.showerror("Model error", str(e))

    def select_video(self):
        path = filedialog.askopenfilename(
            title="Select video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")],
        )
        if not path:
            return
        self.video_path = path
        self.video_name.configure(text=os.path.basename(path))
        self.video_label.configure(text="🎬  Video loaded — press Start")

    def start(self):
        if self.detector is None:
            messagebox.showwarning("No model", "Please load a YOLO model first.")
            return
        if not self.video_path or not os.path.exists(self.video_path):
            messagebox.showwarning("No video", "Please select a video first.")
            return
        if self.running:
            return

        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            messagebox.showerror("Video error", "Could not open the video file.")
            return

        self.running = True
        self.stop_flag.clear()
        self.btn_start.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.banner.configure(text="🟢  Detection running...", fg_color="#132a1e", text_color="#22c55e")

        self.worker = threading.Thread(target=self._run_loop, daemon=True)
        self.worker.start()

    def stop(self):
        self.stop_flag.set()
        self.running = False

    def _run_loop(self):
        fps_t0 = time.time()
        frames = 0
        fps = 0.0

        while not self.stop_flag.is_set():
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (1020, 560))
            annotated, stats = self.detector.process_frame(frame)

            # Update UI
            self._update_stats(stats, fps)

            # Show frame
            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(1020, 560))
            self.video_label.configure(image=ctk_img, text="")
            self.video_label._image = ctk_img  # keep ref

            frames += 1
            if time.time() - fps_t0 >= 1.0:
                fps = frames / (time.time() - fps_t0)
                frames = 0
                fps_t0 = time.time()

        if self.cap:
            self.cap.release()
        if self.detector:
            self.detector.finalize()

        self.running = False
        self.btn_start.configure(state="normal")
        self.btn_stop.configure(state="disabled")
        self.banner.configure(text="⏹  Detection stopped", fg_color="#161a24", text_color="gray60")

    def _update_stats(self, stats, fps):
        self.lbl_humans.configure(text=str(stats["humans"]))
        self.lbl_density.configure(text=f"{stats['density']:.2f}")
        self.lbl_fps.configure(text=f"{fps:.1f}")

        if stats["abnormal"]:
            self.lbl_status.configure(text="ABNORMAL", text_color="#ff6b6b")
            uniq = sorted(set(stats["alerts"]))
            self.banner.configure(
                text=f"⚠  ABNORMAL: {', '.join(uniq)}",
                fg_color="#3a1111", text_color="#ff8b8b",
            )
        else:
            self.lbl_status.configure(text="NORMAL", text_color="#22c55e")
            self.banner.configure(text="✔  All normal", fg_color="#132a1e", text_color="#22c55e")

    # ---------- Helpers ----------
    def _open_folder(self, path):
        if os.path.isdir(path):
            try:
                os.startfile(path)   # Windows
            except AttributeError:
                import subprocess, sys
                subprocess.Popen(["xdg-open" if sys.platform != "darwin" else "open", path])
        else:
            messagebox.showinfo("Empty", "Folder is empty / not created yet.")

    def view_logs(self):
        win = ctk.CTkToplevel(self)
        win.title("Abnormal Activity Logs")
        win.geometry("720x500")
        txt = ctk.CTkTextbox(win, font=ctk.CTkFont(family="Consolas", size=12))
        txt.pack(fill="both", expand=True, padx=12, pady=12)
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                txt.insert("1.0", f.read())
        else:
            txt.insert("1.0", "No logs yet.")

    def _on_close(self):
        self.stop_flag.set()
        self.running = False
        if self.cap:
            self.cap.release()
        self.destroy()


if __name__ == "__main__":
    app = App()
    app.mainloop()