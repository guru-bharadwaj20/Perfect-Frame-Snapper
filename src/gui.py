"""
Perfect-Frame Auto-Snapper — Tkinter GUI.

Wraps the capture / detect / score pipeline in a graphical window so a
non-coder can run the tool without touching a terminal or editing constants.

Why Tkinter: it ships with Python (no extra install), its after() scheduler
naturally drives the 30-fps capture loop on the main thread without threads,
and Pillow (already pulled in by MediaPipe) handles the BGR->PhotoImage step.
The tradeoff vs PyQt is a plainer default look — acceptable for this tool.

Launch:
    python run.py            one-command launcher (keeps the venv path simple)
    python -m src.gui        direct, from the project root with venv active
"""

from __future__ import annotations

import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk

sys.path.insert(0, str(Path(__file__).parent.parent))

import src.config as _cfg
from src.capture import FrameBuffer
from src.detector import FaceDetector
from src.scoring import score_faces, eye_aspect_ratio
from src.main import ScoredFrame, _write_frame, _save_best
from src.overlay import (
    draw_score_hud, draw_face_labels, draw_readiness_bar,
    draw_mode_badge, draw_post_trigger, draw_flash, draw_footer,
    GREEN, ORANGE,
)
from src.calibration import (
    load_calibration, calibration_blink_threshold,
    compute_calibration_from_samples, save_calibration,
)
from src.feedback import log_decision, compute_adjusted_threshold

_POLL_MS      = 33          # target interval between frame updates (~30 fps)
_THUMB_WH     = (220, 165)  # max thumbnail dimensions
_CALIB_SECS   = 4.0         # open-eye collection window for inline calibration

# Tk label colours (CSS hex, not BGR)
_HEX_GREEN  = "#1ec81e"
_HEX_RED    = "#d22222"
_HEX_ORANGE = "#cc6600"
_HEX_GREY   = "#888888"


class SnapperGUI:
    """Single-window GUI that wraps the Perfect-Frame capture pipeline."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Perfect-Frame Auto-Snapper")
        self.root.resizable(False, False)

        # ── Rolling buffers ───────────────────────────────────────────────────
        self._scored_buf: deque[ScoredFrame] = deque(maxlen=_cfg.BUFFER_SIZE)
        self._score_hist: deque[float]       = deque(maxlen=_cfg.SCORE_HISTORY_N)

        # ── Runtime state ─────────────────────────────────────────────────────
        self._post_until:    float = 0.0
        self._auto_consec:   int   = 0
        self._auto_cd_until: float = 0.0
        self._flash_msg:     str   = ""
        self._flash_clr:     tuple = (255, 255, 255)
        self._flash_until:   float = 0.0
        self._frames:        int   = 0
        self._snaps:         int   = 0
        self._best_saved:    float = 0.0
        self._running:       bool  = True
        self._thumb_ref             = None   # keep ImageTk reference alive

        # ── Calibration state ─────────────────────────────────────────────────
        self._calibrating:   bool        = False
        self._calib_samples: list[float] = []
        self._calib_start:   float       = 0.0

        # ── Feedback state ────────────────────────────────────────────────────
        self._last_snap_score: float = 0.0

        # ── Load per-user calibration (falls back to config default) ──────────
        _calib = load_calibration(_cfg.CALIBRATION_FILE)
        _blink_default = (
            calibration_blink_threshold(_calib) if _calib else _cfg.BLINK_THRESHOLD
        )

        # ── Load adaptive threshold (falls back to config default) ────────────
        _auto_default = _cfg.AUTO_SNAP_THRESHOLD
        if _cfg.ENABLE_ADAPTIVE_LEARNING:
            _auto_default = compute_adjusted_threshold(
                _cfg.ADAPTIVE_LEARNING_FILE,
                _cfg.AUTO_SNAP_THRESHOLD,
                _cfg.ADAPTIVE_LEARNING_WINDOW,
                _cfg.ADAPTIVE_LEARNING_STEP,
            )

        # ── Live-control Tk variables (sliders / checkboxes drive these) ──────
        self._vblink = tk.DoubleVar(value=_blink_default)
        self._vauto  = tk.DoubleVar(value=_auto_default)
        self._vcd    = tk.DoubleVar(value=_cfg.AUTO_SNAP_COOLDOWN)
        self._vpost  = tk.DoubleVar(value=_cfg.POST_TRIGGER_SECONDS)
        self._vmouth = tk.BooleanVar(value=_cfg.ENABLE_MOUTH_PENALTY)
        self._vamode = tk.BooleanVar(value=_cfg.AUTO_SNAP_MODE)

        # ── Build UI, then open hardware ──────────────────────────────────────
        self._build_ui()
        self._init_capture()

        root.bind("<space>",  lambda _e: self._snap())
        root.bind("a",        lambda _e: self._toggle_auto())
        root.bind("q",        lambda _e: self._quit())
        root.bind("<Escape>", lambda _e: self._quit())
        root.protocol("WM_DELETE_WINDOW", self._quit)

        root.after(_POLL_MS, self._update_frame)

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        # Left column: live video feed
        vf = ttk.Frame(self.root)
        vf.pack(side="left", padx=(8, 4), pady=8)
        self._vid_lbl = ttk.Label(vf, background="black")
        self._vid_lbl.pack()

        # Right column: controls (fixed width, does not shrink to children)
        cf = ttk.Frame(self.root, width=278)
        cf.pack(side="left", padx=(4, 8), pady=8, fill="y")
        cf.pack_propagate(False)

        # ── Title + live score display ────────────────────────────────────────
        ttk.Label(cf, text="Perfect-Frame Snapper",
                  font=("Helvetica", 11, "bold")).pack(pady=(6, 0))

        self._score_lbl = ttk.Label(cf, text="SCORE: --",
                                     font=("Helvetica", 20, "bold"),
                                     foreground=_HEX_GREY)
        self._score_lbl.pack()

        self._status_lbl = ttk.Label(cf, text="",
                                      font=("Helvetica", 8),
                                      foreground=_HEX_GREY)
        self._status_lbl.pack()

        ttk.Separator(cf).pack(fill="x", pady=5)

        # ── Action buttons ────────────────────────────────────────────────────
        bf = ttk.Frame(cf)
        bf.pack(fill="x", padx=6)
        self._snap_btn = ttk.Button(bf, text="Snap  [SPACE]",
                                     command=self._snap, width=15)
        self._snap_btn.pack(side="left", padx=(0, 4))
        self._auto_btn = ttk.Button(bf, text="AUTO: OFF",
                                     command=self._toggle_auto, width=10)
        self._auto_btn.pack(side="left")

        self._post_lbl = ttk.Label(cf, text="",
                                    font=("Helvetica", 8),
                                    foreground=_HEX_ORANGE)
        self._post_lbl.pack()

        ttk.Separator(cf).pack(fill="x", pady=5)

        # ── Live-config sliders ───────────────────────────────────────────────
        self._slider(cf, "Blink threshold",     self._vblink, 0.12, 0.35, 0.01, "{:.2f}")
        self._slider(cf, "Auto-snap threshold", self._vauto,  0.15, 0.40, 0.01, "{:.2f}")
        self._slider(cf, "Auto-snap cooldown",  self._vcd,    1.0,  30.0, 0.5,  "{:.1f} s")
        self._slider(cf, "Post-trigger window", self._vpost,  0.0,  5.0,  0.5,  "{:.1f} s")

        ttk.Checkbutton(cf,
                        text="Mouth penalty  (penalise mid-speech frames)",
                        variable=self._vmouth).pack(anchor="w", padx=6, pady=2)

        # ── Calibration button (Feature 1) ────────────────────────────────────
        self._calib_btn = ttk.Button(
            cf, text="Calibrate  (set personal EAR baseline)",
            command=self._start_calibration,
        )
        self._calib_btn.pack(padx=6, pady=(4, 2), fill="x")

        ttk.Separator(cf).pack(fill="x", pady=5)

        # ── Last-saved gallery ────────────────────────────────────────────────
        ttk.Label(cf, text="Last saved",
                  font=("Helvetica", 9, "bold")).pack(anchor="w", padx=6)

        gf = ttk.Frame(cf)
        gf.pack(fill="x", padx=6, pady=2)

        self._thumb_lbl = ttk.Label(gf, background="#1a1a1a",
                                     text="(no snap yet)",
                                     foreground=_HEX_GREY,
                                     width=27, anchor="center")
        self._thumb_lbl.pack(side="left")

        nf = ttk.Frame(gf)
        nf.pack(side="left", padx=(6, 0))
        self._thumb_name  = ttk.Label(nf, text="",
                                       font=("Helvetica", 8),
                                       wraplength=100, justify="left")
        self._thumb_name.pack(anchor="w")
        self._thumb_score = ttk.Label(nf, text="",
                                       font=("Helvetica", 9, "bold"))
        self._thumb_score.pack(anchor="w")

        ttk.Button(cf, text="Open output folder",
                   command=self._open_folder).pack(padx=6, pady=(3, 0), fill="x")

        # ── Keep / Discard buttons (Feature 3) — hidden until a snap is saved ─
        self._feedback_frame = ttk.Frame(cf)
        ttk.Label(self._feedback_frame,
                  text="Rate last snap:", font=("Helvetica", 8)).pack(side="left", padx=(0, 4))
        ttk.Button(self._feedback_frame, text="✓ Keep",
                   command=lambda: self._feedback(True), width=8).pack(side="left", padx=(0, 2))
        ttk.Button(self._feedback_frame, text="✗ Discard",
                   command=lambda: self._feedback(False), width=8).pack(side="left")
        # starts hidden; shown by _on_save when ENABLE_ADAPTIVE_LEARNING=True

        ttk.Separator(cf).pack(fill="x", pady=5)

        # ── Session stats + quit ──────────────────────────────────────────────
        self._stats_lbl = ttk.Label(cf,
                                     text="Frames: 0   Snaps: 0   Best: --",
                                     font=("Helvetica", 8),
                                     foreground=_HEX_GREY)
        self._stats_lbl.pack(anchor="w", padx=6)

        ttk.Button(cf, text="Quit  [Q / ESC]",
                   command=self._quit).pack(padx=6, pady=(4, 6), fill="x")

    @staticmethod
    def _slider(parent, label, var, lo, hi, res, fmt) -> None:
        """Add a labelled horizontal Scale with a live value display."""
        f = ttk.Frame(parent)
        f.pack(fill="x", padx=6, pady=1)
        hdr = ttk.Frame(f)
        hdr.pack(fill="x")
        ttk.Label(hdr, text=label, font=("Helvetica", 8)).pack(side="left")
        val = ttk.Label(hdr, text=fmt.format(var.get()),
                        font=("Helvetica", 8, "bold"), width=8, anchor="e")
        val.pack(side="right")
        def _cb(v, _val=val, _fmt=fmt):
            _val.configure(text=_fmt.format(float(v)))
        tk.Scale(f, variable=var, from_=lo, to=hi, resolution=res,
                 orient="horizontal", showvalue=False, length=258,
                 command=_cb).pack(fill="x")

    # ── Hardware init ─────────────────────────────────────────────────────────

    def _init_capture(self) -> None:
        self._buf = FrameBuffer(camera_index=_cfg.CAMERA_INDEX, maxlen=_cfg.BUFFER_SIZE)
        self._det = FaceDetector(
            max_num_faces=_cfg.MAX_NUM_FACES,
            min_face_detection_confidence=_cfg.MIN_FACE_DETECTION_CONFIDENCE,
            min_face_presence_confidence=_cfg.MIN_FACE_PRESENCE_CONFIDENCE,
            min_tracking_confidence=_cfg.MIN_TRACKING_CONFIDENCE,
        )
        self._buf.open()

    # ── Frame update loop (driven by Tkinter's after() scheduler) ─────────────

    def _update_frame(self) -> None:
        if not self._running:
            return

        ok, frame = self._buf.read_frame()
        if not ok:
            self.root.after(_POLL_MS, self._update_frame)
            return

        now = time.monotonic()

        # ── Calibration mode: delegate to dedicated tick ──────────────────────
        if self._calibrating:
            self._run_calibration_tick(frame, now)
            self.root.after(_POLL_MS, self._update_frame)
            return

        # Read live control values once per frame
        blink_thr = self._vblink.get()
        auto_thr  = self._vauto.get()
        cooldown  = self._vcd.get()
        post_trig = self._vpost.get()
        mouth_pen = self._vmouth.get()
        auto_mode = self._vamode.get()

        # Detect + score (passes live slider values and composite/weighting flags)
        faces  = self._det.detect(frame)
        result = score_faces(
            faces,
            blink_threshold=blink_thr,
            enable_mouth_penalty=mouth_pen,
            frame=frame,
            use_composite=_cfg.USE_COMPOSITE_SCORING,
            use_face_weighting=_cfg.USE_FACE_WEIGHTING,
        )

        self._scored_buf.append(ScoredFrame(score=result.score, frame=frame.copy()))
        self._score_hist.append(result.score)
        self._frames += 1

        # ── Post-trigger deadline ─────────────────────────────────────────────
        if 0 < self._post_until <= now:
            self._post_until = 0.0
            ok2, msg, saved = _save_best(self._scored_buf, _cfg.OUTPUT_DIR)
            self._on_save(ok2, msg, saved)

        # ── Auto-snap stable-frame gate ───────────────────────────────────────
        elif auto_mode and self._post_until == 0:
            if result.score >= auto_thr:
                self._auto_consec += 1
            else:
                self._auto_consec = 0
            if (self._auto_consec >= _cfg.AUTO_SNAP_STABLE_FRAMES
                    and now >= self._auto_cd_until):
                recent  = list(self._scored_buf)[-_cfg.AUTO_SNAP_STABLE_FRAMES:]
                best_sf = max(recent, key=lambda sf: sf.score)
                ok2, msg, saved = _write_frame(best_sf, _cfg.OUTPUT_DIR)
                self._on_save(ok2, f"Auto: {msg}", saved)
                self._auto_consec   = 0
                self._auto_cd_until = now + cooldown

        # ── Build OpenCV overlay ──────────────────────────────────────────────
        vis = frame.copy()
        draw_face_labels(vis, faces, result, blink_thr)
        draw_readiness_bar(vis, self._score_hist, blink_thr)
        draw_score_hud(vis, result.score, len(faces),
                       len(self._scored_buf), _cfg.BUFFER_SIZE, blink_thr)
        draw_mode_badge(vis, auto_mode, self._auto_consec, _cfg.AUTO_SNAP_STABLE_FRAMES)
        if self._post_until > now:
            draw_post_trigger(vis, self._post_until - now)
        draw_footer(vis, auto_mode, self._auto_cd_until, post_trig, now)
        if now < self._flash_until:
            draw_flash(vis, self._flash_msg, self._flash_clr)

        # ── Push frame to Tk label ────────────────────────────────────────────
        rgb   = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(image=Image.fromarray(rgb))
        self._vid_lbl.configure(image=photo)
        self._vid_lbl.image = photo   # prevent garbage collection

        # ── Update right-panel widgets ────────────────────────────────────────
        hex_c = _HEX_GREEN if result.score >= blink_thr else _HEX_RED
        self._score_lbl.configure(text=f"SCORE: {result.score:.3f}", foreground=hex_c)

        cd_rem = max(0.0, self._auto_cd_until - now)
        cd_txt = f"   cd {cd_rem:.1f}s" if cd_rem > 0 and auto_mode else ""
        self._status_lbl.configure(
            text=f"faces: {len(faces)}   buf: {len(self._scored_buf)}/{_cfg.BUFFER_SIZE}{cd_txt}")

        if self._post_until > now:
            self._post_lbl.configure(
                text=f"Post-trigger: {self._post_until - now:.1f}s ...")
        else:
            self._post_lbl.configure(text="")

        best_s = f"{self._best_saved:.3f}" if self._snaps else "--"
        self._stats_lbl.configure(
            text=f"Frames: {self._frames:,}   Snaps: {self._snaps}   Best: {best_s}")

        self.root.after(_POLL_MS, self._update_frame)

    # ── Actions ───────────────────────────────────────────────────────────────

    def _snap(self) -> None:
        """Fire the manual shutter (ignored in auto mode or mid-post-trigger)."""
        if self._vamode.get() or self._post_until > 0:
            return
        now = time.monotonic()
        pt  = self._vpost.get()
        if pt > 0:
            self._post_until = now + pt
        else:
            ok2, msg, saved = _save_best(self._scored_buf, _cfg.OUTPUT_DIR)
            self._on_save(ok2, msg, saved)

    def _toggle_auto(self) -> None:
        """Switch between manual and auto-snap modes."""
        new_val = not self._vamode.get()
        self._vamode.set(new_val)
        self._auto_consec = 0
        if new_val:
            self._auto_btn.configure(text="AUTO: ON ")
            self._snap_btn.configure(state="disabled")
        else:
            self._auto_btn.configure(text="AUTO: OFF")
            self._snap_btn.configure(state="normal")

    def _on_save(self, success: bool, msg: str, score: float) -> None:
        """Handle the result of any save operation."""
        self._flash_msg   = msg
        self._flash_clr   = GREEN if success else ORANGE
        self._flash_until = time.monotonic() + _cfg.FLASH_SECONDS
        if success:
            self._snaps           += 1
            self._best_saved       = max(self._best_saved, score)
            self._last_snap_score  = score
            print(msg)
            self._update_thumbnail(score)
            # Show Keep/Discard buttons when adaptive learning is enabled
            if _cfg.ENABLE_ADAPTIVE_LEARNING:
                self._feedback_frame.pack(padx=6, pady=(2, 0), fill="x")

    def _update_thumbnail(self, score: float) -> None:
        """Load the most recently saved JPEG and show it in the gallery."""
        _cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        jpegs = sorted(_cfg.OUTPUT_DIR.glob("snap_*.jpg"),
                       key=lambda p: p.stat().st_mtime)
        if not jpegs:
            return
        try:
            img = Image.open(jpegs[-1])
            img.thumbnail(_THUMB_WH)
            self._thumb_ref = ImageTk.PhotoImage(image=img)   # keep reference
            self._thumb_lbl.configure(image=self._thumb_ref, text="")
            self._thumb_name.configure(text=jpegs[-1].name)
            hex_s = _HEX_GREEN if score >= self._vblink.get() else _HEX_RED
            self._thumb_score.configure(text=f"Score: {score:.3f}", foreground=hex_s)
        except Exception:
            pass

    # ── Calibration helpers (Feature 1) ───────────────────────────────────────

    def _start_calibration(self) -> None:
        """Begin the inline 4-second open-eye calibration capture."""
        self._calibrating    = True
        self._calib_samples  = []
        self._calib_start    = time.monotonic()
        self._calib_btn.configure(state="disabled", text="Calibrating…")

    def _run_calibration_tick(self, frame: np.ndarray, now: float) -> None:
        """Called from _update_frame() during calibration instead of normal flow."""
        elapsed   = now - self._calib_start
        remaining = _CALIB_SECS - elapsed

        # Collect EAR samples
        faces = self._det.detect(frame)
        if remaining > 0 and faces:
            for face in faces:
                r   = eye_aspect_ratio(face.right_eye_pts)
                l   = eye_aspect_ratio(face.left_eye_pts)
                self._calib_samples.append((r + l) * 0.5)

        # Draw calibration overlay
        vis   = frame.copy()
        h, w  = vis.shape[:2]
        if remaining > 0:
            msg = f"Calibrating — keep eyes open  ({remaining:.1f}s)"
            clr = (200, 200, 0)
        else:
            msg = "Done!  Saving…"
            clr = (0, 220, 0)
        cv2.putText(vis, msg, (10, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, clr, 2, cv2.LINE_AA)
        if not faces:
            cv2.putText(vis, "No face detected — look at camera",
                        (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (0, 100, 255), 1, cv2.LINE_AA)
        else:
            avg = self._calib_samples[-1] if self._calib_samples else 0.0
            cv2.putText(vis, f"EAR: {avg:.3f}  samples: {len(self._calib_samples)}",
                        (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (255, 255, 255), 1, cv2.LINE_AA)

        rgb   = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(image=Image.fromarray(rgb))
        self._vid_lbl.configure(image=photo)
        self._vid_lbl.image = photo

        if elapsed >= _CALIB_SECS:
            self._finish_calibration()

    def _finish_calibration(self) -> None:
        """Compute calibration result, save it, and update the blink slider."""
        data = compute_calibration_from_samples(self._calib_samples)
        save_calibration(_cfg.CALIBRATION_FILE, data)
        new_thr = float(data["blink_threshold"])
        self._vblink.set(new_thr)
        self._calibrating = False
        self._calib_btn.configure(state="normal",
                                   text="Calibrate  (set personal EAR baseline)")
        self._flash_msg   = (f"Calibrated!  baseline={data['baseline_ear']:.3f}  "
                              f"threshold={new_thr:.3f}")
        self._flash_clr   = GREEN
        self._flash_until = time.monotonic() + _cfg.FLASH_SECONDS * 2

    # ── Feedback helpers (Feature 3) ──────────────────────────────────────────

    def _feedback(self, kept: bool) -> None:
        """Log a keep/discard decision and adjust the auto-snap threshold."""
        log_decision(_cfg.ADAPTIVE_LEARNING_FILE, self._last_snap_score, kept)
        self._feedback_frame.pack_forget()
        new_thr = compute_adjusted_threshold(
            _cfg.ADAPTIVE_LEARNING_FILE,
            _cfg.AUTO_SNAP_THRESHOLD,
            _cfg.ADAPTIVE_LEARNING_WINDOW,
            _cfg.ADAPTIVE_LEARNING_STEP,
        )
        if abs(new_thr - self._vauto.get()) > 1e-4:
            self._vauto.set(new_thr)
            print(f"[Adaptive] Auto-snap threshold adjusted → {new_thr:.3f}")

    def _open_folder(self) -> None:
        """Open the output directory in the system file manager."""
        _cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        folder = str(_cfg.OUTPUT_DIR.resolve())
        if sys.platform == "win32":
            subprocess.run(["explorer", folder], check=False)
        elif sys.platform == "darwin":
            subprocess.run(["open", folder], check=False)
        else:
            subprocess.run(["xdg-open", folder], check=False)

    def _quit(self) -> None:
        """Release hardware and destroy the window."""
        self._running = False
        try:
            self._buf.release()
        except Exception:
            pass
        try:
            self._det.close()
        except Exception:
            pass
        print("\nSession summary")
        print(f"  Frames processed : {self._frames:,}")
        print(f"  Snaps saved      : {self._snaps}")
        print(f"  Best score saved : {self._best_saved:.3f}")
        print(f"  Output dir       : {_cfg.OUTPUT_DIR.resolve()}")
        self.root.destroy()


def main() -> None:
    """Create the Tk root, attach the app, and run the event loop."""
    root = tk.Tk()
    SnapperGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
