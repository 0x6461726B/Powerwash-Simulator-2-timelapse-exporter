#!/usr/bin/env python3
"""
PW2 Timelapse Exporter — GUI (guided flow)
MP4 defaults to 24 fps, GIF defaults to 12 fps. No FPS control in UI.

Deps:
  pip install zstandard pillow imageio numpy
  # for MP4:
  pip install imageio-ffmpeg
"""

import io, os, sys, json, time, struct, threading, queue, subprocess
from pathlib import Path
from typing import List, Tuple, Optional, Callable
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import zstandard as zstd
from PIL import Image
import imageio.v2 as imageio
import numpy as np

# ---------- Core constants ----------
STRIDE, HEIGHT, ORIGIN = 1280, 720, "bottom"
ZSTD_MAGIC = b"\x28\xB5\x2F\xFD"
APP_NAME = "PW2 Timelapse Exporter"
SETTINGS_PATH = Path.home() / ".pw2_timelapse_exporter.json"

# ---------- Codec helpers ----------
def u32le(b: bytes, o: int) -> int:
    import struct as _struct
    return _struct.unpack_from("<I", b, o)[0]

def is_jpeg_lenient(bs: bytes) -> bool:
    return bs.startswith(b"\xFF\xD8") and (bs.rfind(b"\xFF\xD9") >= 0)

def normalize_jpeg(bs: bytes) -> bytes:
    if not bs.startswith(b"\xFF\xD8"): return bs
    eoi = bs.rfind(b"\xFF\xD9")
    return bs if eoi < 0 else bs[:eoi + 2]

def find_zstd_frames(data: bytes) -> List[Tuple[int, int]]:
    offs, i = [], 0
    while True:
        j = data.find(ZSTD_MAGIC, i)
        if j < 0: break
        offs.append(j); i = j + 1
    return [(offs[k], offs[k + 1] if k + 1 < len(offs) else len(data))
            for k in range(len(offs))]

def enc_tiles_per_row(stride: int, tile: int) -> int:
    return max(1, (stride + tile - 1) // tile)

def decode_pos(pos: int, stride: int, tile: int):
    tpr = enc_tiles_per_row(stride, tile)
    tile_index = (pos >> 12) & 0xFFFFF
    tx, ty = tile_index % tpr, tile_index // tpr
    x_px, y_px = ((pos & 0xFFF) >> 6) * tile, ty * tile
    return x_px, y_px

def get_tile_rect(pixel_offset: int, tile_size: int, tex_w: int, tex_h: int):
    x = pixel_offset % tex_w
    y = pixel_offset // tex_w
    if x >= tex_w or y >= tex_h: return None
    w = min(tile_size, tex_w - x)
    h = min(tile_size, tex_h - y)
    return (x, y, w, h) if w > 0 and h > 0 else None

def looks_like_keyframe_payload(b: bytes) -> bool:
    off, n, seen = 0, len(b), 0
    while off + 4 <= n and seen < 3:
        ln = u32le(b, off); off += 4
        if ln <= 2 or off + ln > n: return seen > 0
        jb = b[off:off+ln]; off += ln
        if not is_jpeg_lenient(jb): return seen > 0
        seen += 1
    return seen > 0

def read_keyframe_tiles(b: bytes) -> List[bytes]:
    tiles, off, n = [], 0, len(b)
    while off + 4 <= n:
        ln = u32le(b, off); off += 4
        if ln <= 0 or off + ln > n: break
        jb = b[off:off+ln]; off += ln
        if not is_jpeg_lenient(jb): break
        tiles.append(normalize_jpeg(jb))
    return tiles

def parse_delta(b: bytes) -> List[Tuple[int, int, bytes]]:
    recs: List[Tuple[int, int, bytes]] = []
    n, ti = len(b), 0

    def try_parse(start: int):
        out = []; off = start
        while off + 8 <= n:
            a = u32le(b, off); c = u32le(b, off+4)
            chosen: Optional[Tuple[int, int, bytes]] = None
            if 0 < c <= (n - (off + 8)):
                jb = b[off+8:off+8+c]
                if is_jpeg_lenient(jb): chosen = (a, c, jb)
            if chosen is None and 0 < a <= (n - (off + 8)):
                jb = b[off+8:off+8+a]
                if is_jpeg_lenient(jb): chosen = (c, a, jb)
            if chosen is None:
                if not out: return []
                break
            pos, ln, jb2 = chosen
            out.append((pos, ln, normalize_jpeg(jb2)))
            off += 8 + ln
        return out

    for start in range(0, 64, 4):
        out = try_parse(start)
        if out:
            for pos, _ln, jb in out:
                recs.append((ti, pos, jb)); ti += 1
            break
    return recs

def build_keyframe_engine_exact(tiles: List[bytes], stride: int, height: int, origin: str = "bottom"):
    first = Image.open(io.BytesIO(tiles[0])).convert("RGB")
    tw, th = first.size; W, H = stride, height
    canvas = Image.new("RGB", (W, H))
    tpr = enc_tiles_per_row(stride, tw)
    for i, jb in enumerate(tiles):
        tx, ty = i % tpr, i // tpr
        x_top, y_top = tx * tw, ty * th
        w, h = min(tw, stride - x_top), min(th, H - y_top)
        if w <= 0 or h <= 0: continue
        im = Image.open(io.BytesIO(jb)).convert("RGB")
        y_paste = (H - y_top - h) if origin == "bottom" else y_top
        canvas.paste(im.crop((0, 0, w, h)), (x_top, y_paste))
    return canvas, (tw, th)

# ---------- Writers ----------
def write_gif(frames: List[Image.Image], out_path: Path, delay_ms: int) -> None:
    # Defaults: ADAPTIVE palette + Floyd–Steinberg, infinite loop
    pal_img = frames[0].convert("P", palette=Image.Palette.ADAPTIVE, colors=256)
    pframes = [im.quantize(palette=pal_img, dither=Image.Dither.FLOYDSTEINBERG) for im in frames]
    pframes[0].save(
        str(out_path),
        save_all=True,
        append_images=pframes[1:],
        duration=delay_ms,
        loop=0, optimize=False, disposal=2,
    )

def _writer_add_frame(writer, frame_np: np.ndarray) -> None:
    if hasattr(writer, "append_data"): writer.append_data(frame_np)
    elif hasattr(writer, "write_frame"): writer.write_frame(frame_np)
    else: raise RuntimeError("Unsupported imageio writer")

def write_mp4(frames: List[Image.Image], out_path: Path, fps: float) -> None:
    try:
        with imageio.get_writer(str(out_path), fps=fps, codec="libx264", quality=9) as w:
            for im in frames: _writer_add_frame(w, np.asarray(im))
    except Exception:
        imageio.mimsave(str(out_path), [np.asarray(im) for im in frames], fps=fps, codec="libx264", quality=9)

# ---------- Top-level convert ----------
def convert_timelapse(
    sav_file: Path,
    out_dir: Path,
    *,
    fmt: str = "gif",
    progress: Optional[Callable[[int, int], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> Path:
    fps = 24.0 if fmt.lower() == "mp4" else 12.0
    delay_ms = max(1, int(round(1000.0 / fps)))

    data = sav_file.read_bytes()
    auto_i = data.find(ZSTD_MAGIC)
    if auto_i < 0: raise RuntimeError("Zstd magic not found; not a PW2 timelapse?")
    body = data[auto_i:]

    frame_ranges = find_zstd_frames(body)
    if not frame_ranges: raise RuntimeError("No Zstd frames found.")

    dctx = zstd.ZstdDecompressor()
    chunks = [dctx.decompress(body[s:e]) for (s, e) in frame_ranges]
    if not chunks: raise RuntimeError("No decompressed chunks.")

    total_chunks = len(chunks)
    frames: List[Image.Image] = []

    ktiles = read_keyframe_tiles(chunks[0])
    if not ktiles: raise RuntimeError("First chunk isn't a valid keyframe.")
    canvas, (tw, th) = build_keyframe_engine_exact(ktiles, STRIDE, HEIGHT, ORIGIN)
    frames.append(canvas.copy())
    if progress: progress(1, total_chunks)

    processed = 1
    for chunk in chunks[1:]:
        if cancel_event and cancel_event.is_set(): raise RuntimeError("Cancelled by user.")
        if looks_like_keyframe_payload(chunk):
            ktiles2 = read_keyframe_tiles(chunk)
            if ktiles2: canvas, (tw, th) = build_keyframe_engine_exact(ktiles2, STRIDE, HEIGHT, ORIGIN)
            frames.append(canvas.copy())
            processed += 1
            if progress: progress(processed, total_chunks)
            continue

        for _, pos, jb in parse_delta(chunk):
            x_px, y_px = decode_pos(pos, STRIDE, tw)
            rect = get_tile_rect(y_px * STRIDE + x_px, th, STRIDE, HEIGHT)
            if rect is None: continue
            x, y, w, h = rect
            y_paste = (HEIGHT - y - h) if ORIGIN == "bottom" else y
            try:
                im = Image.open(io.BytesIO(jb)).convert("RGB")
            except Exception:
                continue
            canvas.paste(im.crop((0, 0, w, h)), (x, y_paste))
        frames.append(canvas.copy())
        processed += 1
        if progress: progress(processed, total_chunks)

    out_dir.mkdir(parents=True, exist_ok=True)
    if fmt.lower() == "mp4":
        out_path = out_dir / "timelapse.mp4"
        write_mp4(frames, out_path, fps)
    else:
        out_path = out_dir / "timelapse.gif"
        write_gif(frames, out_path, delay_ms)
    return out_path

# ---------- Settings ----------
def load_settings():
    try: return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except Exception: return {}

def save_settings(d: dict):
    try: SETTINGS_PATH.write_text(json.dumps(d, indent=2), encoding="utf-8")
    except Exception: pass

# ---------- Paths / discovery ----------
def default_savedata_dir() -> Optional[Path]:
    # Steam on Windows
    if os.name == "nt":
        localapp = os.environ.get("LOCALAPPDATA")
        if localapp:
            p = Path(localapp).parent / "LocalLow" / "FuturLab" / "PowerWash Simulator 2" / "PC_Steam" / "SaveData"
            if p.exists(): return p
    return None

def list_levels(savedata_dir: Path) -> List[Tuple[str, Path]]:
    out: List[Tuple[str, Path]] = []
    if not savedata_dir or not savedata_dir.exists(): return out
    for child in sorted(savedata_dir.iterdir()):
        if child.is_dir() and (child / "timelapse.sav").exists():
            out.append((child.name, child))
    return out

def default_video_root() -> Path:
    # Nice default: ~/Videos/PowerWash Timelapses or ~/PW2 Timelapses
    v = Path.home() / "Videos" / "PowerWash Timelapses"
    if (Path.home() / "Videos").exists(): return v
    return Path.home() / "PW2 Timelapses"

def sizeof_fmt(n: float) -> str:
    for unit in ["B","KB","MB","GB","TB"]:
        if n < 1024.0: return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{float(n):.1f} PB"

def shorten_path(p: str, maxlen: int = 55) -> str:
    if len(p) <= maxlen: return p
    head, tail = p[:maxlen//2-2], p[-maxlen//2:]
    return f"{head}…{tail}"

# Simple tooltip
class Tooltip:
    def __init__(self, widget, text=""):
        self.widget = widget
        self.text = text
        self.tip = None
        widget.bind("<Enter>", self.show)
        widget.bind("<Leave>", self.hide)
    def show(self, _e=None):
        if self.tip or not self.text: return
        x, y, cx, cy = self.widget.bbox("insert") if hasattr(self.widget, "bbox") else (0,0,0,0)
        x = x + self.widget.winfo_rootx() + 25
        y = y + self.widget.winfo_rooty() + 20
        self.tip = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(tw, text=self.text, justify="left",
                         background="#ffffe0", relief="solid", borderwidth=1,
                         font=("Segoe UI", 9))
        label.pack(ipadx=6, ipady=4)
    def hide(self, _e=None):
        if self.tip: self.tip.destroy(); self.tip = None
    def set(self, text: str):
        self.text = text

# ---------- GUI ----------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_NAME)
        self.geometry("760x380")
        self.minsize(720, 360)

        st = load_settings()
        self.var_savedata_real = tk.StringVar(value=st.get("savedata_dir", "") or (str(default_savedata_dir() or "")))
        self.var_savedata_display = tk.StringVar()
        self.var_level = tk.StringVar(value="")
        self.var_out_real = tk.StringVar(value=st.get("out_dir", ""))
        self.var_out_display = tk.StringVar()
        self.var_fmt = tk.StringVar(value=st.get("fmt", "mp4"))  # default to mp4 in UI
        self.sav_status = tk.StringVar(value="Select a SaveData folder.")
        self.status_var = tk.StringVar(value="Ready")
        self.exporting = False
        self.can_open_output = False
        self._locked_states = {}

        self.cancel_event = threading.Event()
        self.msgq: "queue.Queue[Tuple[str, tuple]]" = queue.Queue()
        self.start_time: Optional[float] = None
        self.levels: List[Tuple[str, Path]] = []

        self._build_menu()
        self._build_ui()
        self._refresh_levels(auto=True)
        self._sync_path_labels()
        self.after(80, self._drain_queue)
        self._update_controls_state()

    # ----- Menu -----
    def _build_menu(self):
        m = tk.Menu(self)
        filem = tk.Menu(m, tearoff=0)
        filem.add_command(label="Detect SaveData (Steam)", command=lambda: self._refresh_levels(from_detect=True))
        filem.add_command(label="Browse SaveData…", command=self.browse_savedata)
        filem.add_separator()
        filem.add_command(label="Open Selected Level Folder", command=self.open_level_folder)
        filem.add_separator()
        filem.add_command(label="Exit", command=self.destroy, accelerator="Ctrl+Q")
        self.bind_all("<Control-q>", lambda e: self.destroy())
        m.add_cascade(label="File", menu=filem)
        helpm = tk.Menu(m, tearoff=0)
        helpm.add_command(label="About", command=lambda: messagebox.showinfo(
            "About", f"{APP_NAME}\nGuided export of PW2 timelapses.\nMP4=24fps, GIF=12fps."))
        m.add_cascade(label="Help", menu=helpm)
        self.config(menu=m)

    # ----- UI -----
    def _build_ui(self):
        content = ttk.Frame(self, padding=(12, 12, 12, 12))
        content.pack(fill="both", expand=True)

        # Source group
        grp_src = ttk.Labelframe(content, text="1) Source")
        grp_src.pack(fill="x", expand=False)

        row = 0
        ttk.Label(grp_src, text="SaveData folder:").grid(row=row, column=0, sticky="w")
        self.ent_savedata = ttk.Entry(grp_src, textvariable=self.var_savedata_display, state="readonly")
        self.ent_savedata.grid(row=row, column=1, columnspan=3, sticky="we")
        self.tt_savedata = Tooltip(self.ent_savedata, "")
        self.btn_detect = ttk.Button(grp_src, text="Detect (Steam)", width=14, command=lambda: self._refresh_levels(from_detect=True))
        self.btn_detect.grid(row=row, column=4)
        self.btn_browse_savedata = ttk.Button(grp_src, text="Browse…", width=10, command=self.browse_savedata)
        self.btn_browse_savedata.grid(row=row, column=5)
        row += 1

        ttk.Label(grp_src, text="Level:").grid(row=row, column=0, sticky="w")
        self.combo_level = ttk.Combobox(grp_src, textvariable=self.var_level, state="disabled")
        self.combo_level.grid(row=row, column=1, columnspan=3, sticky="we")
        self.btn_refresh = ttk.Button(grp_src, text="Refresh", width=10, command=self._refresh_levels)
        self.btn_refresh.grid(row=row, column=4)
        row += 1
        self.lbl_sav_status = ttk.Label(grp_src, textvariable=self.sav_status)
        self.lbl_sav_status.grid(row=row, column=1, columnspan=5, sticky="w", pady=(2, 0))
        self.combo_level.bind("<<ComboboxSelected>>", lambda e: self._apply_level_selection())
        grp_src.grid_columnconfigure(0, weight=0)
        for c in (1, 2, 3):
            grp_src.grid_columnconfigure(c, weight=1)
        grp_src.grid_columnconfigure(4, weight=0)
        grp_src.grid_columnconfigure(5, weight=0)

        # Output group
        grp_out = ttk.Labelframe(content, text="2) Output")
        grp_out.pack(fill="x", expand=False, pady=(8, 0))

        row = 0
        ttk.Label(grp_out, text="Folder:").grid(row=row, column=0, sticky="w")
        self.ent_out = ttk.Entry(grp_out, textvariable=self.var_out_display, state="readonly")
        self.ent_out.grid(row=row, column=1, columnspan=3, sticky="we")
        self.tt_out = Tooltip(self.ent_out, "")
        self.btn_browse_out = ttk.Button(grp_out, text="Browse…", command=self.browse_out)
        self.btn_browse_out.grid(row=row, column=4)
        self.btn_open_out = ttk.Button(grp_out, text="Open", command=self.open_out_folder)
        self.btn_open_out.grid(row=row, column=5)
        row += 1

        ttk.Label(grp_out, text="Format:").grid(row=row, column=0, sticky="w")
        self.combo_fmt = ttk.Combobox(grp_out, textvariable=self.var_fmt, values=["mp4","gif"], state="readonly", width=10)
        self.combo_fmt.grid(row=row, column=1, columnspan=3, sticky="we")
        grp_out.grid_columnconfigure(0, weight=0)
        for c in (1, 2, 3):
            grp_out.grid_columnconfigure(c, weight=1)
        grp_out.grid_columnconfigure(4, weight=0)
        grp_out.grid_columnconfigure(5, weight=0)

        # Export group
        grp_x = ttk.Labelframe(content, text="3) Export")
        grp_x.pack(fill="x", expand=False, pady=(8, 0))

        self.btn_start = ttk.Button(grp_x, text="Start Export", command=self.start, width=18)
        self.btn_start.grid(row=0, column=0, sticky="w")
        self.btn_cancel = ttk.Button(grp_x, text="Cancel", command=self.cancel, width=12, state="disabled")
        self.btn_cancel.grid(row=0, column=1, sticky="w", padx=(8,0))
        self.btn_open = ttk.Button(grp_x, text="Open output", command=self.open_out_folder, width=14, state="disabled")
        self.btn_open.grid(row=0, column=2, sticky="w", padx=(8,0))

        self.prog = ttk.Progressbar(grp_x, orient="horizontal", mode="determinate")
        self.prog.grid(row=1, column=0, columnspan=6, sticky="we", pady=(10, 2))
        for c in range(6): grp_x.grid_columnconfigure(c, weight=1)

        self.status_bar = ttk.Label(self, textvariable=self.status_var, anchor="w", relief="sunken", padding=(10, 2))
        self.status_bar.pack(fill="x", side="bottom")

        self._toggle_widgets = [
            self.ent_savedata,
            self.btn_detect,
            self.btn_browse_savedata,
            self.combo_level,
            self.btn_refresh,
            self.ent_out,
            self.btn_browse_out,
            self.btn_open_out,
            self.combo_fmt,
        ]

        # react/save settings
        for v in (self.var_savedata_real, self.var_level, self.var_out_real, self.var_fmt):
            v.trace_add("write", lambda *_: self._on_state_change())

    # ----- Helpers / actions -----
    def _sync_path_labels(self):
        sd = self.var_savedata_real.get().strip()
        self.var_savedata_display.set("Steam save detected" if sd and default_savedata_dir() and str(default_savedata_dir())==sd else (shorten_path(sd) if sd else "—"))
        self.tt_savedata.set(sd)

        od = self.var_out_real.get().strip()
        self.var_out_display.set(shorten_path(od) if od else "—")
        self.tt_out.set(od)

    def browse_savedata(self):
        p = filedialog.askdirectory(title="Select PW2 SaveData folder")
        if p:
            self.var_savedata_real.set(p)
            self._refresh_levels()

    def browse_out(self):
        p = filedialog.askdirectory(title="Select output folder")
        if p:
            self.var_out_real.set(p)
            self._sync_path_labels()

    def _refresh_levels(self, auto: bool=False, from_detect: bool=False):
        base = None
        if from_detect or (auto and not self.var_savedata_real.get().strip()):
            base = default_savedata_dir()
            if base: self.var_savedata_real.set(str(base))
        else:
            base = Path(self.var_savedata_real.get()) if self.var_savedata_real.get().strip() else default_savedata_dir()

        levels = list_levels(base) if base else []
        self.levels = levels
        names = [name for name, _ in levels]
        self.combo_level["values"] = names
        if names:
            self.combo_level.config(state="readonly")
            cur = self.var_level.get()
            if cur not in names:
                self.var_level.set(names[0])
            self._apply_level_selection()
        else:
            self.combo_level.config(state="disabled")
            self.var_level.set("")
            self.sav_status.set("No timelapse.sav found in this folder.")
        self._sync_path_labels()
        self._update_controls_state()

    def _apply_level_selection(self):
        sel = self.var_level.get()
        match = next((p for (n, p) in self.levels if n == sel), None)
        if not match:
            self.sav_status.set("Select a level.")
            return
        tl = match / "timelapse.sav"
        if tl.exists():
            self.sav_status.set(f"timelapse.sav found ({sizeof_fmt(tl.stat().st_size)})")
            # set default output to nice location grouped by level
            root = default_video_root()
            out_dir = root / sel
            self.var_out_real.set(str(out_dir))
            self._sync_path_labels()
        else:
            self.sav_status.set("timelapse.sav missing in level.")
        self._update_controls_state()

    def open_level_folder(self):
        sel = self.var_level.get()
        match = next((p for (n, p) in self.levels if n == sel), None)
        if match: self._open_file_browser(match)

    def open_out_folder(self):
        p = self.var_out_real.get().strip()
        if p: self._open_file_browser(Path(p))

    def _open_file_browser(self, path: Path):
        if not path.exists(): path.mkdir(parents=True, exist_ok=True)
        if sys.platform.startswith("win"):
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=False)
        else:
            subprocess.run(["xdg-open", str(path)], check=False)

    def _on_state_change(self):
        self.can_open_output = False
        self._update_controls_state()
        save_settings({
            "savedata_dir": self.var_savedata_real.get(),
            "out_dir": self.var_out_real.get(),
            "fmt": self.var_fmt.get(),
        })

    def _update_controls_state(self):
        has_sd = bool(self.var_savedata_real.get().strip())
        level_ok = bool(self.var_level.get().strip())
        out_ok = bool(self.var_out_real.get().strip())
        if self.exporting:
            self.btn_start.config(state="disabled")
        else:
            self.btn_start.config(state="normal" if (has_sd and level_ok and out_ok) else "disabled")
        if self.exporting and not self.cancel_event.is_set():
            self.btn_cancel.config(state="normal")
        else:
            self.btn_cancel.config(state="disabled")
        self.btn_open.config(state="normal" if self.can_open_output else "disabled")

    def _set_export_mode(self, exporting: bool):
        if exporting == self.exporting:
            return
        if exporting:
            self._locked_states = {}
            for widget in self._toggle_widgets:
                try:
                    self._locked_states[widget] = widget.cget("state")
                    widget.config(state="disabled")
                except tk.TclError:
                    pass
        else:
            for widget, prev in self._locked_states.items():
                try:
                    widget.config(state=prev)
                except tk.TclError:
                    pass
            self._locked_states = {}
        self.exporting = exporting
        self._update_controls_state()

    # ----- Export flow -----
    def start(self):
        sel = self.var_level.get()
        level_dir = next((p for (n, p) in self.levels if n == sel), None)
        if not level_dir:
            messagebox.showerror("Missing level", "Please select a level.")
            return
        sav_file = level_dir / "timelapse.sav"
        out_dir = Path(self.var_out_real.get()).expanduser()
        fmt = self.var_fmt.get().lower()
        if not sav_file.exists():
            messagebox.showerror("Missing file", f"{sav_file} not found.")
            return
        out_dir.mkdir(parents=True, exist_ok=True)

        self.cancel_event.clear()
        self.can_open_output = False
        self._set_export_mode(True)
        self.btn_start.config(text="Exporting...")
        self.prog.config(value=0, maximum=100)
        self.status_var.set("Exporting... 0%")
        self.start_time = time.time()

        def _progress(cur: int, total: int):
            self.msgq.put(("progress", (cur, total)))
        def _worker():
            try:
                out = convert_timelapse(
                    sav_file, out_dir, fmt=fmt,
                    progress=_progress, cancel_event=self.cancel_event
                )
                self.msgq.put(("done", (out,)))
            except Exception as e:
                self.msgq.put(("error", (str(e),)))

        threading.Thread(target=_worker, daemon=True).start()

    def cancel(self):
        self.cancel_event.set()
        self.status_var.set("Cancelling...")
        self._update_controls_state()

    def _drain_queue(self):
        try:
            while True:
                kind, args = self.msgq.get_nowait()
                if kind == "progress":
                    cur, total = args
                    pct = int((cur / max(1, total)) * 100)
                    self.prog.config(value=pct)
                    txt = f"Exporting... {pct}% ({cur}/{total})"
                    if self.start_time and cur > 0:
                        elapsed = time.time() - self.start_time
                        rate = cur / elapsed if elapsed > 0 else 0
                        remaining = (total - cur) / rate if rate > 0 else 0
                        txt += f" | ETA {int(remaining)}s"
                    self.status_var.set(txt)
                elif kind == "done":
                    (out_path,) = args
                    self.can_open_output = True
                    self._set_export_mode(False)
                    self.btn_start.config(text="Start Export")
                    self.prog.config(value=100)
                    self.status_var.set(f"Done! Exported to {out_path}")
                    self.start_time = None
                elif kind == "error":
                    (msg,) = args
                    self.can_open_output = False
                    self._set_export_mode(False)
                    self.btn_start.config(text="Start Export")
                    self.status_var.set("Error")
                    self.start_time = None
                    messagebox.showerror("Export failed", msg if "Cancelled" not in msg else "Export was cancelled.")
        except queue.Empty:
            pass
        self.after(80, self._drain_queue)

def main():
    app = App()
    app.mainloop()

if __name__ == "__main__":
    main()
