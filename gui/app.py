"""GUI 主窗口：Tkinter + matplotlib 嵌入。

双击就能跑，不需要起服务器、不需要额外依赖（tkinter 是标准库，matplotlib 已在 dev 依赖里）。
界面分五块：
    顶部工具条   品牌 / 用例 / 尺寸 / 打开图片 / 运行 / 自动刷新 / 导出 / 坐标
    发丝线       金色渐隐分割线，纯装饰（对应站点那条 gradient rule）
    左侧面板     算子树 + 动态参数 + 实现变体 + 信息格视图 + 参数扫描
    右侧画布     2×2 四格（渲染逻辑全在 render.py）
    底部状态区   误差 / 耗时 / 口径说明 / 坑提示

视觉规范取自 www.wanzi.me 的 CSS 变量：#070707 页面底、#0d0d0e 卡片、#262626 描边、
#d8b46a 强调金、等宽字体走数字。色值全部集中在 THEME，改一处即可整体换肤。

事件流：任何控件变化 -> _schedule() 防抖 150ms -> _recompute() ->
engine.run() -> render.draw() -> 状态区刷新。自动刷新关掉时只更新参数显示，
等你点「运行」。

参数扫描是另一条路：_sweep_tick 每帧改一个参数值后直接 _recompute()（不经过防抖），
画布逐帧刷新，扫完把该指标的 min~max 写进状态区。取值序列由 params.sweep_sequence 给。
帧会顺路缩到 960 宽缓存，扫完（或中途停止）点「存 GIF」导出循环动图，帧间隔与屏幕一致。
"""

from __future__ import annotations

import pathlib
import sys
import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog, ttk

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gui import engine, params as P, render   # noqa: E402

SIZES = (64, 96, 128, 192, 256, 384, 512)
DEBOUNCE_MS = 150
FRAME_MS = 150          # 参数扫描的帧间隔：比防抖稍慢一点，肉眼跟得上
SWEEP_STEPS = (12, 24, 36, 48)
GIF_WIDTHS = ("480", "720", "960")   # 导出宽度：缓存帧就是 960，只往下缩不放大
GIF_SKIPS = ("1", "2", "3", "4")     # 抽帧：隔 k 帧取一帧，帧间隔同步乘 k

_OK = ("逐像素精确", "一致", "浮点噪声级一致", "IoU 合格")
_WARN = ("仅记录", "两边都无边缘")
_BAD = ("超出容差", "不一致", "IoU 偏低")

# --- 主题：wanzi.me 的 :root 设计变量 ----------------------------------------
THEME = {
    "bg": "#070707",          # --bg：页面底
    "bg_soft": "#0a0a0b",     # --bg-soft：树、状态区、输入框底
    "panel": "#0d0d0e",       # --panel：卡片 / 控件底
    "hover": "#161617",
    "line": "#262626",        # --line：常规描边
    "line_soft": "#1b1b1c",   # --line-soft：卡片描边
    "hover_line": "#3b3b3c",
    "ink": "#f3f3f1",         # --ink：主文字
    "ink_soft": "#ededeb",
    "muted": "#9b9ea2",       # --muted：次要文字
    "faint": "#6b6d70",       # --faint：最弱
    "gold": "#d8b46a",        # 强调金（active tab 下划线、渐变标题都用它）
    "gold_dim": "#b98f3e",
    "gold_pale": "#f6e7c4",
    "gold_soft": "#1a1610",   # 金色按钮 / 选中项的底
    "gold_line": "#5d4d2b",
    "ok": "#86c991",
    "warn": "#d8b46a",
    "bad": "#e0707a",
    "info": "#c9cbcc",
    "note": "#6b6d70",
}

MONO_CANDIDATES = ["Cascadia Mono", "Cascadia Code", "JetBrains Mono", "Consolas",
                   "DejaVu Sans Mono"]
CN_CANDIDATES = ["HarmonyOS Sans SC", "Microsoft YaHei UI", "Microsoft YaHei",
                 "PingFang SC", "Noto Sans CJK SC", "SimHei"]


def _hex_rgb(c: str) -> tuple[int, int, int]:
    c = c.lstrip("#")
    return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)


def _blend(fg: str, bg: str, a: float) -> str:
    """把 fg 按透明度 a 混到 bg 上，返回十六进制色。"""
    f, b = _hex_rgb(fg), _hex_rgb(bg)
    return "#%02x%02x%02x" % tuple(int(round(b[i] + (f[i] - b[i]) * a)) for i in range(3))


def assemble_gif(frames, path, duration_ms: int = FRAME_MS) -> int:
    """把一列 PIL 帧写成循环 GIF，返回文件字节数。

    调色板取自「全部帧缩略图拼成的马赛克」：只用第一帧的话，万一首帧色少
    （比如纯色用例），后面帧的颜色会被量化碾平、帧间全同被 PIL 并帧；
    各帧独立调色板又会闪。马赛克用最近邻取样、再贴上主题与渲染强调色的
    标准色块，1px 细彩线也不会在取色时丢。不加抖动——界面本来就是大平色块，
    抖出来的噪点只会把体积撑大。
    """
    from PIL import Image
    thumbs = [f.convert("RGB").resize((64, 40), Image.NEAREST) for f in frames]
    swatches = [THEME[k] for k in ("bg", "panel", "ink", "muted", "gold")] + [
        render._C_MINE, render._C_REF, render._C_INPUT] + [
        "#%02x%02x%02x" % tuple(round(c * 255) for c in rgb)
        for rgb in (render._C_EDGE_MINE, render._C_EDGE_REF, render._C_EDGE_BOTH)]
    strip = Image.new("RGB", (64 * len(thumbs) + 16 * len(swatches), 40))
    x = 0
    for t in thumbs:
        strip.paste(t, (x, 0))
        x += 64
    for c in swatches:
        strip.paste(Image.new("RGB", (16, 40), c), (x, 0))
        x += 16
    pal = strip.quantize(colors=256, dither=Image.NONE)
    q = [f.convert("RGB").quantize(palette=pal, dither=Image.NONE) for f in frames]
    q[0].save(path, save_all=True, append_images=q[1:],
              duration=int(duration_ms), loop=0, optimize=True)
    return pathlib.Path(path).stat().st_size


def prepare_gif_frames(frames, width: int, skip: int):
    """导出前的帧处理：隔 skip 帧取一帧，再缩到 width 宽。

    返回 (帧列表, 每帧毫秒)。帧间隔乘 skip，这样抽帧之后播放速度仍接近
    扫描时的实时速度，而不是变成快进。不改动传入的缓存帧。
    """
    from PIL import Image
    k = max(1, int(skip))
    sel = list(frames[::k])
    w = max(1, int(width))
    if sel and sel[0].width != w:
        h = max(1, round(sel[0].height * w / sel[0].width))
        sel = [f.resize((w, h), Image.LANCZOS) for f in sel]
    return sel, FRAME_MS * k


def _pick_family(root, candidates, fallback: str) -> str:
    try:
        fams = set(tkfont.families(root))
    except Exception:
        return candidates[0]
    for c in candidates:
        if c in fams:
            return c
    return fallback


def _apply_theme(root: tk.Tk) -> dict:
    """把站点的设计变量翻译成 ttk 样式，返回一套字体元组给 tk 控件用。"""
    default_cn = tkfont.nametofont("TkDefaultFont").actual("family")
    mono = _pick_family(root, MONO_CANDIDATES, "Consolas")
    cn = _pick_family(root, CN_CANDIDATES, default_cn or "TkDefaultFont")
    fonts = {
        "mono": mono, "cn": cn,
        "mono9": (mono, 9), "mono10": (mono, 10),
        "cn9": (cn, 9), "cn10": (cn, 10),
        "brand": (mono, 12, "bold"),
        "tagline": (mono, 9),
        "head": (cn, 9),
        "label": (mono, 9),
    }

    root.configure(bg=THEME["bg"])
    # 原生 tk 控件（Text / Canvas / 文件对话框 / matplotlib 工具条）走 X 资源
    root.option_add("*Background", THEME["bg"])
    root.option_add("*Foreground", THEME["ink"])
    root.option_add("*selectBackground", THEME["gold_soft"])
    root.option_add("*selectForeground", THEME["gold"])
    root.option_add("*insertBackground", THEME["gold"])
    root.option_add("*activeBackground", THEME["hover"])
    root.option_add("*activeForeground", THEME["ink_soft"])
    root.option_add("*disabledForeground", THEME["faint"])
    root.option_add("*highlightThickness", 1)
    root.option_add("*highlightBackground", THEME["line"])
    root.option_add("*highlightColor", THEME["gold_line"])
    # 下拉框弹出层是 Tk 的 Listbox，ttk 样式管不到，只能这样递进去
    root.option_add("*TCombobox*Listbox*Background", THEME["panel"])
    root.option_add("*TCombobox*Listbox*Foreground", THEME["ink"])
    root.option_add("*TCombobox*Listbox*selectBackground", THEME["gold_soft"])
    root.option_add("*TCombobox*Listbox*selectForeground", THEME["gold"])
    root.option_add("*TCombobox*Listbox*BorderWidth", 1)
    root.option_add("*TCombobox*Listbox*Relief", "solid")

    st = ttk.Style(root)
    if "clam" in st.theme_names():
        st.theme_use("clam")          # vista/aqua 不吃自定义配色，clam 才听话

    st.configure(".", background=THEME["bg"], foreground=THEME["ink"],
                 bordercolor=THEME["line"], lightcolor=THEME["panel"],
                 darkcolor=THEME["panel"], font=fonts["cn9"])
    st.configure("TFrame", background=THEME["bg"])
    st.configure("Card.TFrame", background=THEME["panel"])
    st.configure("TLabel", background=THEME["bg"], foreground=THEME["ink"])
    st.configure("Card.TLabel", background=THEME["panel"], foreground=THEME["ink"])
    st.configure("Muted.TLabel", background=THEME["bg"], foreground=THEME["muted"])
    st.configure("Faint.TLabel", background=THEME["bg"], foreground=THEME["faint"],
                 font=fonts["mono9"])
    st.configure("Card.Muted.TLabel", background=THEME["panel"], foreground=THEME["muted"])
    st.configure("Card.Faint.TLabel", background=THEME["panel"], foreground=THEME["faint"],
                 font=fonts["mono9"])
    st.configure("Brand.TLabel", background=THEME["bg"], foreground=THEME["gold"],
                 font=fonts["brand"])
    st.configure("Tag.TLabel", background=THEME["bg"], foreground=THEME["faint"],
                 font=fonts["tagline"])
    st.configure("Head.TLabel", background=THEME["bg"], foreground=THEME["muted"],
                 font=fonts["head"])

    st.configure("TButton", background=THEME["panel"], foreground=THEME["ink_soft"],
                 bordercolor=THEME["line"], focuscolor=THEME["gold_line"],
                 padding=(10, 4))
    st.map("TButton",
           background=[("active", THEME["hover"]), ("pressed", THEME["panel"]),
                       ("disabled", THEME["bg_soft"])],
           bordercolor=[("active", THEME["hover_line"])],
           foreground=[("active", THEME["ink"]), ("disabled", THEME["faint"])])
    st.configure("Gold.TButton", background=THEME["gold_soft"], foreground=THEME["gold"],
                 bordercolor=THEME["gold_line"], focuscolor=THEME["gold"], padding=(10, 4))
    st.map("Gold.TButton",
           background=[("active", THEME["gold"]), ("pressed", THEME["gold_soft"]),
                       ("disabled", THEME["bg_soft"])],
           foreground=[("active", "#0b0b0c"), ("disabled", THEME["faint"])],
           bordercolor=[("active", THEME["gold"])])

    st.configure("TCheckbutton", background=THEME["bg"], foreground=THEME["ink_soft"],
                 focuscolor=THEME["gold_line"])
    st.map("TCheckbutton", background=[("active", THEME["bg"])],
           indicatorcolor=[("selected", THEME["gold"]), ("!selected", THEME["bg_soft"])])
    st.configure("Card.TRadiobutton", background=THEME["panel"], foreground=THEME["ink_soft"],
                 focuscolor=THEME["gold_line"])
    st.map("Card.TRadiobutton", background=[("active", THEME["panel"])],
           indicatorcolor=[("selected", THEME["gold"]), ("!selected", THEME["bg_soft"])])

    st.configure("TLabelframe", background=THEME["panel"], bordercolor=THEME["line_soft"],
                 relief="solid", borderwidth=1)
    st.configure("TLabelframe.Label", background=THEME["panel"], foreground=THEME["gold"],
                 font=fonts["label"])

    st.configure("TCombobox", fieldbackground=THEME["bg_soft"], background=THEME["panel"],
                 foreground=THEME["ink"], arrowcolor=THEME["muted"],
                 bordercolor=THEME["line"], lightcolor=THEME["panel"],
                 darkcolor=THEME["panel"], selectbackground=THEME["gold_soft"],
                 selectforeground=THEME["gold"], padding=3)
    st.map("TCombobox",
           fieldbackground=[("readonly", THEME["bg_soft"]), ("disabled", THEME["bg"])],
           foreground=[("readonly", THEME["ink"]), ("disabled", THEME["faint"])],
           arrowcolor=[("disabled", THEME["faint"])],
           selectbackground=[("readonly", THEME["bg_soft"])],
           selectforeground=[("readonly", THEME["ink"])],
           bordercolor=[("active", THEME["hover_line"]), ("focus", THEME["gold_line"])])

    st.configure("TEntry", fieldbackground=THEME["bg_soft"], foreground=THEME["ink"],
                 insertcolor=THEME["gold"], bordercolor=THEME["line"],
                 lightcolor=THEME["line"], darkcolor=THEME["line"], padding=2)
    st.map("TEntry", bordercolor=[("focus", THEME["gold_line"])],
           lightcolor=[("focus", THEME["gold_line"])],
           foreground=[("disabled", THEME["faint"])])

    # rowheight 是物理像素单位，写死 21 在 200% 缩放下会把字拦腰截断，
    # 按字体实测行高给
    row_h = tkfont.Font(root=root, font=fonts["mono9"]).metrics("linespace") + 8
    st.configure("Treeview", background=THEME["bg_soft"], fieldbackground=THEME["bg_soft"],
                 foreground=THEME["ink_soft"], bordercolor=THEME["line"],
                 borderwidth=1, relief="solid", rowheight=row_h, font=fonts["mono9"])
    st.map("Treeview", background=[("selected", THEME["gold_soft"])],
           foreground=[("selected", THEME["gold"])])
    st.configure("Treeview.Heading", background=THEME["panel"], foreground=THEME["muted"])

    st.configure("Horizontal.TScale", background=THEME["gold"], troughcolor=THEME["bg_soft"],
                 bordercolor=THEME["line"], darkcolor=THEME["gold_dim"],
                 lightcolor=THEME["gold_pale"], sliderthickness=11)
    st.map("Horizontal.TScale", background=[("active", THEME["gold_pale"])])
    st.configure("TProgressbar", background=THEME["gold"], troughcolor=THEME["bg_soft"],
                 bordercolor=THEME["line"], lightcolor=THEME["gold"],
                 darkcolor=THEME["gold"])
    st.configure("Vertical.TScrollbar", background=THEME["panel"], troughcolor=THEME["bg"],
                 bordercolor=THEME["line"], arrowcolor=THEME["muted"],
                 lightcolor=THEME["line"], darkcolor=THEME["line"])
    st.map("Vertical.TScrollbar", background=[("active", THEME["hover"])],
           arrowcolor=[("active", THEME["gold"])])
    return fonts


def _style_toolbar(tb) -> None:
    """matplotlib 的 NavigationToolbar2Tk 用的是原生 tk.Button，ttk 样式管不到。

    3.9 的 _set_image_for_button 会读按钮的 foreground 生成一版浅色图标，
    再按 background 亮暗挑用哪一版 —— 所以只要把底色改成暗的、前景改成亮的，
    然后调一次 _rescale() 让它重算图标，工具条就跟上了暗色主题。
    """
    tb.configure(bg=THEME["bg"], highlightbackground=THEME["bg"], bd=0)
    for w in tb.winfo_children():
        if isinstance(w, (tk.Button, tk.Checkbutton)):
            w.configure(bg=THEME["bg"], fg=THEME["muted"], activebackground=THEME["hover"],
                        activeforeground=THEME["gold"], highlightbackground=THEME["bg"],
                        highlightthickness=0, bd=0)
            if isinstance(w, tk.Checkbutton):     # selectcolor 只有 Checkbutton 认
                w.configure(selectcolor=THEME["panel"])
        elif isinstance(w, tk.Label):
            w.configure(bg=THEME["bg"], fg=THEME["muted"])
        elif isinstance(w, tk.Frame):
            w.configure(bg=THEME["bg"])
    try:
        tb._rescale()
    except Exception:
        pass


def _dpi_aware() -> None:
    """Windows 下让 tkinter 用真实 DPI，否则高分屏上整窗发虚。"""
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass


def _initial_geometry(root: tk.Tk) -> None:
    """按 DPI 缩放给初始窗口尺寸。

    SetProcessDpiAwareness 之后 tkinter 的 geometry 单位是物理像素，
    但字体大小走的是点（会被系统缩放放大），150% 缩放下 1420x900 物理像素
    只相当于 946x600 的逻辑窗口，控件全挤在一起。这里按 tk scaling 换算，
    再拿屏幕尺寸兜底。
    """
    try:
        scale = root.winfo_fpixels("1i") / 96.0
    except Exception:
        scale = 1.0
    scale = max(1.0, min(scale, 3.0))
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    w = int(min(1420 * scale, sw * 0.96))
    h = int(min(900 * scale, sh * 0.92))
    root.geometry(f"{w}x{h}")
    root.minsize(int(1120 * scale), int(720 * scale))


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("myimg 图像算子对拍台 —— 手写 NumPy vs OpenCV")
        _initial_geometry(root)

        render.setup_style()
        self.fonts = _apply_theme(root)

        # --- 状态 -----------------------------------------------------------
        self.cur_op = "gaussian_filter"
        self.cur_variant = "vector"
        self.cur_case = P.DEFAULT_CASE_FOR_GRAY
        self.cur_size = 128
        self.cur_view: str | None = None
        self.res = None
        self.real_imgs: dict[str, np.ndarray] = {}
        self.file_img: np.ndarray | None = None
        self.file_name = ""
        self.param_vars: dict[str, tk.Variable] = {}
        self.param_entries: dict[str, ttk.Entry] = {}
        self._debounce: str | None = None
        self._hairline_job: str | None = None
        # 参数扫描
        self._sweep_active = False
        self._sweep_after: str | None = None
        self._sweep_param_def = None
        self._sweep_seq: list = []
        self._sweep_i = 0
        self._sweep_rows: list = []
        self._sweep_frames: list = []      # GIF 帧：扫描途中逐帧缩到 960 宽存着
        self._sweep_meta: tuple | None = None

        self.auto_var = tk.BooleanVar(value=True)
        self.variant_var = tk.StringVar(value="vector")
        self.case_var = tk.StringVar()
        self.size_var = tk.StringVar(value="128")
        self.view_var = tk.StringVar()
        self.coord_var = tk.StringVar(value="")
        self.sweep_var = tk.StringVar()
        self.sweep_steps_var = tk.StringVar(value="24")
        self.sweep_info_var = tk.StringVar(value="")
        self.gif_w_var = tk.StringVar(value="960")
        self.gif_skip_var = tk.StringVar(value="1")

        self._build_toolbar()
        self._build_hairline()
        self._build_left()
        self._build_canvas()
        self._build_status()

        self._refresh_case_options()
        self._select_op(self.cur_op)

    # ---------------------------------------------------------------- 布局
    def _build_toolbar(self):
        bar = ttk.Frame(self.root, padding=(10, 8, 10, 4))
        bar.grid(row=0, column=0, columnspan=2, sticky="ew")

        # 右侧坐标先 pack，后面控件才不会把它挤下去
        ttk.Label(bar, textvariable=self.coord_var, style="Faint.TLabel").pack(side="right")

        ttk.Label(bar, text="myimg", style="Brand.TLabel").pack(side="left")
        ttk.Label(bar, text="hand-written numpy ops vs opencv",
                  style="Tag.TLabel").pack(side="left", padx=(10, 22))

        ttk.Label(bar, text="用例", style="Head.TLabel").pack(side="left", padx=(0, 4))
        self.case_cb = ttk.Combobox(bar, textvariable=self.case_var,
                                    state="readonly", width=34, font=self.fonts["mono9"])
        self.case_cb.pack(side="left", padx=(0, 12))
        self.case_cb.bind("<<ComboboxSelected>>", lambda e: self._on_case_change())

        ttk.Label(bar, text="尺寸", style="Head.TLabel").pack(side="left", padx=(0, 4))
        self.size_cb = ttk.Combobox(bar, textvariable=self.size_var, state="readonly",
                                    width=6, values=SIZES, font=self.fonts["mono9"])
        self.size_cb.pack(side="left", padx=(0, 12))
        self.size_cb.bind("<<ComboboxSelected>>", lambda e: self._on_size_change())

        ttk.Button(bar, text="打开图片", width=10,
                   command=self._open_file).pack(side="left", padx=(0, 6))
        self.run_btn = ttk.Button(bar, text="运行对比", width=10, style="Gold.TButton",
                                  command=self._recompute)
        self.run_btn.pack(side="left", padx=(0, 6))
        ttk.Checkbutton(bar, text="自动刷新", variable=self.auto_var,
                        command=self._schedule).pack(side="left", padx=(0, 6))
        ttk.Button(bar, text="导出 PNG", width=10,
                   command=self._export).pack(side="left")

    def _build_hairline(self):
        """金色渐隐发丝线：站点的 .rule 就是这个效果，纯装饰但对观感很关键。"""
        c = tk.Canvas(self.root, width=1, height=2, bd=0, highlightthickness=0,
                      bg=THEME["bg"])
        c.grid(row=1, column=0, columnspan=2, sticky="ew")
        self.hairline = c
        c.bind("<Configure>", lambda e: self._on_hairline_configure())
        self._draw_hairline()

    def _on_hairline_configure(self):
        if self._hairline_job is not None:
            self.root.after_cancel(self._hairline_job)
        self._hairline_job = self.root.after_idle(self._draw_hairline)

    def _draw_hairline(self):
        self._hairline_job = None
        c = self.hairline
        try:
            w = c.winfo_width()
        except tk.TclError:
            return
        if w <= 1:
            return
        c.delete("all")
        step = 4
        for x0 in range(0, w, step):
            t = x0 / max(1, w - 1)
            a = (1.0 - t) ** 1.6 * 0.85          # 左端最亮，往右指数衰减到背景
            c.create_rectangle(x0, 0, min(x0 + step, w), 2,
                               fill=_blend(THEME["gold"], THEME["bg"], a), outline="")

    def _build_left(self):
        left = ttk.Frame(self.root, padding=(10, 6, 4, 8))
        left.grid(row=2, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=0, minsize=340)

        ttk.Label(left, text="算子", style="Head.TLabel").grid(row=0, column=0,
                                                              sticky="w", pady=(0, 3))
        # 树单独占一个 Frame + 滚动条：19 行算子放不下时靠滚轮，
        # 而不是被 grid 把行高压扁（压扁会把文字拦腰截断）
        tf = ttk.Frame(left)
        tf.grid(row=1, column=0, sticky="nsew")
        self.tree = ttk.Treeview(tf, show="tree", height=9, selectmode="browse")
        self.tree.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(tf, orient="vertical", command=self.tree.yview)
        sb.pack(side="right", fill="y")
        self.tree.configure(yscrollcommand=sb.set)
        left.rowconfigure(1, weight=1)
        self._op_nodes: dict[str, str] = {}
        for gid, (gname, ops) in enumerate(P.OP_GROUPS):
            gnode = self.tree.insert("", "end", text=gname, open=True)
            for op in ops:
                node = self.tree.insert(gnode, "end", text=op)
                self._op_nodes[op] = node
        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)

        pf = ttk.LabelFrame(left, text="参数", padding=6)
        pf.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        self.param_frame = pf

        vf = ttk.LabelFrame(left, text="实现变体", padding=6)
        vf.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        self.variant_frame = vf

        sf = ttk.LabelFrame(left, text="参数扫描", padding=6)
        sf.grid(row=4, column=0, sticky="ew", pady=(10, 0))
        self._build_sweep(sf)

        ttk.Label(left, text="右下信息格", style="Head.TLabel").grid(
            row=5, column=0, sticky="w", pady=(12, 3))
        self.view_cb = ttk.Combobox(left, textvariable=self.view_var,
                                    state="readonly", font=self.fonts["cn9"])
        self.view_cb.grid(row=6, column=0, sticky="ew")
        self.view_cb.bind("<<ComboboxSelected>>", lambda e: self._on_view_change())

        left.columnconfigure(0, weight=1)

    def _build_sweep(self, sf):
        """参数扫描卡片：选一个参数，逐帧把画布从 lo 扫到 hi。"""
        ttk.Label(sf, text="参数", style="Card.TLabel").grid(row=0, column=0, sticky="w")
        self.sweep_cb = ttk.Combobox(sf, textvariable=self.sweep_var, state="readonly",
                                     width=13, font=self.fonts["cn9"])
        self.sweep_cb.grid(row=0, column=1, sticky="w", padx=(4, 10))

        ttk.Label(sf, text="帧数", style="Card.TLabel").grid(row=0, column=2, sticky="w")
        self.sweep_steps_cb = ttk.Combobox(sf, textvariable=self.sweep_steps_var,
                                           state="readonly", width=4,
                                           values=SWEEP_STEPS, font=self.fonts["mono9"])
        self.sweep_steps_cb.grid(row=0, column=3, sticky="w", padx=(4, 0))

        self.sweep_btn = ttk.Button(sf, text="开始扫描", style="Gold.TButton", width=10,
                                    command=self._toggle_sweep)
        self.sweep_btn.grid(row=1, column=0, columnspan=2, sticky="w", pady=(7, 0))
        self.sweep_gif_btn = ttk.Button(sf, text="存 GIF", width=10, state="disabled",
                                        command=self._save_sweep_gif)
        self.sweep_gif_btn.grid(row=1, column=2, columnspan=2, sticky="w",
                                padx=(8, 0), pady=(7, 0))
        ttk.Label(sf, text="GIF 宽", style="Card.Faint.TLabel").grid(
            row=2, column=0, sticky="w", pady=(6, 0))
        self.gif_w_cb = ttk.Combobox(sf, textvariable=self.gif_w_var, state="readonly",
                                     width=4, values=GIF_WIDTHS, font=self.fonts["mono9"])
        self.gif_w_cb.grid(row=2, column=1, sticky="w", padx=(4, 10), pady=(6, 0))
        ttk.Label(sf, text="抽帧", style="Card.Faint.TLabel").grid(
            row=2, column=2, sticky="w", pady=(6, 0))
        self.gif_skip_cb = ttk.Combobox(sf, textvariable=self.gif_skip_var,
                                        state="readonly", width=4, values=GIF_SKIPS,
                                        font=self.fonts["mono9"])
        self.gif_skip_cb.grid(row=2, column=3, sticky="w", padx=(4, 0), pady=(6, 0))
        ttk.Label(sf, textvariable=self.sweep_info_var, style="Card.Faint.TLabel",
                  width=36, anchor="w", justify="left").grid(
            row=3, column=0, columnspan=4, sticky="w", pady=(6, 0))
        sf.columnconfigure(3, weight=1)

    def _build_canvas(self):
        from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                                       NavigationToolbar2Tk)

        wrap = ttk.Frame(self.root, padding=(4, 6, 10, 8))
        wrap.grid(row=2, column=1, sticky="nsew")
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(2, weight=1)
        wrap.rowconfigure(0, weight=1)
        wrap.columnconfigure(0, weight=1)

        self.fig = render.make_figure(dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=wrap)
        self.canvas.get_tk_widget().configure(background=THEME["bg"])
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        # NavigationToolbar2Tk 内部用 pack，不能直接塞进 grid 管理的容器，
        # 给它一个专属 Frame 隔离两种几何管理器
        toolframe = ttk.Frame(wrap)
        toolframe.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolframe)
        self.toolbar.update()
        _style_toolbar(self.toolbar)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        # 高分屏首帧的坑：canvas 在 <Map> 里把 device pixel ratio 从 1 提到 2
        # （figure.dpi 100 -> 200），随后靠 configure(widget) 触发 <Configure>
        # 重算 figure 英寸数。但 grid 布局下分配尺寸不变、<Configure> 不来，
        # figure 英寸数就停在旧 dpi 算的值 -> Agg 缓冲是 photo 的 4 倍大，
        # 屏幕上只剩左上 1/4（正好是输入格）。映射完成后用 matplotlib 自己的
        # resize() 对齐一次即可；"+" 是追加绑定，不覆盖 backend 的 <Map>。
        self.canvas.get_tk_widget().bind("<Map>", self._sync_canvas_size, "+")

    def _sync_canvas_size(self, event=None):
        """映射后把画布 widget 的"请求尺寸"收回去，让 grid 说了算。

        高分屏的坑分两层：
        1) <Map> 时 backend 把 figure.dpi 从 100 提到 200，并且顺手把 tk canvas
           的 width/height 请求改成新的物理尺寸（1880x1480）。grid 的格子不会
           比子控件的请求小，于是整个右栏被撑到窗口外 -> 四格被裁。
        2) 请求尺寸不收回去，窗口缩小时格子也缩不下去（grid 不缩到请求以下）。
        把请求改成 1x1 后，格子尺寸完全由窗口决定，backend 自己绑的
        <Configure> -> resize() 会把 figure 英寸数对齐到分配尺寸（forward=False，
        不会再改回请求），放大缩小都跟手。
        """
        import types

        def _do():
            try:
                w = self.canvas.get_tk_widget()
                w.configure(width=1, height=1)
                self.root.update_idletasks()
                aw, ah = w.winfo_width(), w.winfo_height()
                if aw < 80 or ah < 80:
                    return
                # 分配尺寸没变就不会有 <Configure>，backend 的 resize 不再跑，
                # 这里手动补一次，把 figure 英寸数按新 dpi 对齐到分配尺寸
                self.canvas.resize(types.SimpleNamespace(width=aw, height=ah))
            except tk.TclError:
                pass
        self.root.after_idle(_do)

    def _build_status(self):
        f = self.fonts
        self.status = tk.Text(self.root, height=7, wrap="word", relief="flat",
                              background=THEME["bg_soft"], foreground=THEME["ink_soft"],
                              font=f["mono10"], state="disabled", padx=10, pady=7,
                              highlightthickness=1, highlightbackground=THEME["line"],
                              highlightcolor=THEME["line"], insertbackground=THEME["gold"],
                              selectbackground=THEME["gold_soft"],
                              selectforeground=THEME["gold"])
        self.status.grid(row=3, column=0, columnspan=2, sticky="ew", padx=(10, 10),
                         pady=(0, 8))
        self.status.tag_configure("ok", foreground=THEME["ok"])
        self.status.tag_configure("warn", foreground=THEME["warn"])
        self.status.tag_configure("bad", foreground=THEME["bad"])
        self.status.tag_configure("info", foreground=THEME["info"])
        self.status.tag_configure("note", foreground=THEME["note"])

    # ---------------------------------------------------------------- 事件
    def _on_tree_select(self, event):
        sel = self.tree.selection()
        if not sel:
            return
        text = self.tree.item(sel[0], "text")
        # _select_op 里会 selection_set，那会再触发一次本回调；
        # 不拦掉就是无限递归（实测能把 CPU 吃满、界面永不返回）
        if text == self.cur_op or text not in self._op_nodes:
            return
        self._select_op(text)

    def _select_op(self, op: str):
        self._stop_sweep()
        self.cur_op = op
        self.tree.selection_set(self._op_nodes[op])
        # 形态学算子配灰度图没有意义，自动换到二值用例
        if op in P.BINARY_OPS and self.cur_case not in engine.BINARY_CASES \
                and self.file_img is None:
            self.cur_case = P.DEFAULT_CASE_FOR_BINARY
            self._refresh_case_options()
        elif op not in P.BINARY_OPS and self.cur_case in engine.BINARY_CASES:
            self.cur_case = P.DEFAULT_CASE_FOR_GRAY
            self._refresh_case_options()
        self._build_params(op)
        self._build_variants(op)
        self.cur_view = None
        self._recompute()

    def _build_params(self, op: str):
        for w in self.param_frame.winfo_children():
            w.destroy()
        self.param_vars.clear()
        self.param_entries.clear()
        params = P.PARAMS.get(op, ())
        if not params:
            ttk.Label(self.param_frame, text="（这个算子没有可调参数）",
                      style="Card.Muted.TLabel").grid(row=0, column=0, sticky="w")
            self._refresh_sweep_options()
            return
        for row, p in enumerate(params):
            ttk.Label(self.param_frame, text=p.label, style="Card.TLabel").grid(
                row=row, column=0, sticky="w", pady=2)
            if p.kind == "choice":
                var = tk.StringVar(value=str(p.default))
                cb = ttk.Combobox(self.param_frame, textvariable=var, state="readonly",
                                  width=12, values=list(p.options), font=self.fonts["mono9"])
                cb.grid(row=row, column=1, sticky="w", pady=2)
                var.trace_add("write", lambda *a: self._schedule())
                self.param_vars[p.key] = var
            else:
                var = tk.DoubleVar(value=float(p.default))
                sc = ttk.Scale(self.param_frame, from_=p.lo, to=p.hi,
                               variable=var, orient="horizontal")
                sc.grid(row=row, column=1, sticky="we", pady=2)
                ent = ttk.Entry(self.param_frame, width=7, font=self.fonts["mono9"])
                ent.grid(row=row, column=2, padx=(6, 0), pady=2)
                ent.insert(0, self._fmt_param(p, p.default))
                sc.configure(command=lambda v, p=p, e=ent: self._on_scale(p, v, e))
                ent.bind("<Return>", lambda e, p=p, s=sc, en=ent: self._on_entry(p, s, en))
                ent.bind("<FocusOut>", lambda e, p=p, s=sc, en=ent: self._on_entry(p, s, en))
                self.param_vars[p.key] = var
                self.param_entries[p.key] = ent
        self.param_frame.columnconfigure(1, weight=1)
        self._refresh_sweep_options()

    def _build_variants(self, op: str):
        for w in self.variant_frame.winfo_children():
            w.destroy()
        opts = P.variants(op)
        if self.cur_variant not in [k for k, _ in opts]:
            self.cur_variant = opts[0][0]
        self.variant_var.set(self.cur_variant)
        for i, (key, label) in enumerate(opts):
            ttk.Radiobutton(self.variant_frame, text=label, value=key,
                            variable=self.variant_var, style="Card.TRadiobutton",
                            command=self._on_variant_change).grid(row=i, column=0, sticky="w")
        if len(opts) == 1:
            ttk.Label(self.variant_frame, text="（只有这一种实现）",
                      style="Card.Muted.TLabel").grid(row=0, column=1, sticky="w", padx=(8, 0))

    def _refresh_sweep_options(self):
        """换算子后重填扫描参数下拉框，默认挑数值型（扫起来才有动画感）。"""
        if getattr(self, "sweep_cb", None) is None:
            return
        params = P.PARAMS.get(self.cur_op, ())
        labels = [p.label for p in params]
        self.sweep_cb.configure(values=labels)
        if not labels:
            self.sweep_var.set("")
            self.sweep_btn.configure(state="disabled")
            self.sweep_info_var.set("（这个算子没有可调参数）")
            return
        if self.sweep_var.get() not in labels:
            pref = next((p.label for p in params if p.kind in ("float", "int", "odd")),
                        labels[0])
            self.sweep_var.set(pref)
        self.sweep_btn.configure(state="normal")
        self.sweep_info_var.set("选好参数点「开始扫描」，画布逐帧刷新；扫完可存 GIF（宽/抽帧可选）")

    def _on_scale(self, p: P.Param, val: str, entry: ttk.Entry):
        try:
            v = float(val)
        except ValueError:
            return
        entry.delete(0, "end")
        entry.insert(0, self._fmt_param(p, v))
        self._schedule()

    def _on_entry(self, p: P.Param, scale: ttk.Scale, entry: ttk.Entry):
        try:
            v = float(entry.get())
        except ValueError:
            return
        v = min(max(v, p.lo), p.hi)
        scale.set(v)
        entry.delete(0, "end")
        entry.insert(0, self._fmt_param(p, v))
        self._schedule()

    @staticmethod
    def _fmt_param(p: P.Param, v) -> str:
        v = float(v)
        if p.kind == "float":
            return f"{v:.{p.digits}f}"
        return str(int(round(v)))

    def _on_variant_change(self):
        self._stop_sweep()
        self.cur_variant = self.variant_var.get()
        self._schedule()

    def _on_view_change(self):
        self.cur_view = self.view_var.get()
        if self.res is not None:
            render.draw(self.fig, self.res, self.cur_view)

    def _on_case_change(self):
        self._stop_sweep()
        label = self.case_var.get()
        self.cur_case = label.split(" — ")[0]
        self._schedule()

    def _on_size_change(self):
        self._stop_sweep()
        self.cur_size = int(self.size_var.get())
        self._schedule()

    def _open_file(self):
        self._stop_sweep()
        path = filedialog.askopenfilename(
            title="选一张图片",
            filetypes=[("图片", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("所有文件", "*.*")])
        if not path:
            return
        try:
            self.file_img = engine.open_file(path)
        except Exception as exc:
            self._status_error(f"读图失败：{exc}")
            return
        self.file_name = pathlib.Path(path).stem
        self.cur_case = f"file:{self.file_name}"
        self._refresh_case_options()
        self.size_cb.configure(state="disabled")
        self._schedule()

    # ---------------------------------------------------------------- 数据
    def _refresh_case_options(self):
        self.real_imgs = engine.real_images()
        opts = [f"{k} — {engine.CASE_LABELS[k]}" for k in engine.GRAY_CASES]
        opts += [f"{k} — {engine.CASE_LABELS[k]}" for k in engine.BINARY_CASES]
        opts += [f"data:{k} — data/ 下的真实图" for k in self.real_imgs]
        if self.file_img is not None:
            opts.insert(0, f"file:{self.file_name} — 本地打开的图片")
        self.case_cb.configure(values=opts)
        if self.cur_case not in [o.split(" — ")[0] for o in opts]:
            self.cur_case = (P.DEFAULT_CASE_FOR_BINARY if self.cur_op in P.BINARY_OPS
                             else P.DEFAULT_CASE_FOR_GRAY)
        label = next((o for o in opts if o.split(" — ")[0] == self.cur_case), opts[0])
        self.case_var.set(label)
        is_synth = self.cur_case in engine.GRAY_CASES or self.cur_case in engine.BINARY_CASES
        self.size_cb.configure(state="readonly" if is_synth else "disabled")

    def _current_image(self) -> np.ndarray:
        if self.cur_case.startswith("file:"):
            return self.file_img
        if self.cur_case.startswith("data:"):
            return self.real_imgs[self.cur_case[5:]]
        return engine.synth_case(self.cur_case, self.cur_size)

    # ---------------------------------------------------------------- 计算
    def _schedule(self):
        if self._sweep_active:      # 扫描期间自己控帧，别被防抖打断
            return
        if self._debounce is not None:
            self.root.after_cancel(self._debounce)
        if not self.auto_var.get():
            return
        self._debounce = self.root.after(DEBOUNCE_MS, self._recompute)

    def _recompute(self):
        if self._debounce is not None:
            self.root.after_cancel(self._debounce)
            self._debounce = None
        try:
            img = self._current_image()
            raw = {k: v.get() for k, v in self.param_vars.items()}
            res = engine.run(self.cur_op, self.cur_variant, raw, img, self.cur_case)
        except Exception as exc:
            self._status_error(f"{type(exc).__name__}: {exc}")
            return
        self.res = res
        views = render.views_for(res)
        self.view_cb.configure(values=views)
        if self.cur_view not in views:
            self.cur_view = views[0]
        self.view_var.set(self.cur_view)
        render.draw(self.fig, res, self.cur_view)
        self._update_status(res)

    # ------------------------------------------------------------ 参数扫描
    def _sweep_param(self):
        label = self.sweep_var.get()
        for q in P.PARAMS.get(self.cur_op, ()):
            if q.label == label:
                return q
        return None

    def _toggle_sweep(self):
        if self._sweep_active:
            self._stop_sweep()
            self.sweep_info_var.set("已停止")
        else:
            self._start_sweep()

    def _start_sweep(self):
        if self.res is None:
            self._recompute()
        p = self._sweep_param()
        if p is None:
            self.sweep_info_var.set("（先选一个要扫的参数）")
            return
        try:
            steps = int(self.sweep_steps_var.get())
        except ValueError:
            steps = 24
        seq = P.sweep_sequence(p, steps)
        if len(seq) < 2:
            self.sweep_info_var.set("（这个参数可扫的取值不足两个）")
            return
        self._sweep_param_def = p
        self._sweep_seq = seq
        self._sweep_i = 0
        self._sweep_rows = []
        self._sweep_empty = 0
        self._sweep_frames = []
        self._sweep_meta = (self.cur_op, p.key)
        try:
            self.sweep_gif_btn.configure(state="disabled")
        except tk.TclError:
            pass
        self._sweep_active = True
        self.sweep_btn.configure(text="停止")
        self._sweep_tick()

    def _sweep_tick(self):
        if not self._sweep_active:
            return
        if self._sweep_i >= len(self._sweep_seq):
            self._finish_sweep()
            return
        p = self._sweep_param_def
        v = self._sweep_seq[self._sweep_i]
        self._set_param_value(p, v)
        try:
            self._recompute()
        except tk.TclError:           # 窗口在扫描中被关掉
            self._stop_sweep()
            return
        name, val = self._headline(self.res)
        if self.res.kind == "iou" and self.res.metrics.get("both_empty"):
            self._sweep_empty += 1
        self._sweep_rows.append((v, val))
        self._sweep_i += 1
        shown = render._fmt(val) if val is not None else "-"
        self.sweep_info_var.set(f"{p.label}={self._fmt_sweep_v(p, v)}　"
                                f"{self._sweep_i}/{len(self._sweep_seq)}　{name} {shown}")
        self._capture_sweep_frame()
        self._sweep_after = self.root.after(FRAME_MS, self._sweep_tick)

    def _capture_sweep_frame(self):
        """把当前画布截成一帧 GIF 素材（缩到 960 宽，控体积也控内存）。"""
        try:
            from PIL import Image
            self.canvas.draw()   # render.draw 走 draw_idle 是异步的，取帧前强制同步渲染一次
            im = Image.fromarray(np.array(self.canvas.buffer_rgba())[:, :, :3])
        except Exception:
            return
        w = 960
        h = max(1, round(im.height * w / im.width))
        self._sweep_frames.append(im.resize((w, h), Image.LANCZOS))
        if len(self._sweep_frames) >= 2:
            try:
                self.sweep_gif_btn.configure(state="normal")
            except tk.TclError:
                pass

    def _save_sweep_gif(self):
        if len(self._sweep_frames) < 2:
            return
        try:
            width = int(self.gif_w_var.get())
            skip = int(self.gif_skip_var.get())
        except ValueError:
            width, skip = 960, 1
        frames, ms = prepare_gif_frames(self._sweep_frames, width, skip)
        if len(frames) < 2:
            self.sweep_info_var.set("抽帧后不足两帧，把抽帧调小一点")
            return
        op = self._sweep_meta[0] if self._sweep_meta else self.cur_op
        path = filedialog.asksaveasfilename(
            defaultextension=".gif", filetypes=[("GIF 动图", "*.gif")],
            initialfile=f"sweep_{op}.gif", title="保存扫描动画")
        if not path:
            return
        n = assemble_gif(frames, path, ms)
        self._status_append(f"扫描动画已存：{path}（{len(frames)} 帧，{width} 宽，"
                            f"{ms} ms/帧，{n / 1024:.0f} KB）\n", "ok")

    def _finish_sweep(self):
        p = self._sweep_param_def
        rows = [(v, m) for v, m in self._sweep_rows if m is not None]
        self._stop_sweep()
        if not rows or p is None:
            self.sweep_info_var.set("扫描完成：没有可用指标")
            return
        name, _ = self._headline(self.res)
        vals = [m for _, m in rows]
        lo_i, hi_i = int(np.argmin(vals)), int(np.argmax(vals))
        summary = (f"参数扫描完成：{p.label} {self._fmt_sweep_v(p, rows[0][0])} → "
                   f"{self._fmt_sweep_v(p, rows[-1][0])}，共 {len(rows)} 帧。"
                   f"{name} 最小 {render._fmt(vals[lo_i])}"
                   f"（{p.label}={self._fmt_sweep_v(p, rows[lo_i][0])}）、"
                   f"最大 {render._fmt(vals[hi_i])}"
                   f"（{p.label}={self._fmt_sweep_v(p, rows[hi_i][0])}）")
        if getattr(self, "_sweep_empty", 0):
            summary += (f"。其中 {self._sweep_empty} 帧两边都无边缘，"
                        f"IoU=1 只表示「都空」，不代表对齐")
        self._status_append(summary + "\n", "warn")
        self.sweep_info_var.set(f"完成 {len(rows)} 帧　{name} "
                                f"{render._fmt(vals[lo_i])} ~ {render._fmt(vals[hi_i])}")

    def _stop_sweep(self):
        self._sweep_active = False
        if self._sweep_after is not None:
            try:
                self.root.after_cancel(self._sweep_after)
            except Exception:
                pass
            self._sweep_after = None
        btn = getattr(self, "sweep_btn", None)
        if btn is not None:
            try:
                btn.configure(text="开始扫描")
            except tk.TclError:
                pass
        gif = getattr(self, "sweep_gif_btn", None)
        if gif is not None and len(self._sweep_frames) >= 2:
            try:
                gif.configure(state="normal")
            except tk.TclError:
                pass

    def _set_param_value(self, p: P.Param, v):
        """把扫描到的值写回界面控件，滑块和输入框跟着动。"""
        var = self.param_vars.get(p.key)
        if var is None:
            return
        if p.kind == "choice":
            var.set(str(v))
            return
        var.set(float(v))
        ent = self.param_entries.get(p.key)
        if ent is not None:
            try:
                ent.delete(0, "end")
                ent.insert(0, self._fmt_param(p, v))
            except tk.TclError:
                pass

    @staticmethod
    def _fmt_sweep_v(p: P.Param, v) -> str:
        return str(v) if p.kind == "choice" else App._fmt_param(p, v)

    @staticmethod
    def _headline(res):
        """这一帧最该盯的那个数：不同 kind 的算子主指标不一样。"""
        if res is None:
            return "指标", None
        m = res.metrics or {}
        k = res.kind
        if k == "iou":
            return "IoU", m.get("iou")
        if k == "tuple3":
            pc = (m.get("per_component") or {}).get("mag") or {}
            return "mag max|err|", pc.get("max_abs")
        if k == "labels":
            try:
                return "个数差", abs(int(m.get("mine_n", 0)) - int(m.get("ref_n", 0)))
            except (TypeError, ValueError):
                return "个数差", None
        if k == "stats":
            return "外接框最大差", m.get("max_box_err")
        return "max|err|", m.get("max_abs")

    # ---------------------------------------------------------------- 显示
    def _on_motion(self, event):
        if event.inaxes is None or event.xdata is None or self.res is None:
            return
        x, y = int(round(event.xdata)), int(round(event.ydata))
        img = self.res.img
        if 0 <= y < img.shape[0] and 0 <= x < img.shape[1]:
            self.coord_var.set(f"x={x} y={y} 输入={int(img[y, x])}")

    def _status_error(self, msg: str):
        t = self.status
        t.configure(state="normal")
        t.delete("1.0", "end")
        t.insert("end", "出错了：\n", "bad")
        t.insert("end", msg + "\n", "bad")
        t.insert("end", "（参数组合不合法时算子会直接抛异常，界面不崩，改参数即可）", "note")
        t.configure(state="disabled")

    def _status_append(self, text: str, tag: str = "info"):
        t = self.status
        t.configure(state="normal")
        t.insert("end", text, tag)
        t.see("end")
        t.configure(state="disabled")

    def _update_status(self, res):
        m = res.metrics
        t = self.status
        t.configure(state="normal")
        t.delete("1.0", "end")

        kind = res.kind
        if kind == "image":
            line = (f"max|err| = {render._fmt(m.get('max_abs'))}　"
                    f"mean|err| = {render._fmt(m.get('mean_abs'))}")
            if m.get("n_over_tol") is not None:
                line += (f"　超容差 {m['n_over_tol']} 像素"
                         f"（{m['pct_over_tol']:.3f}%）")
        elif kind == "tuple3":
            pc = m["per_component"]
            line = "三分量 max|err|：" + "　".join(
                f"{k}={render._fmt(v['max_abs'])}" for k, v in pc.items())
        elif kind == "scalar":
            line = f"myimg t={m['mine']:.0f}　cv2 t={m['ref']:.0f}　差 {render._fmt(m['max_abs'])}"
        elif kind == "counts":
            line = f"bins={len(res.mine)}　max|err| = {render._fmt(m['max_abs'])}"
        elif kind == "labels":
            line = (f"连通域个数 {m['mine_n']} vs {m['ref_n']}　"
                    f"面积多重集合{'一致' if m['areas_match'] else '不一致'}")
        elif kind == "stats":
            line = (f"域数 {m['n_regions']}　外接框最大差 {m['max_box_err']} px　"
                    f"质心最大差 {render._fmt(m['max_centroid_err'])} px")
        else:  # iou
            line = (f"IoU = {m['iou']:.3f}　我的 {m['mine_px']} px / "
                    f"cv2 {m['ref_px']} px"
                    + ("　（两边都空）" if m.get("both_empty") else ""))

        tag = "ok" if res.verdict in _OK else ("warn" if res.verdict in _WARN else "bad")
        t.insert("end", f"{line}\n判定：{res.verdict}　（口径 = 报告里的 {res.spec_op}）\n", tag)

        slow = res.slowdown
        t.insert("end",
                 f"耗时：myimg {res.t_mine:.2f} ms / OpenCV {res.t_ref:.2f} ms"
                 f"（{res.repeats} 次取最小 @ {res.img.shape[0]}×{res.img.shape[1]}）　"
                 f"慢 {slow:.1f}×\n", "info")
        t.insert("end", f"口径：{res.note}\n", "note")
        for w in res.warnings:
            t.insert("end", f"⚠ {w}\n", "warn")
        t.configure(state="disabled")

    def _export(self):
        if self.res is None:
            return
        default = f"myimg_{self.cur_op}_{self.cur_variant}_{self.cur_case.replace(':', '_')}.png"
        path = filedialog.asksaveasfilename(defaultextension=".png",
                                            initialfile=default,
                                            filetypes=[("PNG", "*.png")])
        if path:
            render.save_png(self.fig, path, dpi=150)


def main():
    _dpi_aware()
    root = tk.Tk()
    App(root)          # 主题在 App.__init__ 里装，单独建窗口也不会漏
    root.mainloop()


if __name__ == "__main__":
    main()
