"""GUI 主窗口：Tkinter + matplotlib 嵌入。

双击就能跑，不需要起服务器、不需要额外依赖（tkinter 是标准库，matplotlib 已在 dev 依赖里）。
界面分四块：
    顶部工具条   用例 / 尺寸 / 打开文件 / 运行 / 自动刷新 / 导出
    左侧面板     算子树 + 动态参数 + 实现变体 + 信息格视图
    右侧画布     2×2 四格（渲染逻辑全在 render.py）
    底部状态区   误差 / 耗时 / 口径说明 / 坑提示

事件流：任何控件变化 -> _schedule() 防抖 150ms -> _recompute() ->
engine.run() -> render.draw() -> 状态区刷新。自动刷新关掉时只更新参数显示，
等你点「运行」。
"""

from __future__ import annotations

import pathlib
import sys
import tkinter as tk
from tkinter import filedialog, ttk

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gui import engine, params as P, render   # noqa: E402

SIZES = (64, 96, 128, 192, 256, 384, 512)
DEBOUNCE_MS = 150

_OK = ("逐像素精确", "一致", "浮点噪声级一致", "IoU 合格")
_WARN = ("仅记录", "两边都无边缘")
_BAD = ("超出容差", "不一致", "IoU 偏低")


def _dpi_aware() -> None:
    """Windows 下让 tkinter 用真实 DPI，否则高分屏上整窗发虚。"""
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("myimg 图像算子对拍台 —— 手写 NumPy vs OpenCV")
        root.geometry("1420x900")
        root.minsize(1120, 720)

        render.setup_style()

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
        self._debounce: str | None = None

        self.auto_var = tk.BooleanVar(value=True)
        self.variant_var = tk.StringVar(value="vector")
        self.case_var = tk.StringVar()
        self.size_var = tk.StringVar(value="128")
        self.view_var = tk.StringVar()
        self.coord_var = tk.StringVar(value="")

        self._build_toolbar()
        self._build_left()
        self._build_canvas()
        self._build_status()

        self._refresh_case_options()
        self._select_op(self.cur_op)

    # ---------------------------------------------------------------- 布局
    def _build_toolbar(self):
        bar = ttk.Frame(self.root, padding=(8, 6, 8, 2))
        bar.grid(row=0, column=0, columnspan=2, sticky="ew")

        ttk.Label(bar, text="用例").grid(row=0, column=0, padx=(0, 4))
        self.case_cb = ttk.Combobox(bar, textvariable=self.case_var,
                                    state="readonly", width=34)
        self.case_cb.grid(row=0, column=1, padx=(0, 10))
        self.case_cb.bind("<<ComboboxSelected>>", lambda e: self._on_case_change())

        ttk.Label(bar, text="尺寸").grid(row=0, column=2, padx=(0, 4))
        self.size_cb = ttk.Combobox(bar, textvariable=self.size_var,
                                    state="readonly", width=6, values=SIZES)
        self.size_cb.grid(row=0, column=3, padx=(0, 10))
        self.size_cb.bind("<<ComboboxSelected>>", lambda e: self._on_size_change())

        ttk.Button(bar, text="打开图片…", width=10,
                   command=self._open_file).grid(row=0, column=4, padx=(0, 6))
        self.run_btn = ttk.Button(bar, text="运行对比", width=10,
                                  command=self._recompute)
        self.run_btn.grid(row=0, column=5, padx=(0, 6))
        ttk.Checkbutton(bar, text="自动刷新", variable=self.auto_var,
                        command=self._schedule).grid(row=0, column=6, padx=(0, 6))
        ttk.Button(bar, text="导出 PNG…", width=10,
                   command=self._export).grid(row=0, column=7)

        ttk.Label(bar, textvariable=self.coord_var, foreground="#666666",
                  font=("Consolas", 9)).grid(row=0, column=8, padx=(14, 0))

    def _build_left(self):
        left = ttk.Frame(self.root, padding=(8, 2, 4, 6))
        left.grid(row=1, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=0, minsize=330)

        self.tree = ttk.Treeview(left, show="tree", height=10, selectmode="browse")
        self.tree.grid(row=0, column=0, sticky="ew")
        self._op_nodes: dict[str, str] = {}
        for gid, (gname, ops) in enumerate(P.OP_GROUPS):
            gnode = self.tree.insert("", "end", text=gname, open=True)
            for op in ops:
                node = self.tree.insert(gnode, "end", text=op)
                self._op_nodes[op] = node
        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)

        pf = ttk.LabelFrame(left, text="参数", padding=6)
        pf.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        self.param_frame = pf

        vf = ttk.LabelFrame(left, text="实现变体", padding=6)
        vf.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        self.variant_frame = vf

        ttk.Label(left, text="右下信息格").grid(row=3, column=0, sticky="w", pady=(8, 2))
        self.view_cb = ttk.Combobox(left, textvariable=self.view_var,
                                    state="readonly")
        self.view_cb.grid(row=4, column=0, sticky="ew")
        self.view_cb.bind("<<ComboboxSelected>>", lambda e: self._on_view_change())

        left.columnconfigure(0, weight=1)

    def _build_canvas(self):
        from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                                       NavigationToolbar2Tk)

        wrap = ttk.Frame(self.root, padding=(4, 2, 8, 6))
        wrap.grid(row=1, column=1, sticky="nsew")
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(1, weight=1)
        wrap.rowconfigure(0, weight=1)
        wrap.columnconfigure(0, weight=1)

        self.fig = render.make_figure(dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=wrap)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        # NavigationToolbar2Tk 内部用 pack，不能直接塞进 grid 管理的容器，
        # 给它一个专属 Frame 隔离两种几何管理器
        toolframe = ttk.Frame(wrap)
        toolframe.grid(row=1, column=0, sticky="ew")
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolframe)
        self.toolbar.update()
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        # 高分屏首帧的坑：canvas 在 <Map> 里把 device pixel ratio 从 1 提到 2
        # （figure.dpi 100 -> 200），随后靠 configure(widget) 触发 <Configure>
        # 重算 figure 英寸数。但 grid 布局下分配尺寸不变、<Configure> 不来，
        # figure 英寸数就停在旧 dpi 算的值 -> Agg 缓冲是 photo 的 4 倍大，
        # 屏幕上只剩左上 1/4（正好是输入格）。映射完成后用 matplotlib 自己的
        # resize() 对齐一次即可；"+" 是追加绑定，不覆盖 backend 的 <Map>。
        self.canvas.get_tk_widget().bind("<Map>", self._sync_canvas_size, "+")

    def _sync_canvas_size(self, event=None):
        """映射完成后把 figure 英寸数 / photo 尺寸对齐到 widget 实际尺寸。"""
        import types

        def _do():
            w = self.canvas.get_tk_widget().winfo_width()
            h = self.canvas.get_tk_widget().winfo_height()
            if w < 80 or h < 80:
                return
            self.canvas.resize(types.SimpleNamespace(width=w, height=h))
        self.root.after_idle(_do)

    def _build_status(self):
        self.status = tk.Text(self.root, height=7, wrap="word", relief="flat",
                              background="#f5f5f5", font=("Consolas", 10),
                              state="disabled", padx=8, pady=6)
        self.status.grid(row=2, column=0, columnspan=2, sticky="ew")
        self.status.tag_configure("ok", foreground="#1a7f37")
        self.status.tag_configure("warn", foreground="#9a6700")
        self.status.tag_configure("bad", foreground="#cf222e")
        self.status.tag_configure("info", foreground="#444444")
        self.status.tag_configure("note", foreground="#666666")

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
        params = P.PARAMS.get(op, ())
        if not params:
            ttk.Label(self.param_frame, text="（这个算子没有可调参数）",
                      foreground="#888888").grid(row=0, column=0, sticky="w")
            return
        for row, p in enumerate(params):
            ttk.Label(self.param_frame, text=p.label).grid(row=row, column=0,
                                                           sticky="w", pady=2)
            if p.kind == "choice":
                var = tk.StringVar(value=str(p.default))
                cb = ttk.Combobox(self.param_frame, textvariable=var,
                                  state="readonly", width=12, values=list(p.options))
                cb.grid(row=row, column=1, sticky="w", pady=2)
                var.trace_add("write", lambda *a: self._schedule())
                self.param_vars[p.key] = var
            else:
                var = tk.DoubleVar(value=float(p.default))
                sc = ttk.Scale(self.param_frame, from_=p.lo, to=p.hi,
                               variable=var, orient="horizontal")
                sc.grid(row=row, column=1, sticky="we", pady=2)
                ent = ttk.Entry(self.param_frame, width=7,
                                font=("Consolas", 9))
                ent.grid(row=row, column=2, padx=(6, 0), pady=2)
                ent.insert(0, self._fmt_param(p, p.default))
                sc.configure(command=lambda v, p=p, e=ent: self._on_scale(p, v, e))
                ent.bind("<Return>", lambda e, p=p, s=sc, en=ent: self._on_entry(p, s, en))
                ent.bind("<FocusOut>", lambda e, p=p, s=sc, en=ent: self._on_entry(p, s, en))
                self.param_vars[p.key] = var
        self.param_frame.columnconfigure(1, weight=1)

    def _build_variants(self, op: str):
        for w in self.variant_frame.winfo_children():
            w.destroy()
        opts = P.variants(op)
        if self.cur_variant not in [k for k, _ in opts]:
            self.cur_variant = opts[0][0]
        self.variant_var.set(self.cur_variant)
        for i, (key, label) in enumerate(opts):
            ttk.Radiobutton(self.variant_frame, text=label, value=key,
                            variable=self.variant_var,
                            command=self._on_variant_change).grid(row=i, column=0, sticky="w")
        if len(opts) == 1:
            ttk.Label(self.variant_frame, text="（只有这一种实现）",
                      foreground="#888888").grid(row=0, column=1, sticky="w", padx=(8, 0))

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
        self.cur_variant = self.variant_var.get()
        self._schedule()

    def _on_view_change(self):
        self.cur_view = self.view_var.get()
        if self.res is not None:
            render.draw(self.fig, self.res, self.cur_view)

    def _on_case_change(self):
        label = self.case_var.get()
        self.cur_case = label.split(" — ")[0]
        self._schedule()

    def _on_size_change(self):
        self.cur_size = int(self.size_var.get())
        self._schedule()

    def _open_file(self):
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
    style = ttk.Style(root)
    if "vista" in style.theme_names():
        style.theme_use("vista")
    elif "clam" in style.theme_names():
        style.theme_use("clam")
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
