"""画布渲染：把 engine 的结果画成 2×2 四格。

布局约定：**左上永远是输入**，其余三格按算子的输出类型变。
    image / tuple3 / iou  -> 输入 | myimg | OpenCV | 信息格
    scalar  (otsu)        -> 输入 | 我的分割 | cv2 分割 | 直方图 + 两条阈值线
    counts  (histogram)   -> 输入 | 我的柱状 | cv2 柱状 | 差值柱状 / CDF
    labels                -> 输入 | 我的染色 | cv2 染色 | 面积分布对比
    stats                 -> 输入 | 我的外接框 | cv2 外接框 | 面积柱状 + 质心误差

第四格（右下）是整个界面的重点：前三格看效果，第四格看「差在哪、差多少」。
编号顺序、颜色、值域都在这一层统一处理，app.py 只管布局和事件。
"""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

# Windows 上是微软雅黑；找不到就退回 DejaVu（中文会变方框，但至少不崩）
FONT_CANDIDATES = ["Microsoft YaHei", "SimHei", "PingFang SC", "Noto Sans CJK SC",
                   "WenQuanYi Zen Hei", "DejaVu Sans"]

_C_EDGE_MINE = (1.0, 0.25, 0.25)     # 只有 myimg 检出 -> 红
_C_EDGE_REF = (0.2, 0.75, 1.0)       # 只有 OpenCV 检出 -> 青
_C_EDGE_BOTH = (1.0, 1.0, 1.0)       # 两边都有 -> 白
_C_MINE = "#d62728"
_C_REF = "#1f77b4"


def setup_style() -> None:
    """中文字体 + 全局字号。必须在建 Figure 之前调一次。"""
    from matplotlib import rcParams

    rcParams["font.sans-serif"] = FONT_CANDIDATES
    rcParams["axes.unicode_minus"] = False   # 负号用 ASCII，否则也是方框
    rcParams["font.size"] = 9
    rcParams["axes.titlesize"] = 9.5
    rcParams["image.interpolation"] = "nearest"


def make_figure(dpi: int = 100) -> Figure:
    fig = Figure(figsize=(9.4, 7.4), dpi=dpi)
    # constrained 比 tight 稳：标题两行 + 图例 + log 轴同时出现时 tight 会直接放弃排版
    fig.set_layout_engine("constrained")
    return fig


# ---------------------------------------------------------------------------
# 每个算子能看什么
# ---------------------------------------------------------------------------
def views_for(res) -> list[str]:
    """当前结果支持的信息格 / 显示模式列表，第一个是默认。"""
    if res.kind == "tuple3":
        return ["mag（梯度幅值）", "gx（水平梯度）", "gy（垂直梯度）"]
    if res.kind == "iou":
        return ["边缘叠加（红=仅我的，青=仅 cv2）", "差值热图"]
    if res.kind == "scalar":
        return ["直方图 + 阈值线", "差值柱状"]
    if res.kind == "counts":
        return ["差值柱状", "累积分布 CDF"]
    if res.kind == "labels":
        return ["面积分布对比", "编号差值（仅供参考）"]
    if res.kind == "stats":
        return ["面积柱状 + 质心误差"]
    # image：直方图类算子多看一眼分布，其余看差值热图
    if res.op in ("hist_equalize", "clahe"):
        return ["差值热图", "直方图对比"]
    return ["差值热图"]


def default_view(res) -> str:
    return views_for(res)[0]


# ---------------------------------------------------------------------------
# 基础绘制
# ---------------------------------------------------------------------------
def _clean(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.6)
        s.set_color("#888888")


def _show(ax, arr, title, fixed: bool = True, cmap: str = "gray"):
    """fixed=True 时锁定 0..255，三格之间才可比；否则自动缩放并在标题里写清楚。"""
    a = np.asarray(arr, dtype=np.float64)
    if fixed:
        im = ax.imshow(a, cmap=cmap, vmin=0, vmax=255)
    else:
        im = ax.imshow(a, cmap=cmap)
        lo, hi = float(a.min()), float(a.max())
        title = f"{title}\n（自动缩放 {lo:.1f}~{hi:.1f}）"
    ax.set_title(title)
    _clean(ax)
    return im


def _text_box(ax, lines: list[str]):
    """在格子中央写几行说明文字（用于没有图像可画的场合）。"""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.5, 0.5, "\n".join(lines), transform=ax.transAxes,
            ha="center", va="center", fontsize=9.5, family="monospace")


def _fmt(v) -> str:
    if v is None:
        return "-"
    v = float(v)
    if np.isnan(v):
        return "NaN"
    if v == 0:
        return "0"
    if abs(v) >= 1e4 or abs(v) < 1e-3:
        return f"{v:.2e}"
    return f"{v:.4f}".rstrip("0").rstrip(".")


# ---------------------------------------------------------------------------
# 各视图
# ---------------------------------------------------------------------------
def _mine_ref_arrays(res):
    """把 mine / ref 整理成能直接 imshow 的 2D 数组（tuple3 按分量取）。"""
    if res.kind == "tuple3":
        return np.asarray(res.mine), np.asarray(res.ref)   # (3, H, W)
    if res.kind == "labels":
        return res.mine[0], res.ref[0]
    return np.asarray(res.mine), np.asarray(res.ref)


def _diff_view(ax, res, comp: str = "mag"):
    a, b = _mine_ref_arrays(res)
    if res.kind == "tuple3":
        idx = {"gx": 0, "gy": 1, "mag": 2}[comp]
        a, b = a[idx], b[idx]
    elif a.ndim == 3:
        a, b = a[0], b[0]
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    mx = float(d.max())
    if mx == 0:
        ax.imshow(np.zeros_like(d), cmap="gray", vmin=0, vmax=1)
        ax.text(0.5, 0.5, "完全一致\nmax|err| = 0", transform=ax.transAxes,
                ha="center", va="center", color="white", fontsize=11)
        ax.set_title("差值 |myimg − OpenCV|")
    else:
        ax.imshow(d, cmap="magma")
        over = int((d > 1e-8).sum())
        ax.set_title(f"差值 |myimg − OpenCV|　max={_fmt(mx)}\n"
                     f"非零像素 {over} 个（{100.0 * over / d.size:.3f}%）")
    _clean(ax)


def _overlay_view(ax, res):
    """边缘叠加：白=两边都检出，红=只有我的，青=只有 cv2。

    比差值热图有信息量：能一眼看出偏移方向（比如整条边往右挪一列，
    就会看到左红右青的双色边）。
    """
    a = np.asarray(res.mine) > 0
    b = np.asarray(res.ref) > 0
    rgb = np.zeros(a.shape + (3,), dtype=np.float64)
    only_m = a & ~b
    only_r = b & ~a
    both = a & b
    rgb[only_m] = _C_EDGE_MINE
    rgb[only_r] = _C_EDGE_REF
    rgb[both] = _C_EDGE_BOTH
    ax.imshow(rgb)
    m = res.metrics
    ax.set_title(f"边缘叠加　IoU={m.get('iou', float('nan')):.3f}\n"
                 f"我的 {m.get('mine_px', 0)} px / cv2 {m.get('ref_px', 0)} px / "
                 f"重合 {int((both.sum()))} px")
    _clean(ax)


def _hist_bars(ax, h_mine, h_ref, title, log: bool = False):
    x = np.arange(len(h_mine))
    ax.bar(x, h_mine, width=0.9, color=_C_MINE, alpha=0.75, label="myimg")
    if h_ref is not None:
        ax.bar(x, h_ref, width=0.45, color=_C_REF, alpha=0.9, label="OpenCV")
    if log:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel("灰度 / 分箱")
    ax.set_ylabel("像素数")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.tick_params(labelsize=8)


def _hist_view(ax, res, diff: bool = False):
    if res.kind == "counts":
        hm = np.asarray(res.mine, dtype=np.float64)
        hr = np.asarray(res.ref, dtype=np.float64)
        if diff:
            d = hm - hr
            ax.bar(np.arange(len(d)), d, width=0.9, color="#7f7f7f")
            ax.axhline(0, color="k", linewidth=0.6)
            ax.set_title(f"计数差值 myimg − cv2　max|err|={_fmt(res.metrics['max_abs'])}")
            ax.set_xlabel("灰度 / 分箱")
            ax.set_ylabel("差值")
        else:
            _hist_bars(ax, hm, hr, f"直方图对比（bins={len(hm)}）", log=True)
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)
        ax.tick_params(labelsize=8)
        return
    # image 类（hist_equalize / clahe）：比输入输出各自的分布
    src = np.asarray(res.img).ravel()
    a = np.clip(np.asarray(res.mine, dtype=np.float64).ravel(), 0, 255)
    b = np.clip(np.asarray(res.ref, dtype=np.float64).ravel(), 0, 255)
    bins = np.arange(257) - 0.5
    ax.hist(src, bins=bins, histtype="step", color="#7f7f7f", linewidth=0.8, label="输入")
    ax.hist(a, bins=bins, histtype="step", color=_C_MINE, linewidth=1.1, label="myimg")
    ax.hist(b, bins=bins, histtype="step", color=_C_REF, linewidth=1.1, label="OpenCV")
    ax.set_yscale("log")
    ax.set_title(f"{res.op} 的灰度分布")
    ax.set_xlabel("灰度")
    ax.set_ylabel("像素数（log）")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.tick_params(labelsize=8)


def _cdf_view(ax, res):
    hm = np.asarray(res.mine, dtype=np.float64)
    hr = np.asarray(res.ref, dtype=np.float64)
    cm = np.cumsum(hm) / hm.sum()
    cr = np.cumsum(hr) / hr.sum()
    ax.plot(cm, color=_C_MINE, linewidth=1.2, label="myimg CDF")
    ax.plot(cr, color=_C_REF, linewidth=1.0, linestyle="--", label="cv2 CDF")
    ax.set_title("累积分布（直方图均衡化就是拿它当映射表）")
    ax.set_xlabel("灰度")
    ax.set_ylabel("累积占比")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.tick_params(labelsize=8)


def _otsu_view(ax, res, diff: bool = False):
    img = np.asarray(res.img)
    h = np.bincount(img.ravel(), minlength=256).astype(np.float64)
    tm, tr = float(res.mine), float(res.ref)
    if diff:
        ax.bar(np.arange(256), h, width=1.0, color="#bbbbbb")
        ax.set_title(f"输入直方图　mine={tm:.0f} / cv2={tr:.0f} / 差={_fmt(res.metrics['max_abs'])}")
    else:
        ax.bar(np.arange(256), h, width=1.0, color="#bbbbbb", label="输入直方图")
        ax.axvline(tm, color=_C_MINE, linewidth=1.6, label=f"myimg t={tm:.0f}")
        if tr != tm:
            ax.axvline(tr, color=_C_REF, linewidth=1.4, linestyle="--", label=f"cv2 t={tr:.0f}")
        else:
            ax.axvline(tr, color=_C_REF, linewidth=0.8, linestyle=":", label=f"cv2 t={tr:.0f}（重合）")
        ax.set_title("Otsu：类间方差最大的那个 t")
        ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.set_xlabel("灰度")
    ax.set_ylabel("像素数")
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.tick_params(labelsize=8)


def _label_rgb(labels: np.ndarray) -> np.ndarray:
    """标签图染色。背景透明，用固定 colormap 保证同一个编号颜色一致。"""
    from matplotlib import colormaps

    n = int(labels.max())
    if n <= 0:
        return np.zeros(labels.shape + (4,), dtype=np.float64)
    cmap = colormaps["tab20"].resampled(max(20, n + 1))
    idx = np.clip(labels, 0, None) % 20
    rgb = cmap(idx)[:, :, :3]
    alpha = (labels > 0).astype(np.float64)[:, :, None]
    return np.concatenate([rgb, alpha], axis=2)


def _labels_view(ax, res, which: str):
    lab = res.mine[0] if which == "mine" else res.ref[0]
    ax.imshow(_label_rgb(np.asarray(lab)))
    n_mine, n_ref = res.metrics["mine_n"], res.metrics["ref_n"]
    who = "myimg" if which == "mine" else "OpenCV"
    ax.set_title(f"{who} 的连通域染色　n={n_mine if which == 'mine' else n_ref}\n"
                 f"（编号顺序两家不同，颜色对不上是正常的）")
    _clean(ax)


def _area_bars(ax, res):
    """面积分布对比：两边都按面积降序排，逐条比。

    连通域不比编号（编号顺序取决于扫描次序，本来就不保证一致），
    比的是「个数 + 面积多重集合」，这也是 compare_one 里的判定口径。
    """
    m_lab = np.asarray(res.mine[0])
    r_lab = np.asarray(res.ref[0])
    am = np.sort(np.bincount(m_lab.ravel())[1:])[::-1]
    ar = np.sort(np.bincount(r_lab.ravel())[1:])[::-1]
    n = max(len(am), len(ar))
    x = np.arange(n)
    ax.bar(x - 0.2, np.pad(am, (0, n - len(am))), width=0.4, color=_C_MINE, label="myimg")
    ax.bar(x + 0.2, np.pad(ar, (0, n - len(ar))), width=0.4, color=_C_REF, label="OpenCV")
    ax.set_yscale("log")
    ok = res.metrics.get("areas_match")
    ax.set_title(f"面积降序对比　个数 {res.metrics['mine_n']} vs {res.metrics['ref_n']}，"
                 f"面积多重集合{'一致' if ok else '不一致'}")
    ax.set_xlabel("第几个域（按面积降序）")
    ax.set_ylabel("面积 px（log）")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.tick_params(labelsize=8)


def _stats_panel(ax, img, stats, title, color):
    """在 mask 上画外接框和质心。"""
    ax.imshow(np.asarray(img), cmap="gray", vmin=0, vmax=255)
    for d in stats[:60]:
        ax.add_patch(Rectangle((d["x"] - 0.5, d["y"] - 0.5), d["w"], d["h"],
                               fill=False, edgecolor=color, linewidth=0.9))
        ax.plot(d["cx"], d["cy"], marker="+", color=color, markersize=4, mew=0.9)
    ax.set_title(title)
    _clean(ax)


def _stats_compare(ax, res):
    mine, ref = res.mine, res.ref
    n = max(len(mine), len(ref))
    x = np.arange(n)
    am = [d["area"] for d in mine] + [0] * (n - len(mine))
    ar = [d["area"] for d in ref] + [0] * (n - len(ref))
    ax.bar(x - 0.2, am, width=0.4, color=_C_MINE, label="myimg")
    ax.bar(x + 0.2, ar, width=0.4, color=_C_REF, label="OpenCV")
    ax.set_yscale("symlog")
    m = res.metrics
    ax.set_title(f"域面积对比（按面积降序）　n={m['n_regions']}\n"
                 f"外接框最大差 {m['max_box_err']} px，质心最大差 "
                 f"{_fmt(m['max_centroid_err'])} px")
    ax.set_xlabel("第几个域")
    ax.set_ylabel("面积 px")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.tick_params(labelsize=8)


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def draw(fig: Figure, res, view: str | None = None) -> None:
    """重画整张 2×2。view 为空就用该算子的默认视图。"""
    view = view or default_view(res)
    fig.clear()
    axs = fig.subplots(2, 2)
    (a00, a01), (a10, a11) = axs

    case = res.case or "未命名输入"
    _show(a00, res.img, f"输入：{case}　{res.img.shape[0]}×{res.img.shape[1]} uint8")

    kind = res.kind
    if kind == "scalar":                       # otsu_threshold
        tm = float(res.mine)
        _show(a01, (np.asarray(res.img) > tm) * 255.0, f"myimg 分割（t={tm:.0f}）")
        _show(a10, (np.asarray(res.img) > float(res.ref)) * 255.0,
              f"OpenCV 分割（t={float(res.ref):.0f}）")
        _otsu_view(a11, res, diff=view.startswith("差值"))
    elif kind == "counts":                     # histogram
        _hist_bars(a01, np.asarray(res.mine), None, f"myimg 直方图（bins={len(res.mine)}）", log=True)
        _hist_bars(a10, np.asarray(res.ref), None, f"OpenCV 直方图（bins={len(res.ref)}）", log=True)
        if view.startswith("累积"):
            _cdf_view(a11, res)
        else:
            _hist_view(a11, res, diff=True)
    elif kind == "labels":
        _labels_view(a01, res, "mine")
        _labels_view(a10, res, "ref")
        if view.startswith("编号"):
            _diff_view(a11, res)
        else:
            _area_bars(a11, res)
    elif kind == "stats":
        _stats_panel(a01, res.img, res.mine, f"myimg 外接框（{len(res.mine)} 个域）", _C_MINE)
        _stats_panel(a10, res.img, res.ref, f"OpenCV 外接框（{len(res.ref)} 个域）", _C_REF)
        _stats_compare(a11, res)
    elif kind == "tuple3":                     # sobel：view 就是分量名
        comp = view[:2].lower()
        if comp not in ("gx", "gy", "ma"):
            comp = "mag"
        comp = "mag" if comp == "ma" else comp
        idx = {"gx": 0, "gy": 1, "mag": 2}[comp]
        _show(a01, res.mine[idx], f"myimg {comp}", fixed=False)
        _show(a10, res.ref[idx], f"OpenCV {comp}", fixed=False)
        _diff_view(a11, res, comp=comp)
    else:                                      # image / iou
        if kind == "iou":
            _show(a01, np.asarray(res.mine), f"myimg {res.op}")
            _show(a10, np.asarray(res.ref), f"OpenCV {res.spec_op}")
            if view.startswith("边缘叠加"):
                _overlay_view(a11, res)
            else:
                _diff_view(a11, res)
        else:
            _show(a01, np.asarray(res.mine), f"myimg {res.op}（{res.variant_label}）")
            _show(a10, np.asarray(res.ref), "OpenCV（基准）")
            if view.startswith("直方图"):
                _hist_view(a11, res)
            else:
                _diff_view(a11, res)

    fig.canvas.draw_idle()


def save_png(fig: Figure, path: str, dpi: int = 150) -> str:
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    return path
