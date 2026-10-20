"""myimg 对拍台 · web 版（Streamlit）。

跑法
    streamlit run web/app.py        # 默认开在 http://localhost:8501

和桌面版（python -m gui）跑的是同一个计算核：engine.run() 出结果、
render.draw() 画四格、render.summary_line() 报误差、engine.headline() 定主指标，
容差与判定口径全部来自 benchmarks/run_compare.py 的 SPECS。所以两个界面看到的
数字必然一致——web 版只是把 tkinter 的控件换成了侧栏，没有第二套算法。

为什么不直接 import gui.app：那条链会把 tkinter 拖进来，没有显示环境的机器
（服务器 / WSL / CI）一 import 就崩。共用的东西因此都抽成了纯模块：
    gui/theme.py   色值 + 判定语义色
    gui/anim.py    GIF 导出（支持写到 BytesIO，web 直接塞给下载按钮）
    gui/render.py  四格绘制 + 状态区第一行
    gui/engine.py  对拍与指标（headline 也在这）

本文件模块级只有 import 和纯函数：sweep_rows / sweep_summary / image_for /
fmt_value 都能被 tests 直接调，不需要起服务器。所有 st.* 调用都在 main()
及它调用的 _ 开头函数里；streamlit run 会把本文件当 __main__ 执行。
"""

from __future__ import annotations

import io
import pathlib
import sys
import tempfile

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib                                       # noqa: E402

matplotlib.use("Agg")     # 服务器上没有显示后端，必须在 render / pyplot 之前定

import streamlit as st                                  # noqa: E402

from gui import engine, params as P, render             # noqa: E402
from gui.anim import GIF_SKIPS, GIF_WIDTHS              # noqa: E402
from gui.anim import (assemble_gif, grab_frame,   # noqa: E402,F401
                      prepare_gif_frames)
from gui.theme import THEME, verdict_tone               # noqa: E402

SIZES = (64, 96, 128, 192, 256, 384, 512)
SWEEP_STEPS = P.SWEEP_STEPS
GIF_FRAME_DPI = 100          # 抓帧用的 dpi：940px 宽，够缩到 960 以内
MONO = "'Cascadia Mono', 'JetBrains Mono', Consolas, monospace"
CN = ("'HarmonyOS Sans SC', 'Microsoft YaHei UI', 'Microsoft YaHei', "
      "'PingFang SC', 'Noto Sans CJK SC', sans-serif")


# ---------------------------------------------------------------------------
# 纯函数：tests 直接调，不碰 streamlit
# ---------------------------------------------------------------------------
def image_for(case: str, size: int, real: dict | None = None) -> np.ndarray:
    """取输入图：data/ 里的真实图优先，其余走合成用例（固定种子，可复现）。"""
    if real and case in real:
        return real[case]
    return engine.synth_case(case, size)


def fmt_value(p: P.Param, v) -> str:
    """扫描轴上的刻度文字，口径和桌面版 App._fmt_param 一致。"""
    if p.kind == "choice":
        return str(v)
    if p.kind == "float":
        return f"{float(v):.{p.digits}f}"
    return str(int(round(float(v))))


def sweep_rows(op: str, variant: str, case: str, size: int, param_key: str,
               steps: int = 12, img: np.ndarray | None = None,
               on_frame=None) -> list[dict]:
    """把一个参数在取值区间上扫一遍，逐点收主指标。

    一行一个 dict：value / label / metric_name / metric / verdict / both_empty /
    t_mine / t_ref / slowdown。参数不属于这个算子就返回空表（界面不该出现，
    但也别崩）。

    on_frame(i, n, value, res) 是可选回调：web 界面靠它刷进度条并抓帧，
    这样扫描循环只有一份，tests 不传回调就是纯计算。res 为 None 表示这一档
    算子直接抛了异常（回调该跳过的就是帧，进度条照样要往前走）。

    repeats 固定 1：这里看的是指标随参数怎么变，计时不是重点，
    多跑两遍只会让 24 帧的扫描慢三倍。
    """
    prm = {q.key: q for q in P.PARAMS.get(op, ())}.get(param_key)
    if prm is None:
        return []
    arr = image_for(case, size) if img is None else img
    raw = dict(P.defaults(op))
    vals = P.sweep_sequence(prm, steps)
    rows: list[dict] = []
    name_hint = "指标"          # 第一档就失败时还没有主指标名，先用占位
    nan = float("nan")
    for i, v in enumerate(vals):
        raw[param_key] = v
        try:
            res = engine.run(op, variant, raw, arr, case=case, repeats=1)
        except Exception as exc:
            # 有些取值基准本来就不接受（sobel 的 mode='wrap'：cv2.Sobel 不收
            # BORDER_WRAP）。记一行出错、跳过，绝不能拿上一档的结果顶替——
            # 那样折线上会多出一个看着像真的点。
            rows.append({"value": v, "label": fmt_value(prm, v),
                         "metric_name": name_hint, "metric": None,
                         "verdict": "运行出错", "both_empty": False,
                         "t_mine": nan, "t_ref": nan, "slowdown": nan,
                         "error": f"{type(exc).__name__}: {exc}"})
            if on_frame is not None:
                on_frame(i, len(vals), v, None)
            continue
        name, val = engine.headline(res)
        name_hint = name
        rows.append({
            "value": v,
            "label": fmt_value(prm, v),
            "metric_name": name,
            "metric": None if val is None else float(val),
            "verdict": res.verdict,
            "both_empty": bool(res.metrics.get("both_empty")),
            "t_mine": res.t_mine,
            "t_ref": res.t_ref,
            "slowdown": res.slowdown,
            "error": "",
        })
        if on_frame is not None:
            on_frame(i, len(vals), v, res)
    return rows


def sweep_summary(rows: list[dict], prm: P.Param) -> str:
    """扫完那句话：主指标 min~max 落在哪个取值上，和桌面版同一口径。"""
    got = [r for r in rows if r["metric"] is not None]
    if not got:
        return "扫描完成：没有可用指标"
    name = got[0]["metric_name"]
    vals = [r["metric"] for r in got]
    lo_i, hi_i = int(np.argmin(vals)), int(np.argmax(vals))
    s = (f"参数扫描完成：{prm.label} {got[0]['label']} → {got[-1]['label']}，"
         f"共 {len(got)} 帧。{name} 最小 {render._fmt(vals[lo_i])}"
         f"（{prm.label}={got[lo_i]['label']}）、"
         f"最大 {render._fmt(vals[hi_i])}（{prm.label}={got[hi_i]['label']}）")
    n_empty = sum(1 for r in got if r["both_empty"])
    if n_empty:
        s += (f"。其中 {n_empty} 帧两边都无边缘，IoU=1 只表示「都空」，"
              f"不代表对齐")
    n_err = sum(1 for r in rows if r["error"])
    if n_err:
        s += f"。另有 {n_err} 个取值算子直接抛异常（基准不收这个口径），已跳过"
    return s



# ---------------------------------------------------------------------------
# 界面
# ---------------------------------------------------------------------------
def _tag(text: str, color: str, size: str = "13px",
         family: str | None = None, weight: str = "400") -> str:
    """一小段带色文字。HTML 属性一律双引号、Python 串一律单引号，省得转义。"""
    fam = family or CN
    return (f'<span style="font-family:{fam};font-size:{size};'
            f'color:{color};font-weight:{weight}">{text}</span>')


def _brand() -> None:
    gold, muted, faint = THEME["gold"], THEME["muted"], THEME["faint"]
    st.markdown(
        '<div style="display:flex;align-items:baseline;gap:12px;flex-wrap:wrap">'
        + _tag("myimg", gold, "20px", MONO, "600")
        + _tag("图像算子对拍台 · web", muted, "13px")
        + _tag("纯 NumPy 手写 vs OpenCV / scikit-image", faint, "11px", MONO)
        + '</div>'
        + '<div style="height:1px;margin:10px 0 18px;background:'
        f'linear-gradient(90deg,{gold},transparent)"></div>',
        unsafe_allow_html=True)


def _param_widget(op: str, prm: P.Param):
    """一个参数控件。key 里带上 op，换算子时不会撞上另一个算子的同名参数。"""
    key = f"p::{op}::{prm.key}"
    hint = prm.hint or None
    if prm.kind == "choice":
        opts = list(prm.options)
        d = str(prm.default)
        return st.selectbox(prm.label, opts, key=key, help=hint,
                            index=opts.index(d) if d in opts else 0)
    if prm.kind == "float":
        return st.slider(prm.label, float(prm.lo), float(prm.hi),
                         float(prm.default), float(prm.step), key=key, help=hint)
    return st.number_input(prm.label, int(prm.lo), int(prm.hi), int(prm.default),
                           step=max(1, int(prm.step)), key=key, help=hint)


def _sidebar() -> dict:
    """左栏控件。返回一个 dict，主区照着跑。"""
    groups = dict(P.OP_GROUPS)
    with st.sidebar:
        st.markdown(_tag("控制台", THEME["gold"], "12px", MONO, "600"),
                    unsafe_allow_html=True)
        group = st.selectbox("算子分组", list(groups), key="group")
        op = st.selectbox("算子", groups[group], key=f"op::{group}",
                          help="myimg 里手写的算子名，和 benchmarks 报告同一套。")
        vs = dict(P.variants(op))
        variant = st.selectbox("实现变体", list(vs), key=f"variant::{op}",
                               format_func=lambda v: f"{v} — {vs[v]}",
                               help="naive 是逐像素 Python 循环，尺寸一大就明显卡。")
        binary = op in P.BINARY_OPS
        cases = list(engine.BINARY_CASES if binary else engine.GRAY_CASES)
        pref = P.DEFAULT_CASE_FOR_BINARY if binary else P.DEFAULT_CASE_FOR_GRAY
        case = st.selectbox("用例", cases, key=f"case::{binary}",
                            index=cases.index(pref) if pref in cases else 0,
                            format_func=lambda c: f"{c} — {engine.CASE_LABELS.get(c, '')}")
        size = st.selectbox("尺寸（长边 px）", list(SIZES), key="size",
                            index=list(SIZES).index(128))
        up = st.file_uploader("或上传自己的图（转灰度，长边压到 1024）",
                              type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
                              key="upload")
        st.markdown(_tag("参数", THEME["gold"], "12px", MONO, "600"),
                    unsafe_allow_html=True)
        prm_list = P.PARAMS.get(op, ())
        raw = {prm.key: _param_widget(op, prm) for prm in prm_list}
        if not prm_list:
            st.caption("这个算子没有可调参数。")
        st.caption("口径提醒都在每个控件右边的 ? 里，鼠标悬停可看。")
    return {"op": op, "variant": variant, "case": case, "size": int(size),
            "raw": raw, "upload": up, "prm_list": prm_list}


def _resolve_input(cfg: dict) -> tuple[np.ndarray, str]:
    """上传的图优先；否则按用例名取（合成用例 + data/ 里的真实图）。"""
    up = cfg["upload"]
    if up is not None:
        name = pathlib.Path(up.name).name
        tmp = pathlib.Path(tempfile.gettempdir()) / f"myimg_web_{name}"
        tmp.write_bytes(up.getvalue())
        return engine.open_file(tmp), f"file:{name}"
    return image_for(cfg["case"], cfg["size"], engine.real_images()), cfg["case"]


def _status(res) -> None:
    """状态区：和桌面版同一份文案，只是换成 HTML 卡片。"""
    tone = verdict_tone(res.verdict)
    color = THEME[tone]
    muted, info = THEME["muted"], THEME["info"]
    note, warn = THEME["note"], THEME["warn"]
    panel, line = THEME["panel"], THEME["line"]
    h, w = res.img.shape[0], res.img.shape[1]
    parts = [
        f'<div style="color:{color};font-weight:600">{render.summary_line(res)}</div>',
        f'<div style="color:{color}">判定：{res.verdict}'
        f'<span style="color:{muted}">　（口径 = 报告里的 {res.spec_op}）</span></div>',
        f'<div style="color:{info}">耗时：myimg {res.t_mine:.2f} ms / '
        f'OpenCV {res.t_ref:.2f} ms（{res.repeats} 次取最小 @ {h}×{w}）　'
        f'慢 {res.slowdown:.1f}×</div>',
        f'<div style="color:{note}">口径：{res.note}</div>',
    ]
    parts += [f'<div style="color:{warn}">⚠ {x}</div>' for x in res.warnings]
    st.markdown(
        f'<div style="border:1px solid {line};border-left:3px solid {color};'
        f'background:{panel};border-radius:6px;padding:10px 14px;'
        f'font-family:{CN};font-size:13px;line-height:1.85">'
        + "".join(parts) + "</div>", unsafe_allow_html=True)


def _sweep_chart(rows: list[dict], xlabel: str, metric_name: str):
    """扫描折线：点的颜色跟着判定走，红点就是超容差那一档。"""
    from matplotlib.figure import Figure

    fig = Figure(figsize=(9.4, 2.8), dpi=100)
    fig.set_layout_engine("constrained")
    ax = fig.subplots(1, 1)
    xs = list(range(len(rows)))
    ys = [np.nan if r["metric"] is None else r["metric"] for r in rows]
    ax.plot(xs, ys, color=render._C_MINE, lw=1.4, zorder=2)
    for x, y, r in zip(xs, ys, rows):
        if not np.isnan(y):
            ax.plot([x], [y], "o", ms=4.5, zorder=3,
                    color=THEME[verdict_tone(r["verdict"])])
    k = max(1, len(rows) // 8)
    ax.set_xticks(xs[::k])
    ax.set_xticklabels([r["label"] for r in rows][::k], fontsize=8)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(metric_name, fontsize=9)
    ax.grid(True, alpha=0.25, lw=0.6)
    return fig


def _rows_table(rows: list[dict], metric_name: str, index_name: str):
    import pandas as pd

    df = pd.DataFrame({
        metric_name: [r["metric"] for r in rows],
        "判定": [r["verdict"] for r in rows],
        "myimg ms": [round(r["t_mine"], 2) for r in rows],
        "OpenCV ms": [round(r["t_ref"], 2) for r in rows],
        "慢 ×": [round(r["slowdown"], 1) for r in rows],
        "备注": [r["error"].split(":")[0] if r["error"] else "" for r in rows],
    }, index=[r["label"] for r in rows])
    df.index.name = index_name
    return df


def _sweep_panel(cfg: dict, img: np.ndarray, case_label: str) -> None:
    """参数扫描 + 导出循环 GIF。帧是离屏渲染抓的，和主区那张图同一套画法。"""
    prm_list = cfg["prm_list"]
    if not prm_list:
        st.info("这个算子没有可调参数，没什么可扫的。")
        return
    labels = [p.label for p in prm_list]
    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    picked = c1.selectbox("扫哪个参数", labels, key=f"sweep::{cfg['op']}")
    prm = dict(zip(labels, prm_list))[picked]
    steps = int(c2.selectbox("帧数", list(SWEEP_STEPS), key=f"steps::{cfg['op']}"))
    width = int(c3.selectbox("GIF 宽度", list(GIF_WIDTHS), key="gifw", index=1))
    skip = int(c4.selectbox("抽帧", list(GIF_SKIPS), key="gifskip",
                            help="隔 k 帧取一帧，帧间隔同步乘 k，播放速度不掉。"))
    n_pts = len(P.sweep_sequence(prm, steps))
    st.caption(f"取值序列由 params.sweep_sequence 给：{prm.label} 走 {n_pts} 个点，"
               f"每点重算一次对拍并重画四格，帧就是下面 GIF 的帧。")
    if not st.button("开始扫描", type="primary"):
        return

    frames: list = []
    sfig = render.make_figure(dpi=GIF_FRAME_DPI)
    bar = st.progress(0.0, text="扫描中…")

    def grab(i: int, n: int, v, res) -> None:
        bar.progress((i + 1) / n,
                     text=f"扫描中… {i + 1}/{n}　{prm.label}={fmt_value(prm, v)}")
        if res is None:
            return              # 这一档算子抛异常，没东西可画，帧就少一张
        render.draw(sfig, res)
        frames.append(grab_frame(sfig))

    try:
        rows = sweep_rows(cfg["op"], cfg["variant"], case_label, cfg["size"],
                          prm.key, steps, img=img, on_frame=grab)
    finally:
        bar.empty()
    if not rows:
        st.warning("这个参数没有可扫的取值。")
        return

    name = next((r["metric_name"] for r in rows if not r["error"]), "指标")
    st.markdown(_tag(sweep_summary(rows, prm), THEME["warn"]),
                unsafe_allow_html=True)
    bad = [r for r in rows if r["error"]]
    if bad:
        st.warning("这些取值算子直接抛异常，已跳过（不是实现错，是基准不收这个口径）："
                   + "、".join(f"{prm.label}={r['label']}" for r in bad)
                   + f"　例：{bad[0]['error'][:160]}")
    st.pyplot(_sweep_chart(rows, f"{prm.label}（{len(rows)} 帧）", name),
              use_container_width=True)
    st.dataframe(_rows_table(rows, name, prm.label),
                 use_container_width=True, hide_index=False)

    if len(frames) < 2:
        st.warning("帧不够，导不出 GIF。")
        return
    sel, ms = prepare_gif_frames(frames, width, skip)
    if len(sel) < 2:
        st.warning("抽帧后不足两帧，把抽帧调小一点。")
        return
    buf = io.BytesIO()
    n_bytes = assemble_gif(sel, buf, ms)
    st.download_button(
        f"下载循环 GIF（{len(sel)} 帧 · {ms} ms/帧 · {width}px · "
        f"{n_bytes / 1024:.0f} KB）",
        buf.getvalue(), mime="image/gif",
        file_name=f"sweep_{cfg['op']}_{prm.key}.gif")
    st.image(buf.getvalue(), caption=f"{len(sel)} 帧循环")


def _footnote() -> None:
    st.caption("计算核与桌面版（python -m gui）完全同一份：engine.run + render.draw；"
               "容差与判定来自 benchmarks/run_compare.py 的 SPECS，和 112 组对拍报告同源。"
               "myimg/ 包内不出现 cv2 / skimage，基准只在 engine、render 这一层被调用。")


def main() -> None:
    st.set_page_config(page_title="myimg 图像算子对拍台", page_icon="◧",
                       layout="wide", initial_sidebar_state="expanded")
    render.setup_style()
    _brand()

    cfg = _sidebar()
    img, case_label = _resolve_input(cfg)

    try:
        res = engine.run(cfg["op"], cfg["variant"], cfg["raw"], img,
                         case=case_label)
    except Exception as exc:                 # 参数组合非法时算子会直接抛
        st.error(f"出错了：{type(exc).__name__}：{exc}")
        st.caption("参数组合不合法时算子会直接抛异常，页面不崩，改参数即可。")
        st.stop()

    views = render.views_for(res)
    c_view, c_png = st.columns([3, 1])
    view = c_view.selectbox("第四格看什么", views, key="view::" + "|".join(views))

    fig = render.make_figure(dpi=100)
    render.draw(fig, res, view)
    st.pyplot(fig, use_container_width=True)

    png = io.BytesIO()
    fig.savefig(png, format="png", dpi=150, bbox_inches="tight",
                facecolor=THEME["bg"])
    c_png.download_button(
        "导出 PNG", png.getvalue(), mime="image/png",
        file_name="myimg_%s_%s_%s.png" % (cfg["op"], cfg["variant"],
                                          case_label.replace(":", "_")))

    _status(res)
    with st.expander("参数扫描 / 导出循环 GIF", expanded=False):
        _sweep_panel(cfg, img, case_label)
    _footnote()


if __name__ == "__main__":
    main()