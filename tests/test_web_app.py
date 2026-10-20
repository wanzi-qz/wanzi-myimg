"""web 版对拍台（Streamlit）的纯函数部分。

不起服务器：web/app.py 模块级只有 import 和纯函数，所有 st.* 都在 main() 里，
所以这里直接 import 就能对拍。最后那条子进程测试是这轮重构的正题——
web 版必须能在没有显示环境的机器上导入，也就是不许把 tkinter 连累进来。
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import matplotlib

matplotlib.use("Agg")  # 必须在 import 任何会碰后端的模块之前

import numpy as np   # noqa: E402
import pytest        # noqa: E402

from gui import engine, params as P, render   # noqa: E402
from gui.theme import verdict_color, verdict_tone   # noqa: E402
from web import app as W   # noqa: E402

render.setup_style()  # 中文字体；不调的话抓帧会刷一堆 glyph 警告

SIGMA = P.PARAMS["canny"][2]        # canny 的第 3 个参数就是预平滑 σ
KSIZE = P.PARAMS["box_filter"][0]   # choice 型，扫的是候选顺序


def test_sweep_rows_canny_sigma():
    """扫 canny 的 σ：帧数 = sweep_sequence 长度，主指标是 IoU 且落在 [0,1]。"""
    seq = P.sweep_sequence(SIGMA, 4)
    rows = W.sweep_rows("canny", "vector", "microscope", 64, "sigma", steps=4)
    assert seq[0] == 0.0 and len(seq) == 4
    assert len(rows) == len(seq)
    assert [r["value"] for r in rows] == seq
    assert all(r["metric_name"] == "IoU" for r in rows)
    assert all(r["metric"] is None or 0.0 <= r["metric"] <= 1.0 for r in rows)
    assert all(isinstance(r["both_empty"], bool) for r in rows)
    assert all(r["t_mine"] > 0 and r["slowdown"] > 0 for r in rows)
    assert [r["label"] for r in rows] == [W.fmt_value(SIGMA, v) for v in seq]


def test_sweep_rows_choice_param_follows_option_order():
    """choice 型参数按候选顺序走完，不按数值 linspace。"""
    rows = W.sweep_rows("box_filter", "vector", "ramp", 32, "ksize", steps=12)
    assert [r["value"] for r in rows] == list(KSIZE.options)
    assert rows[0]["metric_name"] == "max|err|"
    # box filter 与 cv2.boxFilter 逐像素精确，剩下的是浮点求和次序的噪声
    assert rows[0]["metric"] == pytest.approx(0.0, abs=1e-9)
    assert all(r["error"] == "" for r in rows)


def test_sweep_rows_unknown_param_returns_empty():
    assert W.sweep_rows("canny", "vector", "ramp", 32, "nope") == []
    assert W.sweep_rows("otsu_threshold", "vector", "ramp", 32, "bins") == []


def test_sweep_rows_on_frame_gets_every_step():
    """回调逐帧拿到 (i, n, value, res)：web 就是靠它抓帧 + 刷进度条。"""
    seen = []
    rows = W.sweep_rows("gaussian_filter", "vector", "gauss_noise", 32, "sigma",
                        steps=6,
                        on_frame=lambda i, n, v, res: seen.append((i, n, v, res)))
    assert len(seen) == len(rows) == 6
    assert [i for i, *_ in seen] == list(range(6))
    assert all(n == 6 for _, n, _, _ in seen)
    assert all(res is not None for *_, res in seen)


def test_sweep_rows_skips_values_the_reference_rejects():
    """基准不收的取值要记成出错行、跳过，不能拿上一档的结果顶替。

    sobel 的 mode='wrap'：cv2.Sobel 明确拒绝 BORDER_WRAP，这是口径差异。
    """
    seen = []
    rows = W.sweep_rows("sobel", "vector", "step_edge", 32, "mode",
                        on_frame=lambda i, n, v, res: seen.append((v, res)))
    assert [r["value"] for r in rows] == list(P.MODES)
    bad = [r for r in rows if r["error"]]
    assert [r["value"] for r in bad] == ["wrap"]
    assert bad[0]["metric"] is None and bad[0]["verdict"] == "运行出错"
    assert bad[0]["metric_name"] == "mag max|err|"      # 前面几档已经定下主指标名
    assert np.isnan(bad[0]["t_mine"])
    assert [v for v, res in seen if res is None] == ["wrap"]   # 回调照样收到这一档
    assert len(seen) == len(rows)
    assert "已跳过" in W.sweep_summary(rows, P.PARAMS["sobel"][0])


def test_sweep_summary_reports_min_max_location():
    rows = W.sweep_rows("canny", "vector", "microscope", 64, "sigma", steps=4)
    s = W.sweep_summary(rows, SIGMA)
    assert "参数扫描完成" in s and "IoU" in s and "帧" in s
    assert W.sweep_summary([], SIGMA) == "扫描完成：没有可用指标"
    empty = [dict(rows[0], metric=None)]
    assert W.sweep_summary(empty, SIGMA) == "扫描完成：没有可用指标"


def test_image_for_and_fmt_value():
    img = W.image_for("blobs", 48)
    assert img.shape == (48, 48) and img.dtype == np.uint8
    assert engine.is_binary(img)
    other = np.zeros((8, 8), np.uint8)
    assert W.image_for("whatever", 8, real={"whatever": other}) is other
    assert W.fmt_value(SIGMA, 1.23456) == "1.2"          # float 按 digits
    assert W.fmt_value(P.PARAMS["clahe"][1], 8.4) == "8"  # int 取整
    assert W.fmt_value(KSIZE, "5") == "5"                 # choice 原样


def test_status_helpers_are_shared_with_desktop():
    """状态区第一行和判定色是两个界面共用的，不能各写一份。"""
    img = engine.synth_case("microscope", 48)
    res = engine.run("canny", "vector", P.defaults("canny"), img, "microscope")
    line = render.summary_line(res)
    assert line.startswith("IoU =") and "px" in line
    name, val = engine.headline(res)
    assert name == "IoU" and val == pytest.approx(res.metrics["iou"])
    assert verdict_tone("一致") == "ok"
    assert verdict_tone("仅记录") == "warn"
    assert verdict_tone("超出容差") == "bad"
    assert verdict_tone("没见过的判定") == "bad"     # 名单外宁可扎眼
    assert verdict_color("一致") == "#86c991"


@pytest.mark.parametrize("kind,op,case", [
    ("image", "box_filter", "ramp"),
    ("tuple3", "sobel", "step_edge"),
    ("scalar", "otsu_threshold", "gauss_noise"),
    ("counts", "histogram", "microscope"),
    ("labels", "label_components", "blobs"),
    ("stats", "region_stats", "blobs"),
    ("iou", "canny", "microscope"),
])
def test_summary_line_covers_every_kind(kind, op, case):
    """七种 kind 都要能出一行文案，web 的状态卡片才不会空一块。"""
    img = engine.synth_case(case, 48)
    res = engine.run(op, "vector", P.defaults(op), img, case)
    assert res.kind == kind
    line = render.summary_line(res)
    assert isinstance(line, str) and line.strip()


def test_grab_frame_renders_and_refreshes():
    """抓帧必须能出图、且每次抓到的是当前内容（不是上一帧的残留）。

    这条对应一个真出过的 bug：裸 Figure 的 canvas 是 FigureCanvasBase，
    没有 buffer_rgba，扫描到第一帧就 AttributeError。
    """
    img = engine.synth_case("blobs", 48)
    res_a = engine.run("label_components", "vector",
                       P.defaults("label_components"), img, "blobs")
    res_b = engine.run("erode", "vector", P.defaults("erode"), img, "blobs")
    fig = render.make_figure(dpi=50)
    render.draw(fig, res_a)
    a = W.grab_frame(fig)
    assert a.mode == "RGB" and a.size == (470, 370)      # 9.4x7.4 in @ 50dpi
    render.draw(fig, res_b)
    b = W.grab_frame(fig)
    assert not np.array_equal(np.asarray(a), np.asarray(b))


def test_web_app_imports_without_tkinter():
    """把 tkinter 掐掉再 import web.app：这条链必须干净。"""
    code = textwrap.dedent(f"""
        import sys
        sys.modules["tkinter"] = None          # 谁 import 谁报错
        sys.modules["tkinter.font"] = None
        sys.path.insert(0, r"{W.ROOT}")
        import matplotlib
        matplotlib.use("Agg")
        import web.app as w
        assert callable(w.main) and callable(w.sweep_rows)
        print("ok")
    """)
    out = subprocess.run([sys.executable, "-X", "utf8", "-c", code],
                         capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    assert "ok" in out.stdout