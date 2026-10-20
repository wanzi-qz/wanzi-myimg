"""GUI 的计算层 / 渲染层回归测试（不弹窗，Agg 后端）。

守三件事：
1. params 表里的 key 与 myimg 真实签名逐条对齐 —— 名字写错不会在启动时报错，
   只会在点中那个算子的一瞬间炸，所以必须在这里拦；
2. 每个算子 × 界面默认参数，engine.run 不抛异常且判定 / 口径说明非空
   （数值判定复用 benchmarks.run_compare，这里不重复断言数字）；
3. 每个算子的所有信息格视图 render.draw 不崩。
"""

from __future__ import annotations

import inspect

import matplotlib

matplotlib.use("Agg")  # 必须在 import 任何会碰后端的模块之前

import pytest  # noqa: E402

from gui import engine, params as P, render  # noqa: E402

render.setup_style()  # 中文字体；不调的话 savefig 会刷一堆 glyph 警告

KINDS = {"image", "counts", "scalar", "labels", "stats", "tuple3", "iou"}


def _case_for(op: str) -> str:
    return (P.DEFAULT_CASE_FOR_BINARY if op in P.BINARY_OPS
            else P.DEFAULT_CASE_FOR_GRAY)


def _run(op: str, repeats: int = 1):
    case = _case_for(op)
    img = engine.synth_case(case, 64)  # 64² 够快；naive 变体不在这里跑
    variant = P.variants(op)[0][0]
    return engine.run(op, variant, P.defaults(op), img, case, repeats=repeats)


# ---------------------------------------------------------------- 参数表
@pytest.mark.parametrize("op", P.ALL_OPS)
def test_param_keys_match_myimg_signature(op):
    """表里的 key 必须真的能 **kwargs 进 myimg 的函数。"""
    for variant, _ in P.variants(op):
        names = set(inspect.signature(engine.mine_fn(op, variant)).parameters)
        for p in P.PARAMS.get(op, ()):
            assert p.key in names, f"{op}/{variant} 的签名里没有参数 {p.key!r}"


@pytest.mark.parametrize("op", P.ALL_OPS)
def test_param_defaults_inside_widget_range(op):
    """默认值必须在控件范围 / 候选里，否则界面一出生就是错的。"""
    for p in P.PARAMS.get(op, ()):
        if p.kind == "choice":
            assert str(p.default) in p.options, f"{op}.{p.key} 默认值不在候选里"
        else:
            assert p.lo <= float(p.default) <= p.hi, f"{op}.{p.key} 默认值越界"


@pytest.mark.parametrize("op", P.ALL_OPS)
def test_coerce_defaults_are_callable_kwargs(op):
    """coerce 之后的字典能直接喂给算子（AUTO -> None 这类转换也覆盖到）。"""
    kw = P.coerce(op, P.defaults(op))
    fn = engine.mine_fn(op, P.variants(op)[0][0])
    sig = inspect.signature(fn)
    sig.bind(None, **kw)  # 只检查绑定，不真跑


# ---------------------------------------------------------------- 计算层
@pytest.mark.parametrize("op", P.ALL_OPS)
def test_engine_run_default_params(op):
    res = _run(op)
    assert res.kind in KINDS
    assert res.verdict, "判定不能是空串"
    assert res.note, "口径说明不能是空串"
    assert res.spec_op, "必须落到报告里的某条口径"
    assert res.t_mine >= 0.0 and res.t_ref >= 0.0
    assert res.slowdown >= 0.0


def test_engine_spec_routing_canny_sigma():
    """σ>0 时基准必须切到 canny_smoothed（record_only），和报告同口径。"""
    assert engine.spec_for("canny", "vector", {"sigma": 0.0}).op == "canny"
    assert engine.spec_for("canny", "vector", {"sigma": 1.0}).op == "canny_smoothed"
    assert engine.spec_for("box_filter", "integral", {}).op == "box_filter_integral"
    assert engine.spec_for("box_filter", "vector", {}).op == "box_filter"


def test_engine_warnings_median_mode():
    """median 选非 edge 边界时必须给坑提示，别让用户对着超容差发呆。"""
    case = "impulse"
    img = engine.synth_case(case, 64)
    res = engine.run("median_filter", "vector",
                     {"ksize": "5", "mode": "reflect"}, img, case, repeats=1)
    assert any("REPLICATE" in w for w in res.warnings)


# ---------------------------------------------------------------- 渲染层
@pytest.mark.parametrize("op", P.ALL_OPS)
def test_render_every_view(op):
    res = _run(op)
    views = render.views_for(res)
    assert views and render.default_view(res) == views[0]
    fig = render.make_figure(dpi=60)
    for view in views:
        render.draw(fig, res, view)
    render.draw(fig, res, None)  # 空 view 走默认


def test_render_save_png(tmp_path):
    res = _run("gaussian_filter")
    fig = render.make_figure(dpi=60)
    render.draw(fig, res, None)
    out = render.save_png(fig, str(tmp_path / "x.png"), dpi=80)
    assert (tmp_path / "x.png").stat().st_size > 0
    assert out.endswith("x.png")
# ---------------------------------------------------------------- 参数扫描
def test_sweep_choice_keeps_option_order():
    p = P.PARAMS["median_filter"][1]  # 边界 mode
    assert p.kind == "choice"
    assert P.sweep_sequence(p) == list(p.options)


def test_sweep_float_rounds_and_dedups():
    p = P.Param("x", "x", "float", 0.0, lo=0.0, hi=0.1, step=0.01, digits=1)
    seq = P.sweep_sequence(p, steps=24)
    assert seq[0] == 0.0 and seq[-1] == 0.1
    assert len(seq) == len(set(seq))           # 舍入后不留重复帧
    assert all(round(v, 2) == v for v in seq)  # 只保留 digits+1 位小数


def test_sweep_odd_stays_odd():
    p = P.Param("k", "k", "odd", 3, lo=3, hi=15, step=2)
    seq = P.sweep_sequence(p, steps=24)
    assert all(k % 2 == 1 for k in seq)
    assert seq == sorted(set(seq))
    assert seq[0] == 3 and seq[-1] == 15


def test_sweep_int_rounds_and_steps_floor():
    p = P.Param("n", "n", "int", 0, lo=0, hi=10, step=1)
    assert P.sweep_sequence(p, steps=4) == [0, 3, 7, 10]
    assert P.sweep_sequence(p, steps=1) == [0, 10]  # steps<2 兜底成两端点


# ---------------------------------------------------------------- 主题色
def test_render_dark_theme_rcparams():
    import matplotlib.pyplot as plt
    assert plt.rcParams["figure.facecolor"] == "#070707"
    assert plt.rcParams["savefig.facecolor"] == "#070707"
    assert plt.rcParams["axes.facecolor"] == "#0b0b0c"
    assert plt.rcParams["axes.edgecolor"] == "#3b3b3c"


def test_app_theme_tokens_and_blend():
    from gui import app as A
    for k in ("bg", "bg_soft", "panel", "line", "ink", "muted", "faint", "gold"):
        c = A.THEME[k]
        assert c.startswith("#") and len(c) == 7
    assert A.THEME["bg"] == "#070707"
    assert A.THEME["gold"] == "#d8b46a"
    assert A._blend("#ffffff", "#000000", 0.5) == "#808080"
    assert A._blend("#d8b46a", "#070707", 0.0) == "#070707"
    assert A._blend("#d8b46a", "#070707", 1.0) == "#d8b46a"
def test_assemble_gif_frames_palette_and_duration(tmp_path):
    from PIL import Image
    from gui import app as A
    frames = [Image.new("RGB", (48, 32), c)
              for c in ((7, 7, 7), (216, 180, 106), (243, 243, 241))]
    out = tmp_path / "s.gif"
    n = A.assemble_gif(frames, str(out), 120)
    assert n == out.stat().st_size and n > 0
    im = Image.open(out)
    assert getattr(im, "n_frames", 1) == 3
    assert im.info.get("duration") == 120
    # 统一调色板：第二帧取回来应该还是强调金，不允许各帧各自量化跑色
    im.seek(1)
    r, g, b = im.convert("RGB").getpixel((0, 0))
    assert abs(r - 216) <= 2 and abs(g - 180) <= 2 and abs(b - 106) <= 2
def test_prepare_gif_frames_skip_width_and_duration():
    from PIL import Image
    from gui import app as A
    frames = [Image.new("RGB", (96, 60), (i * 7 % 256, 0, 0)) for i in range(10)]
    sel, ms = A.prepare_gif_frames(frames, 48, 3)
    assert len(sel) == 4                      # 0 3 6 9
    assert sel[0].size == (48, 30)
    assert ms == A.FRAME_MS * 3               # 抽帧后放慢，播放速度不掉
    sel2, ms2 = A.prepare_gif_frames(frames, 96, 1)
    assert len(sel2) == 10 and sel2[0].size == (96, 60) and ms2 == A.FRAME_MS
    assert len(frames) == 10 and frames[0].size == (96, 60)   # 缓存帧不被改
