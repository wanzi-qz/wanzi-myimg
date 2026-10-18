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
