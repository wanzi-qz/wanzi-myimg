"""直方图 / 阈值 / 均衡 vs OpenCV。"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import assert_close, call

from benchmarks import refs
import importlib

# 注意：myimg.histogram 这个名字被包级导出遮蔽成了函数，取模块要用 importlib
H = importlib.import_module("myimg.histogram")


def test_histogram_counts(gray):
    h = call(H.histogram, gray)
    assert h.sum() == gray.size
    assert_close(h, refs.histogram_ref(gray), 0, "histogram")


def test_histogram_last_bin_inclusive(gray):
    """255 必须落在最后一个桶里（右闭）。"""
    img = np.full((4, 4), 255, dtype=np.uint8)
    h = call(H.histogram, img)
    assert h[255] == 16


@pytest.mark.parametrize("cn", ["gauss_noise", "microscope", "uneven_light", "step_edge", "random_uint8"])
def test_otsu_within_one_gray(cn):
    from benchmarks import cases
    img = cases.make(cn)
    mine = call(H.otsu_threshold, img)
    refv = refs.otsu_ref(img)
    assert abs(mine - refv) <= 1, f"otsu {mine} vs cv2 {refv}"


def test_otsu_bimodal_exact():
    """干净双峰图必须精确命中两峰之间的谷。"""
    rng = np.random.default_rng(1)
    img = np.concatenate([rng.normal(60, 8, 4000), rng.normal(190, 8, 4000)])
    img = np.clip(img, 0, 255).astype(np.uint8).reshape(80, 100)
    t = call(H.otsu_threshold, img)
    assert 90 <= t <= 150, f"双峰图上 otsu 给出 {t}，没落在谷里"


@pytest.mark.parametrize("cn", ["gauss_noise", "microscope", "uneven_light", "step_edge", "random_uint8"])
def test_otsu_variance_is_maximal(cn):
    """大津法的定义式复核：暴力遍历所有 t 算类间方差，最优 t 必须和递推实现一致。
    这条和上一条互补 —— 上一条查「和 cv2 一样」，这条查「和定义一样」。"""
    from benchmarks import cases
    img = cases.make(cn)
    hist = call(H.histogram, img).astype(np.float64)
    total = hist.sum()
    t_idx = np.arange(256, dtype=np.float64)
    w0 = np.cumsum(hist)
    sum0 = np.cumsum(hist * t_idx)
    sum_all = sum0[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        mu0 = sum0 / w0
        mu1 = (sum_all - sum0) / (total - w0)
        var = w0 * (total - w0) * (mu0 - mu1) ** 2
    var[~np.isfinite(var)] = -1.0
    brute = int(np.argmax(var))  # argmax 取第一个最大值 == 平局取最小 t
    assert int(call(H.otsu_threshold, img)) == brute, f"{cn}: 递推 {call(H.otsu_threshold, img)} vs 暴力 {brute}"


@pytest.mark.parametrize("cn", ["gauss_noise", "microscope", "uneven_light", "step_edge", "flat"])
def test_equalize_exact(cn):
    from benchmarks import cases
    img = cases.make(cn)
    assert_close(call(H.hist_equalize, img), refs.hist_equalize_ref(img), 0, f"equalize {cn}")


def test_equalize_flat_returns_input():
    """单一灰阶是退化情况，OpenCV 原样返回。"""
    from benchmarks import cases
    img = cases.flat()
    assert np.array_equal(call(H.hist_equalize, img), img)


def test_clahe_self_consistent(gray):
    """CLAHE 的削峰取整、重分配轮数、tile 边界插值口径都和 OpenCV 不完全一样，
    不追求逐像素对拍（差异数字由 benchmarks/run_compare.py 生成，见 benchmarks/out/）。
    这里只查自洽性：dtype、形状、确定性。"""
    out = call(H.clahe, gray)
    assert out.dtype == np.uint8 and out.shape == gray.shape
    assert np.array_equal(out, call(H.clahe, gray)), "clahe 不确定（同输入两次结果不同）"


@pytest.mark.parametrize("cn", ["uneven_light", "microscope"])
def test_clahe_boosts_dark_local_contrast(cn):
    """CLAHE 存在的理由：光照不均时暗块的局部对比度要被拉开。

    只用「本来就暗、且有结构」的用例测这条。纯噪声（random_uint8）和脉冲（impulse）
    的直方图本来就摊满 0..255，削峰只会让它们的标准差略降 —— 拿它们测这条会得出
    相反结论，那不是 bug，是用例选错了。
    """
    from benchmarks import cases
    img = cases.make(cn)
    h, w = img.shape
    tile = (slice(0, h // 4), slice(0, w // 4))
    out = call(H.clahe, img)
    before, after = img[tile].std(), out[tile].std()
    assert after >= before, f"{cn} 暗块标准差 {before:.2f} -> {after:.2f}，局部对比度没拉开"


def test_clahe_beats_global_on_uneven_light():
    """CLAHE 相对全局均衡的增益：光照不均时暗块对比度提升倍数更高。"""
    from benchmarks import cases
    img = cases.uneven_light()
    h, w = img.shape
    tile = (slice(0, h // 4), slice(0, w // 4))
    g = call(H.hist_equalize, img)
    c = call(H.clahe, img)
    base = img[tile].std() + 1e-9
    assert (c[tile].std() / base) >= (g[tile].std() / base) * 0.9
