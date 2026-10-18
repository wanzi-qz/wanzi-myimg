"""滤波算子 vs OpenCV 的数值对拍。

容差 1e-8：同口径下纯 NumPy 与 OpenCV 的差应在 1e-13 量级，
留 1e-8 是给累加顺序不同的余量。超过就说明实现有实质差异，不是浮点噪声。
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import assert_close, call

from benchmarks import refs
from myimg import filters as F

TOL = 1e-8


@pytest.mark.parametrize("k", [3, 5, 9])
def test_box_matches_cv2(gray, k):
    assert_close(call(F.box_filter, gray, k), refs.box_ref(gray, k), TOL, f"box k={k}")


@pytest.mark.parametrize("mode", ["reflect", "symmetric", "edge", "wrap"])
def test_box_border_modes(gray, mode):
    assert_close(call(F.box_filter, gray, 5, mode), refs.box_ref(gray, 5, mode), TOL, f"box mode={mode}")


def test_box_integral_matches_vectorized(gray):
    assert_close(call(F.box_filter_integral, gray, 5), call(F.box_filter, gray, 5), TOL, "box integral")


def test_box_integral_matches_cv2(gray):
    assert_close(call(F.box_filter_integral, gray, 7), refs.box_ref(gray, 7), TOL, "box integral vs cv2")


@pytest.mark.parametrize("sigma", [0.5, 1.0, 1.4, 2.0, 3.0])
def test_gaussian_auto_ksize_matches_cv2(gray, sigma):
    """自动核尺寸必须和 OpenCV 一致（±4σ，不是网传的 ±3σ）。"""
    assert_close(call(F.gaussian_filter, gray, sigma), refs.gaussian_ref(gray, sigma), TOL, f"gauss sigma={sigma}")


@pytest.mark.parametrize("k", [5, 7, 11])
def test_gaussian_explicit_ksize(gray, k):
    assert_close(call(F.gaussian_filter, gray, 1.5, k), refs.gaussian_ref(gray, 1.5, k), TOL, f"gauss k={k}")


def test_gaussian_separable_equals_2d(gray):
    assert_close(call(F.gaussian_filter, gray, 1.2), call(F.gaussian_filter_naive, gray, 1.2), TOL, "separable vs 2d")


def test_gaussian_kernel_matches_cv2():
    import cv2
    mine = F.gaussian_kernel1d(2.0, 7)
    theirs = cv2.getGaussianKernel(7, 2.0).ravel()
    assert_close(mine, theirs, 1e-12, "gaussian kernel1d")


@pytest.mark.parametrize("k", [3, 5])
def test_median_matches_cv2(gray, k):
    assert_close(call(F.median_filter, gray, k), refs.median_ref(gray, k), TOL, f"median k={k}")


def test_median_naive_matches_vectorized(gray):
    assert_close(call(F.median_filter_naive, gray, 3), call(F.median_filter, gray, 3), TOL, "median naive")


def test_median_preserves_step_edge():
    """中值滤波保边：阶跃边缘经过中值后不应被糊掉。"""
    from benchmarks import cases
    step = cases.step_edge()
    out = call(F.median_filter, step, 3)
    col = out[64]
    jump = np.abs(np.diff(col)).max()
    assert jump > 100, f"中值滤波把阶跃糊掉了，边缘跳变只剩 {jump}"


def test_gaussian_blurs_step_edge():
    """对照组：高斯会把同一条阶跃糊掉。这两条测试放一起就是「保边性」的完整论证。"""
    from benchmarks import cases
    step = cases.step_edge()
    out = call(F.gaussian_filter, step, 2.0)
    col = out[64]
    grad = np.abs(np.diff(col))
    assert grad.max() < 60, "高斯滤波没有起到平滑作用"
