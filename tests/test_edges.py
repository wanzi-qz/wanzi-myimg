"""边缘检测 vs OpenCV。

Sobel 要求逐像素精确（同口径下就是精确的）。
Canny 不要求逐像素一致：OpenCV 用亚像素插值 NMS，这里用量化 NMS，
判据是 IoU 和边缘像素数量级。
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import assert_close, call

from benchmarks import cases, refs
from myimg import edges as E

TOL = 1e-10


def test_sobel_exact(gray):
    gx, gy, mag = call(E.sobel, gray)
    rgx, rgy, rmag = refs.sobel_ref(gray)
    assert_close(gx, rgx, TOL, "sobel gx")
    assert_close(gy, rgy, TOL, "sobel gy")
    assert_close(mag, rmag, TOL, "sobel magnitude")


@pytest.mark.parametrize("mode", ["reflect", "edge", "symmetric"])
def test_sobel_border_modes(gray, mode):
    gx, gy, mag = call(E.sobel, gray, mode)
    rgx, rgy, rmag = refs.sobel_ref(gray, mode)
    assert_close(mag, rmag, TOL, f"sobel mag mode={mode}")


def test_sobel_naive_matches_separable(gray):
    gx, gy, mag = call(E.sobel, gray)
    ngx, ngy, nmag = call(E.sobel_naive, gray)
    assert_close(mag, nmag, TOL, "sobel naive vs separable")


def test_sobel_magnitude_nonneg_and_float(gray):
    gx, gy, mag = call(E.sobel, gray)
    assert mag.min() >= 0
    for a in (gx, gy, mag):
        assert a.dtype == np.float64


def test_sobel_direction_on_vertical_edge():
    """垂直阶跃边缘上，梯度应主要落在 x 方向。"""
    step = cases.step_edge()
    gx, gy, mag = call(E.sobel, step)
    band = (slice(60, 68), slice(62, 66))
    assert np.abs(gx[band]).mean() > 10 * np.abs(gy[band]).mean()


@pytest.mark.parametrize("cn,lo,hi", [("gauss_noise", 50, 150), ("microscope", 50, 150), ("step_edge", 50, 150)])
def test_canny_iou_with_cv2(cn, lo, hi):
    img = cases.make(cn)
    mine = call(E.canny, img, float(lo), float(hi)) > 0
    refv = refs.canny_ref(img, lo, hi) > 0
    union = int((mine | refv).sum())
    iou = int((mine & refv).sum()) / union if union else 1.0
    assert iou >= 0.55, f"{cn} canny IoU={iou:.3f} < 0.55（量化 NMS 与 cv2 插值 NMS 的差距超出预期）"


def test_canny_pixel_count_same_order(gray):
    mine = call(E.canny, gray, 50.0, 150.0) > 0
    refv = refs.canny_ref(gray, 50, 150) > 0
    if refv.sum() > 0:
        ratio = mine.sum() / refv.sum()
        assert 0.5 <= ratio <= 2.0, f"边缘像素数比值 {ratio:.2f} 不在 0.5~2 之间"


def test_canny_single_pixel_wide_on_step():
    """阶跃边缘上 Canny 应输出单像素宽的边（NMS 生效的直接证据）。"""
    step = cases.step_edge()
    out = call(E.canny, step, 50.0, 150.0) > 0
    row = out[64]
    assert row.sum() == 1, f"阶跃边上检测到 {row.sum()} 个像素，NMS 没把边细化"


def test_canny_hysteresis_connects_weak_edges():
    """滞后连接：弱边只有连通到强边才保留。

    构造一条连续的垂直阶跃边：上半段对比度强（0/250，幅值远超 high），
    下半段对比度弱（120/140，幅值落在 low~high 之间）。
    弱段自身不够格，但因为和强段连通，必须被保留。
    """
    img = np.zeros((64, 64), dtype=np.uint8)
    img[0:32, 32:] = 250                    # 强对比段
    img[32:64, 0:32] = 120                  # 弱对比段（左）
    img[32:64, 32:] = 140                   # 弱对比段（右）
    out = call(E.canny, img, 40.0, 150.0) > 0
    assert out[48, 31] or out[48, 32], "连到强边的弱边没被滞后连接保留"


def test_canny_drops_isolated_weak_edges():
    """对照组：同样对比度的弱边，如果不连任何强边，应被整体丢弃。"""
    img = np.zeros((64, 64), dtype=np.uint8)
    img[:, 32:] = 140
    img[:, :32] = 120
    out = call(E.canny, img, 40.0, 150.0)
    assert out.max() == 0, "没有强边时弱边应被丢弃"


def test_canny_presmoothing_kills_noise_edges():
    """预平滑是 Canny 的第一步，也是它和「Sobel + 阈值」的本质区别。

    microscope 用例 = 暗背景 + 荧光颗粒 + sigma=22 的高斯噪声。
    2026-10-18 实测：sigma=0 时检出 5327 个边缘像素、318 个碎片（全是噪声）；
    sigma=1.4 时只剩 186 个像素、31 个连通块（颗粒的轮廓）。
    数量级地减少，才说明平滑真的在干活。
    """
    import myimg
    img = cases.microscope()
    raw = call(E.canny, img, 50.0, 150.0) > 0
    sm = call(E.canny, img, 50.0, 150.0, sigma=1.4) > 0
    assert int(raw.sum()) > 20 * int(sm.sum()), (
        f"预平滑没压住噪声边缘：raw {int(raw.sum())} vs smoothed {int(sm.sum())}")
    _, n_raw = myimg.label_components(raw.astype(np.uint8), 8)
    _, n_sm = myimg.label_components(sm.astype(np.uint8), 8)
    assert n_sm < n_raw / 5, f"碎片数 {n_raw} -> {n_sm}，平滑后仍然满地碎片"


def test_canny_presmoothed_finds_nothing_on_pure_noise():
    """对照组：纯噪声图没有结构，预平滑之后梯度幅值整体掉到阈值以下，
    应当一个边缘都检不出。检出了说明阈值或幅值尺度错了。"""
    img = cases.gauss_noise()
    out = call(E.canny, img, 50.0, 150.0, sigma=1.4)
    assert out.max() == 0, f"纯噪声图上检出 {int((out > 0).sum())} 个边缘像素"
