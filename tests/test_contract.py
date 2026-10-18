"""契约测试：不关心算得对不对，只关心接口守不守约。

这些测试在算子还是占位时就会 failed（NotImplementedError 冒出来），
实现完之后应当全绿。它们防的是「数值碰巧对了但接口是坏的」这类问题。
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import call

import myimg

UINT8_OPS = ["hist_equalize", "clahe", "canny", "erode", "dilate", "morph_open", "morph_close"]

# 有些算子光给一张图连签名校验都走不到（缺位置参数会先抛 TypeError），
# 契约测试要给足参数，才能测到「彩色图必须被拒」这条。
EXTRA_ARGS = {
    "canny": (50.0, 150.0),
    "label_components": (8,),
    "region_stats": (8,),
}

# 这些算子的输出不是「一张同尺寸的图」：直方图是 1D、阈值是标量、
# 连通域是 (标签图, 个数)、属性表是 list[dict]。形状契约单独一条测试写死。
NON_IMAGE_OPS = {"histogram", "otsu_threshold", "label_components", "region_stats"}


def _run(name, src):
    return call(myimg.OPS[name], src, *EXTRA_ARGS.get(name, ()))


@pytest.mark.parametrize("name", sorted(myimg.OPS))
def test_rejects_non_array(name):
    with pytest.raises(TypeError):
        _run(name, [[1, 2], [3, 4]])


@pytest.mark.parametrize("name", sorted(myimg.OPS))
def test_rejects_color_input(name):
    with pytest.raises(ValueError):
        _run(name, np.zeros((8, 8, 3), dtype=np.uint8))


@pytest.mark.parametrize("name", sorted(myimg.OPS))
def test_preserves_shape_and_no_mutation(name, gray):
    src = gray.copy()
    out = _run(name, src)
    if name == "sobel":
        out = out[2]
    if name not in NON_IMAGE_OPS:
        assert out.shape == gray.shape, f"{name} 输出形状 {out.shape} != 输入 {gray.shape}"
    assert np.array_equal(src, gray), f"{name} 修改了输入数组"


@pytest.mark.parametrize("name", sorted(NON_IMAGE_OPS))
def test_analysis_output_contract(name, gray):
    """分析类算子的输出契约。形状/类型/dtype 逐个写死，别让它以后悄悄变。"""
    out = _run(name, gray.copy())
    if name == "histogram":
        assert out.shape == (256,), f"直方图应是 (256,)，收到 {out.shape}"
        assert out.dtype == np.int64
        assert int(out.sum()) == gray.size, "计数总和 != 像素总数"
    elif name == "otsu_threshold":
        assert isinstance(out, float), f"阈值应是 float，收到 {type(out).__name__}"
        assert 0 <= out <= 255
    elif name == "label_components":
        labels, n = out
        assert labels.shape == gray.shape
        assert labels.dtype.kind in "iu", f"标签图应是整型，收到 {labels.dtype}"
        assert isinstance(n, int) and n >= 0
        assert int(labels.max()) == n, f"最大标签 {labels.max()} != 连通域个数 {n}"
    elif name == "region_stats":
        assert isinstance(out, list)
        assert all(isinstance(d, dict) for d in out)
        need = {"label", "area", "x", "y", "w", "h", "cx", "cy"}
        for d in out:
            assert need <= set(d), f"属性表缺字段 {need - set(d)}"


@pytest.mark.parametrize("name", UINT8_OPS)
def test_uint8_output(name, gray):
    fn = myimg.OPS[name]
    if name == "canny":
        out = call(fn, gray, 50.0, 150.0)
    else:
        out = call(fn, gray)
    assert out.dtype == np.uint8, f"{name} 应返回 uint8，收到 {out.dtype}"


@pytest.mark.parametrize("name", ["box_filter", "gaussian_filter", "median_filter", "sobel"])
def test_accepts_float64(name, gray):
    fn = myimg.OPS[name]
    f = gray.astype(np.float64)
    out = call(fn, f)
    if name == "sobel":
        out = out[2]
    assert out.dtype == np.float64


def test_box_does_not_create_extremes(gray):
    """均值不会创造新的极值：输出值域必须被输入值域包住。"""
    out = call(myimg.box_filter, gray, 5)
    assert out.min() >= gray.min() - 1e-9
    assert out.max() <= gray.max() + 1e-9


def test_equalize_mapping_monotonic(gray):
    """直方图均衡的 LUT 必须单调不减，否则灰度顺序被反转。"""
    u = np.arange(256, dtype=np.uint8).reshape(16, 16)
    eq = call(myimg.hist_equalize, u)
    assert np.all(np.diff(eq.ravel()) >= 0)


def test_canny_is_binary(gray):
    out = call(myimg.canny, gray, 50.0, 150.0)
    assert set(np.unique(out)) <= {0, 255}


def test_erode_border_convention():
    """腐蚀的边界必须当作前景（cv2 的 +inf）。整张前景腐蚀后应仍是整张前景。"""
    full = np.full((16, 16), 255, dtype=np.uint8)
    out = call(myimg.erode, full, 3)
    assert out.min() == 255, "腐蚀把贴边像素吃掉了 —— 边界填充值用错了（应为 +inf/True 而非 0）"


def test_dilate_border_convention():
    """膨胀的边界必须当作背景（cv2 的 -inf）。整张背景膨胀后应仍是整张背景。"""
    empty = np.zeros((16, 16), dtype=np.uint8)
    out = call(myimg.dilate, empty, 3)
    assert out.max() == 0


def test_odd_ksize_enforced(gray):
    with pytest.raises(ValueError):
        call(myimg.median_filter, gray, 4)
    with pytest.raises(ValueError):
        call(myimg.gaussian_kernel1d, 1.0, 6)


def test_sigma_positive_enforced(gray):
    with pytest.raises(ValueError):
        call(myimg.gaussian_filter, gray, 0.0)
