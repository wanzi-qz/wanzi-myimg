"""myimg —— 纯 NumPy 手写图像处理算子，与 OpenCV / scikit-image 对拍。

算子实现见 filters.py / histogram.py / edges.py / morphology.py。
库依赖只允许出现在 tests/ 和 benchmarks/ 里。
"""

from __future__ import annotations

from ._core import (
    MYIMG_EPS,
    PAD_TO_CV2,
    SUPPORTED_PAD_MODES,
    as_float64,
    check_gray,
    odd_kernel_size,
    pad,
    to_uint8,
)
from .edges import canny, sobel
from .filters import box_filter, gaussian_filter, gaussian_kernel1d, median_filter
from .histogram import clahe, hist_equalize, histogram, otsu_threshold
from .morphology import (
    dilate,
    erode,
    label_components,
    morph_close,
    morph_open,
    region_stats,
)

__version__ = "0.1.0"

# 全部手写算子。benchmarks/run_compare.py 和 examples/cli.py 都从这里取。
# 顺序就是建议的实现顺序：从简单到复杂，前一个是后一个的地基。
OPS: dict[str, object] = {
    "box_filter": box_filter,
    "gaussian_filter": gaussian_filter,
    "median_filter": median_filter,
    "histogram": histogram,
    "otsu_threshold": otsu_threshold,
    "hist_equalize": hist_equalize,
    "erode": erode,
    "dilate": dilate,
    "morph_open": morph_open,
    "morph_close": morph_close,
    "sobel": sobel,
    "label_components": label_components,
    "region_stats": region_stats,
    "canny": canny,
    "clahe": clahe,
}

__all__ = [
    "__version__",
    "OPS",
    "implemented_ops",
    "pending_ops",
    "MYIMG_EPS",
    "PAD_TO_CV2",
    "SUPPORTED_PAD_MODES",
    "as_float64",
    "check_gray",
    "odd_kernel_size",
    "pad",
    "to_uint8",
    "box_filter",
    "gaussian_filter",
    "gaussian_kernel1d",
    "median_filter",
    "histogram",
    "otsu_threshold",
    "hist_equalize",
    "clahe",
    "sobel",
    "canny",
    "erode",
    "dilate",
    "morph_open",
    "morph_close",
    "label_components",
    "region_stats",
]


def _probe_call(name: str, fn, img):
    """按各算子的签名喂一组最小可用参数，用来探测它实现了没有。"""
    if name == "gaussian_filter":
        return fn(img, sigma=1.0)
    if name == "canny":
        return fn(img, low=50.0, high=150.0)
    if name == "clahe":
        return fn(img)
    if name == "region_stats":
        return fn(img)
    return fn(img)


def pending_ops() -> list[str]:
    """还没实现（函数体仍是 NotImplementedError）的算子名列表。"""
    import numpy as np

    probe = np.zeros((5, 5), dtype=np.uint8)
    out = []
    for name, fn in OPS.items():
        try:
            _probe_call(name, fn, probe)
        except NotImplementedError:
            out.append(name)
        except Exception:
            # 已经开始写了但还跑不通，不算 pending
            pass
    return out


def implemented_ops() -> list[str]:
    """已经动过手的算子名列表（不代表测试通过，只表示 NotImplementedError 被去掉了）。"""
    return [n for n in OPS if n not in pending_ops()]
