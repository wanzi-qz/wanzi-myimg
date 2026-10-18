"""滤波算子。

每个算子给两份实现：
  xxx_naive  双层 for 循环的朴素版，慢但一眼能看懂在算什么
  xxx        向量化版，测试和对拍用这个
两份都保留是因为 benchmarks/run_compare.py 要测它们的耗时差，
「为什么向量化值得」这件事用数字说比用形容词说有说服力。

口径（2026-10-18 实测）：np.pad(mode='reflect') == cv2.BORDER_REFLECT_101，
可分离高斯对 cv2.GaussianBlur 最大误差 5.68e-14。
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from ._core import as_float64, odd_kernel_size, pad

__all__ = [
    "box_filter", "box_filter_naive", "box_filter_integral",
    "gaussian_filter", "gaussian_filter_naive", "gaussian_kernel1d",
    "median_filter", "median_filter_naive",
]


# ---------------------------------------------------------------------------
# 均值滤波
# ---------------------------------------------------------------------------
def box_filter_naive(img: np.ndarray, ksize: int = 3, mode: str = "reflect") -> np.ndarray:
    """朴素版：每个像素收集 k^2 个邻居求平均。O(H*W*k^2)。"""
    src = as_float64(img)
    k = odd_kernel_size(ksize)
    r = k // 2
    padded = pad(src, r, mode)
    H, W = src.shape
    out = np.empty((H, W), dtype=np.float64)
    for i in range(H):
        for j in range(W):
            out[i, j] = padded[i:i + k, j:j + k].mean()
    return out


def box_filter(img: np.ndarray, ksize: int = 3, mode: str = "reflect") -> np.ndarray:
    """向量化版：滑动窗口视图直接对最后两轴求均值。O(H*W*k^2) 但没有 Python 循环。"""
    src = as_float64(img)
    k = odd_kernel_size(ksize)
    r = k // 2
    padded = pad(src, r, mode)
    windows = sliding_window_view(padded, (k, k))
    return windows.mean(axis=(-2, -1))


def box_filter_integral(img: np.ndarray, ksize: int = 3, mode: str = "reflect") -> np.ndarray:
    """积分图版：O(H*W)，与 k 无关。OpenCV 的 boxFilter 内部就是这个思路。

    积分图 I 满足 I[i,j] = 原图左上角 (0,0)..(i,j) 的和，
    任意矩形窗的和 = 四个角的积分图值做两次差分。
    注意 dtype：积分值会累加到 H*W*255 量级，必须用 float64，int32 会溢出。
    """
    src = as_float64(img)
    k = odd_kernel_size(ksize)
    r = k // 2
    padded = pad(src, r, mode)
    H, W = src.shape
    # 前缀和：先补一行一列 0，避免处理边界特例
    ii = np.zeros((padded.shape[0] + 1, padded.shape[1] + 1), dtype=np.float64)
    ii[1:, 1:] = padded.cumsum(axis=0).cumsum(axis=1)
    # 窗口 [i, i+k) x [j, j+k) 在 padded 里的和
    y2 = np.arange(k, k + H)
    x2 = np.arange(k, k + W)
    total = (ii[np.ix_(y2, x2)]
             - ii[np.ix_(y2 - k, x2)]
             - ii[np.ix_(y2, x2 - k)]
             + ii[np.ix_(y2 - k, x2 - k)])
    return total / (k * k)


# ---------------------------------------------------------------------------
# 高斯滤波
# ---------------------------------------------------------------------------
def gaussian_kernel1d(sigma: float, ksize: int | None = None) -> np.ndarray:
    """一维高斯核，归一化到 sum == 1。

    ksize 为 None 时按 OpenCV 的约定从 sigma 推。

    注意：网上普遍说 OpenCV 用 ±3σ 截断（ksize = 2*ceil(3σ)+1），这是错的。
    2026-10-18 在本机逐个 sigma 实测对拍，OpenCV 的自动核尺寸是
        ksize = 2 * round_half_up(4 * sigma) + 1        （±4σ 截断）
    实测命中表：sigma 0.3/0.5/0.8/1.0/1.4/1.5/2.0/2.5/3.0 -> ksize 3/5/7/9/13/13/17/21/25。
    用 ±3σ 的话 sigma=1.0 会得到 7 而 OpenCV 是 9，截断尾巴差出 2e-2，对拍直接不过。

    取整用 floor(x+0.5) 而不是 Python 的 round()：Python round 是银行家舍入
    （round(4.5)=4），OpenCV 的 cvRound 是四舍五入（=5），4σ 落在 .5 上时两者分叉。
    """
    if sigma <= 0:
        raise ValueError(f"sigma 必须 > 0，收到 {sigma}")
    if ksize is None:
        ksize = 2 * int(np.floor(4.0 * sigma + 0.5)) + 1
    ksize = odd_kernel_size(ksize)
    r = ksize // 2
    x = np.arange(-r, r + 1, dtype=np.float64)
    kern = np.exp(-(x * x) / (2.0 * sigma * sigma))
    return kern / kern.sum()


def _conv1d_axis(arr: np.ndarray, kern: np.ndarray, axis: int, mode: str) -> np.ndarray:
    """沿 axis 做一维相关（核对称时相关 == 卷积）。

    只沿目标轴填充。两个方向都填会让另一轴的尺寸多出一圈 padding，
    第二次卷积之后尺寸就对不回去了 —— 可分离实现最容易踩的就是这个。
    """
    k = kern.size
    r = k // 2
    pad_width = [(0, 0)] * arr.ndim
    pad_width[axis] = (r, r)
    padded = np.pad(arr, pad_width, mode=mode)
    win = sliding_window_view(padded, k, axis=axis)
    return win @ kern


def gaussian_filter_naive(img: np.ndarray, sigma: float = 1.0, ksize: int | None = None,
                          mode: str = "reflect") -> np.ndarray:
    """朴素版：构造二维核直接双循环卷积。O(H*W*k^2)。"""
    src = as_float64(img)
    k1 = gaussian_kernel1d(sigma, ksize)
    k2 = np.outer(k1, k1)  # 二维核 = 一维核外积
    r = k1.size // 2
    padded = pad(src, r, mode)
    H, W = src.shape
    out = np.empty((H, W), dtype=np.float64)
    for i in range(H):
        for j in range(W):
            out[i, j] = (padded[i:i + 2 * r + 1, j:j + 2 * r + 1] * k2).sum()
    return out


def gaussian_filter(img: np.ndarray, sigma: float = 1.0, ksize: int | None = None,
                    mode: str = "reflect") -> np.ndarray:
    """向量化版：可分离，先按列卷一次再按行卷一次。O(H*W*2k)。

    高斯函数是唯一可分离的旋转对称核，这是它比中值滤波快的根本原因。
    """
    src = as_float64(img)
    kern = gaussian_kernel1d(sigma, ksize)
    tmp = _conv1d_axis(src, kern, axis=1, mode=mode)
    return _conv1d_axis(tmp, kern, axis=0, mode=mode)


# ---------------------------------------------------------------------------
# 中值滤波
# ---------------------------------------------------------------------------
def median_filter_naive(img: np.ndarray, ksize: int = 3, mode: str = "edge") -> np.ndarray:
    """朴素版：每个窗口排序取中间。O(H*W*k^2*log k)。"""
    src = as_float64(img)
    k = odd_kernel_size(ksize)
    r = k // 2
    padded = pad(src, r, mode)
    H, W = src.shape
    out = np.empty((H, W), dtype=np.float64)
    for i in range(H):
        for j in range(W):
            out[i, j] = np.median(padded[i:i + k, j:j + k])
    return out


def median_filter(img: np.ndarray, ksize: int = 3, mode: str = "edge") -> np.ndarray:
    """向量化版：滑动窗口 + np.median。

    中值不可分离，所以复杂度下不去；但去掉 Python 循环已经快两个数量级。
    边界默认 'edge'（REPLICATE），与 cv2.medianBlur 实测一致。
    进阶的 Huang 直方图法（O(H*W*256) 与 k 无关）留作练习，见 README。
    """
    src = as_float64(img)
    k = odd_kernel_size(ksize)
    r = k // 2
    padded = pad(src, r, mode)
    windows = sliding_window_view(padded, (k, k))
    return np.median(windows, axis=(-2, -1))
