"""对拍基准：把 OpenCV / scikit-image 的调用封装成和 myimg 同签名的函数。

这里允许 import cv2 / skimage —— 它们是基准，不是实现。
所有口径换算都集中在这一个文件里，别的地方不许出现 cv2。

每个 ref 函数返回的 dtype / 值域已经对齐 myimg 的契约，测试里直接逐像素比。
"""

from __future__ import annotations

import numpy as np

import cv2
from skimage import exposure, filters, measure

from .cases import GRAY_CASES, BINARY_CASES  # noqa: F401  (供 run_compare 复用)

_CV2_BORDER = {
    "reflect": cv2.BORDER_REFLECT_101,
    "symmetric": cv2.BORDER_REFLECT,
    "edge": cv2.BORDER_REPLICATE,
    "wrap": cv2.BORDER_WRAP,
    "constant": cv2.BORDER_CONSTANT,
}


def _u8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    return np.clip(np.round(img), 0, 255).astype(np.uint8)


def _f64(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float64)


# --- 滤波 -----------------------------------------------------------------

def box_ref(img, ksize=3, mode="reflect"):
    src = _u8(img)
    if mode == "wrap":
        # cv2.boxFilter 不接受 BORDER_WRAP（opencv-python 4.10.0 实测直接抛
        # cv2.error），所以环绕边界由 numpy 补出来，再让 cv2 在补好的图上做
        # 常数边界的均值，最后裁回原尺寸。均值这一步仍然由 OpenCV 独立算，
        # numpy 只负责搬边界口径，不算「自己给自己当基准」。
        rad = ksize // 2
        padded = np.pad(src, rad, mode="wrap")
        out = cv2.boxFilter(
            padded, ddepth=cv2.CV_64F, ksize=(ksize, ksize),
            normalize=True, borderType=cv2.BORDER_CONSTANT,
        )
        return out[rad:rad + src.shape[0], rad:rad + src.shape[1]]
    return cv2.boxFilter(
        src, ddepth=cv2.CV_64F, ksize=(ksize, ksize),
        normalize=True, borderType=_CV2_BORDER[mode],
    )


def gaussian_ref(img, sigma=1.0, ksize=None, mode="reflect"):
    k = 0 if ksize is None else int(ksize)
    # GaussianBlur 不接受 ddepth 参数，输出深度跟随输入，所以要喂 float64 才拿得到无损结果
    return cv2.GaussianBlur(
        _f64(img), (k, k), sigmaX=sigma, sigmaY=sigma, borderType=_CV2_BORDER[mode],
    )


def median_ref(img, ksize=3, mode="edge"):
    # cv2.medianBlur 只接受 uint8（ksize<=5 时也接受 float32），边界固定 REPLICATE
    return cv2.medianBlur(_u8(img), ksize).astype(np.float64)


# --- 直方图 / 阈值 ----------------------------------------------------------

def histogram_ref(img, bins=256):
    hist = cv2.calcHist([_u8(img)], [0], None, [bins], [0, 256])
    return hist.ravel().astype(np.int64)


def otsu_ref(img):
    thr, _ = cv2.threshold(_u8(img), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(thr)


def hist_equalize_ref(img):
    return cv2.equalizeHist(_u8(img)).astype(np.float64)


def clahe_ref(img, clip_limit=2.0, grid=8):
    cl = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid, grid))
    return cl.apply(_u8(img)).astype(np.float64)


# --- 边缘 -----------------------------------------------------------------

def sobel_ref(img, mode="reflect"):
    src = _u8(img)
    gx = cv2.Sobel(src, cv2.CV_64F, 1, 0, ksize=3, borderType=_CV2_BORDER[mode])
    gy = cv2.Sobel(src, cv2.CV_64F, 0, 1, ksize=3, borderType=_CV2_BORDER[mode])
    mag = cv2.magnitude(gx, gy)
    return gx, gy, mag


def canny_ref(img, low, high, sigma=1.0):
    return cv2.Canny(_u8(img), low, high, L2gradient=True)


def canny_ref_smoothed(img, low, high, sigma=1.4):
    """带预平滑的对照基准：cv2.Canny 自己没有 sigma 参数，
    工业用法都是先 GaussianBlur 再 Canny，这里把两步显式写出来。"""
    src = cv2.GaussianBlur(_u8(img), (0, 0), sigmaX=sigma, sigmaY=sigma)
    return cv2.Canny(src, low, high, L2gradient=True)


# --- 形态学 / 连通域 ---------------------------------------------------------

def _struct(ksize):
    return np.ones((ksize, ksize), np.uint8)


def erode_ref(mask, ksize=3, iterations=1):
    return cv2.erode(_u8(mask), _struct(ksize), iterations=iterations).astype(np.float64)


def dilate_ref(mask, ksize=3, iterations=1):
    return cv2.dilate(_u8(mask), _struct(ksize), iterations=iterations).astype(np.float64)


def morph_open_ref(mask, ksize=3):
    return cv2.morphologyEx(_u8(mask), cv2.MORPH_OPEN, _struct(ksize)).astype(np.float64)


def morph_close_ref(mask, ksize=3):
    return cv2.morphologyEx(_u8(mask), cv2.MORPH_CLOSE, _struct(ksize)).astype(np.float64)


def label_ref(mask, connectivity=8):
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(
        _u8(mask), connectivity=connectivity
    )
    # labels 的 0 是背景，1..n-1 是连通域
    return labels.astype(np.int32), int(n - 1), stats, centroids


def region_stats_ref(mask, connectivity=8):
    labels, n, stats, centroids = label_ref(mask, connectivity)
    out = []
    for i in range(1, n + 1):
        x, y, w, h, area = stats[i]
        out.append({
            "label": int(i),
            "area": int(area),
            "x": int(x), "y": int(y), "w": int(w), "h": int(h),
            "cx": float(centroids[i][0]), "cy": float(centroids[i][1]),
        })
    return sorted(out, key=lambda d: -d["area"])
