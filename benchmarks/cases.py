"""测试与对拍用的图像用例。

全部用固定随机种子合成，保证任何时候跑出来的输入都一样 —— 对拍结果不可复现的话，
误差数字就没有意义了。

真实图片：把 png/jpg 丢进 data/ 目录，load_real_images() 会自动收进来。
data/ 已在 .gitignore 里，不会跟着仓库上传。
"""

from __future__ import annotations

import pathlib
from typing import Callable

import numpy as np

SEED = 20261018
DEFAULT_SIZE = 128
DATA_DIR = pathlib.Path(__file__).resolve().parents[1] / "data"


def _rng(seed: int = SEED) -> np.random.Generator:
    return np.random.default_rng(seed)


def _mesh(size: int) -> tuple[np.ndarray, np.ndarray]:
    y, x = np.mgrid[0:size, 0:size].astype(np.float64)
    return x, y


# --- 灰度用例（给滤波 / 直方图 / 边缘） -----------------------------------

def flat(size: int = DEFAULT_SIZE) -> np.ndarray:
    """常数图。退化用例：直方图只有一个峰，Otsu / 均衡化在这里很容易出 nan。"""
    return np.full((size, size), 128, dtype=np.uint8)


def ramp(size: int = DEFAULT_SIZE) -> np.ndarray:
    """水平线性渐变 0..255。测边界填充和梯度方向。"""
    return np.linspace(0, 255, size, dtype=np.float64).reshape(1, -1).repeat(size, axis=0).astype(np.uint8)


def step_edge(size: int = DEFAULT_SIZE) -> np.ndarray:
    """左半 40 / 右半 200 的阶跃边缘。测中值滤波的保边性和 Sobel 的方向。"""
    img = np.full((size, size), 40, dtype=np.uint8)
    img[:, size // 2:] = 200
    return img


def impulse(size: int = DEFAULT_SIZE, density: float = 0.02) -> np.ndarray:
    """ramp 上叠椒盐噪声。中值滤波的标准考题。"""
    img = ramp(size).astype(np.float64)
    r = _rng(SEED + 1).random((size, size))
    img[r < density] = 0.0
    img[r > 1.0 - density] = 255.0
    return img.astype(np.uint8)


def gauss_noise(size: int = DEFAULT_SIZE, sigma: float = 20.0) -> np.ndarray:
    """smooth 结构 + 高斯噪声。高斯滤波的标准考题，也是课题场景的简化版。"""
    x, y = _mesh(size)
    base = 128 + 60 * np.sin(2 * np.pi * x / (size / 3)) * np.cos(2 * np.pi * y / (size / 2))
    base += 40 * np.exp(-((x - size * 0.3) ** 2 + (y - size * 0.6) ** 2) / (2 * (size * 0.12) ** 2))
    noisy = base + _rng(SEED + 2).normal(0, sigma, (size, size))
    return np.clip(noisy, 0, 255).astype(np.uint8)


def uneven_light(size: int = DEFAULT_SIZE) -> np.ndarray:
    """强光照梯度 + 一个小目标。Otsu 在这里会失效，用来写「什么时候不该用 Otsu」那一节。"""
    x, y = _mesh(size)
    light = np.linspace(30, 220, size).reshape(1, -1).repeat(size, axis=0)
    img = light.copy()
    cx, cy, r = size * 0.5, size * 0.5, size * 0.08
    img[(x - cx) ** 2 + (y - cy) ** 2 < r * r] = 120
    return np.clip(img + _rng(SEED + 3).normal(0, 6, (size, size)), 0, 255).astype(np.uint8)


def microscope(size: int = DEFAULT_SIZE) -> np.ndarray:
    """合成显微图：暗背景 + 荧光颗粒 + 光照不均 + 高斯噪声。

    和硕士课题的输入口径最接近的一个用例，报告里的效果图主要用它。
    """
    x, y = _mesh(size)
    img = np.full((size, size), 18.0)
    # 光照不均（中心亮四周暗）
    img += 26 * np.exp(-((x - size / 2) ** 2 + (y - size / 2) ** 2) / (2 * (size * 0.55) ** 2))
    # 荧光颗粒
    rng = _rng(SEED + 4)
    for _ in range(14):
        cx, cy = rng.uniform(0.1, 0.9, 2) * size
        rr = rng.uniform(size * 0.02, size * 0.05)
        amp = rng.uniform(90, 200)
        img += amp * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * rr * rr))
    # 噪声
    img += rng.normal(0, 22, (size, size))
    return np.clip(img, 0, 255).astype(np.uint8)


def random_uint8(size: int = DEFAULT_SIZE) -> np.ndarray:
    """纯随机图。最严苛的对拍输入：没有结构，任何实现差异都会暴露出来。"""
    return _rng(SEED + 5).integers(0, 256, (size, size), dtype=np.uint8)


# --- 二值用例（给形态学 / 连通域） ----------------------------------------

def blobs(size: int = DEFAULT_SIZE) -> np.ndarray:
    """若干圆斑 + 细小连接 + 孤立噪点。腐蚀膨胀、连通域的主用例。"""
    x, y = _mesh(size)
    m = np.zeros((size, size), dtype=bool)
    for cx, cy, r in [(0.25, 0.3, 0.12), (0.62, 0.35, 0.16), (0.4, 0.72, 0.1), (0.8, 0.78, 0.07)]:
        m[(x - cx * size) ** 2 + (y - cy * size) ** 2 < (r * size) ** 2] = True
    # 一条细连接（腐蚀应该能断开它）
    m[int(0.3 * size), int(0.36 * size):int(0.48 * size)] = True
    # 孤立单像素噪点
    rng = _rng(SEED + 6)
    pts = rng.integers(0, size, (12, 2))
    m[pts[:, 0], pts[:, 1]] = True
    return (m.astype(np.uint8) * 255)


def holes(size: int = DEFAULT_SIZE) -> np.ndarray:
    """实心块内部挖小孔。闭运算的主用例。"""
    m = np.zeros((size, size), dtype=bool)
    m[size // 4: size * 3 // 4, size // 4: size * 3 // 4] = True
    for cx, cy, r in [(0.4, 0.4, 0.03), (0.6, 0.55, 0.04), (0.45, 0.65, 0.02)]:
        x, y = np.mgrid[0:size, 0:size]
        m[(x - cx * size) ** 2 + (y - cy * size) ** 2 < (r * size) ** 2] = False
    return (m.astype(np.uint8) * 255)


def checker(size: int = DEFAULT_SIZE, cell: int = 8) -> np.ndarray:
    """棋盘格。4 连通和 8 连通在这里给出完全不同的结果，专门用来测连通性口径。"""
    x, y = _mesh(size)
    return ((((x // cell) + (y // cell)) % 2) * 255).astype(np.uint8)


def two_touching(size: int = DEFAULT_SIZE) -> np.ndarray:
    """两个相切的圆。连通域计数的边界用例：相切点算不算连通，取决于连通性定义。"""
    x, y = _mesh(size)
    r = size * 0.18
    m = ((x - size * 0.32) ** 2 + (y - size * 0.5) ** 2 < r * r)
    m |= ((x - size * 0.68) ** 2 + (y - size * 0.5) ** 2 < r * r)
    return (m.astype(np.uint8) * 255)


# --- 注册表 ---------------------------------------------------------------

GRAY_CASES: dict[str, Callable[..., np.ndarray]] = {
    "flat": flat,
    "ramp": ramp,
    "step_edge": step_edge,
    "impulse": impulse,
    "gauss_noise": gauss_noise,
    "uneven_light": uneven_light,
    "microscope": microscope,
    "random_uint8": random_uint8,
}

BINARY_CASES: dict[str, Callable[..., np.ndarray]] = {
    "blobs": blobs,
    "holes": holes,
    "checker": checker,
    "two_touching": two_touching,
}


def make(name: str, size: int = DEFAULT_SIZE) -> np.ndarray:
    """按名字取用例。"""
    if name in GRAY_CASES:
        return GRAY_CASES[name](size)
    if name in BINARY_CASES:
        return BINARY_CASES[name](size)
    raise KeyError(f"未知用例 {name!r}，可选：{sorted(list(GRAY_CASES) + list(BINARY_CASES))}")


def load_real_images(max_side: int = 512) -> dict[str, np.ndarray]:
    """data/ 下的真实图片，转灰度并缩到 max_side 以内。没有就返回空 dict。

    对拍阶段用合成图就够（可复现、无需下载、体积小）；
    真实图只在报告里出效果对比图时用，别拿它当数值对拍的输入 ——
    真实图一旦换文件，所有误差数字都得重跑。
    """
    out: dict[str, np.ndarray] = {}
    if not DATA_DIR.exists():
        return out
    try:
        import cv2
    except ImportError:
        return out
    for p in sorted(DATA_DIR.iterdir()):
        if p.suffix.lower() not in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
            continue
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        h, w = img.shape
        if max(h, w) > max_side:
            s = max_side / max(h, w)
            img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        out[p.stem] = img
    return out
