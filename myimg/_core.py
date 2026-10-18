"""myimg 公共基础设施：dtype 归一、边界填充、输入校验。

算法主体不在这里。去 filters.py / histogram.py / edges.py / morphology.py 手写。

本文件已经写好，可以直接用，也建议你读懂每一行——
里面 pad() 和 as_float64() 的取舍，正是各算子和 OpenCV 对不上误差时最常见的两个原因。
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "SUPPORTED_PAD_MODES",
    "PAD_TO_CV2",
    "as_float64",
    "to_uint8",
    "check_gray",
    "pad",
    "odd_kernel_size",
    "MYIMG_EPS",
]

# 浮点对拍的默认容差。同口径下纯 NumPy 与 OpenCV 的差应该在 1e-10 量级，
# 取 1e-8 是留出累加顺序不同的余量；超过这个数说明实现有实质差异，不是浮点噪声。
MYIMG_EPS = 1e-8

# ---------------------------------------------------------------------------
# 边界模式映射（2026-10-18 本机实测核对）
# numpy 1.26.4 / opencv-python 4.10.0 / scikit-image 0.25.2 / scipy 1.13.1
#
#   np.pad(mode=)   cv2 borderType          scipy/skimage mode
#   -------------   --------------------    ------------------
#   reflect         BORDER_REFLECT_101      mirror
#   symmetric       BORDER_REFLECT          reflect      <-- 注意，跟 numpy 不是一个意思
#   edge            BORDER_REPLICATE        nearest
#   wrap            BORDER_WRAP             wrap
#   constant        BORDER_CONSTANT         constant
# ---------------------------------------------------------------------------
SUPPORTED_PAD_MODES = ("reflect", "symmetric", "edge", "wrap", "constant")

PAD_TO_CV2 = {
    "reflect": "BORDER_REFLECT_101",
    "symmetric": "BORDER_REFLECT",
    "edge": "BORDER_REPLICATE",
    "wrap": "BORDER_WRAP",
    "constant": "BORDER_CONSTANT",
}


def check_gray(img: np.ndarray) -> np.ndarray:
    """校验是二维灰度图，返回原数组（不复制）。

    三通道的直接报错，不做自动转灰度——转换公式（BT.601 / BT.709）本身就是一个
    需要说清楚的口径问题，藏在工具函数里以后会咬人。要转请在调用前自己用
    ``0.299R + 0.587G + 0.114B`` 明确写出来。
    """
    if not isinstance(img, np.ndarray):
        raise TypeError(f"输入必须是 np.ndarray，收到 {type(img).__name__}")
    if img.ndim != 2:
        raise ValueError(
            f"只支持二维灰度图 (H, W)，收到 ndim={img.ndim}, shape={img.shape}。"
            " 彩色图请先显式转灰度。"
        )
    if img.size == 0:
        raise ValueError("输入图像为空")
    return img


def as_float64(img: np.ndarray) -> np.ndarray:
    """转成 float64 计算副本，值域保持原始尺度（uint8 输入 → 0..255，不缩放到 0..1）。

    为什么统一用 float64 而不是 float32：对拍时 OpenCV 的 CV_64F 路径是 float64，
    用 float32 会引入 1e-5 量级的假误差，把真正的实现 bug 淹掉。
    性能不是这个仓库的目标，可复现的对拍才是。

    为什么不做 0..1 归一化：OpenCV 对 uint8 输入直接在 0..255 上算，
    归一化会让「和 cv2 逐像素对拍」这一步多一次乘除，反而容易在取整处出错。
    需要 0..1 的场合（比如和 skimage 对拍）在 benchmarks/refs.py 里单独转。
    """
    check_gray(img)
    if img.dtype == np.float64:
        return img.copy()
    return img.astype(np.float64)


def to_uint8(arr: np.ndarray, clip: bool = True) -> np.ndarray:
    """float 数组转回 uint8，先裁剪到 [0, 255] 再四舍五入。

    clip=False 时直接取模溢出，用来暴露中间结果越界的 bug——
    对拍时如果发现 uint8 结果差得很离谱，先试 clip=False 看看是不是值域炸了。

    注意 np.round 用的是「银行家舍入」（round half to even），
    而 OpenCV 内部是 round half away from zero。整数值上这会造成 ±1 灰阶差，
    这是真实存在的口径差异，不是 bug，直方图均衡那一节的报告要写到。
    """
    if clip:
        arr = np.clip(arr, 0.0, 255.0)
    return np.round(arr).astype(np.uint8)


def pad(img: np.ndarray, width: int, mode: str = "reflect", constant_value: float = 0.0) -> np.ndarray:
    """四边填充 width 个像素，只接受 SUPPORTED_PAD_MODES 里的模式名。

    薄封装 np.pad 的唯一理由：强制你在调用处写明用哪种边界，
    而不是让 numpy 的默认值（'constant'）悄悄决定结果。
    图像算子和 OpenCV 对不上，一半的原因是边界模式默认值不一样。

    腐蚀/膨胀要用不同填充值，走 constant_value 或者自己在 morphology.py 里处理：
    cv2.erode 的默认边界是 +inf，cv2.dilate 是 -inf，不是 0。
    """
    if mode not in SUPPORTED_PAD_MODES:
        raise ValueError(f"mode 必须是 {SUPPORTED_PAD_MODES} 之一，收到 {mode!r}")
    if width < 0:
        raise ValueError(f"width 必须 >= 0，收到 {width}")
    if width == 0:
        return img
    if mode == "constant":
        return np.pad(img, width, mode="constant", constant_values=constant_value)
    return np.pad(img, width, mode=mode)


def odd_kernel_size(k: int, name: str = "ksize") -> int:
    """把偶数核尺寸向上取到奇数并给出提示，或直接报错。

    中值滤波必须是奇数方形窗（中心像素要参与投票）。
    高斯可以从 sigma 推核尺寸，但 OpenCV 的自动核尺寸不是网上流传的 ±3σ ——
    2026-10-18 本机逐个 sigma 对拍 cv2.getGaussianKernel 实测，约定是
        ``ksize = 2 * floor(4 * sigma + 0.5) + 1``      （±4σ 截断）
    命中表和推导写在 filters.gaussian_kernel1d 的注释里。本函数只管奇偶和下界校验，
    从 sigma 推 ksize 的逻辑离核最近，放在那儿最好查。
    """
    k = int(k)
    if k < 1:
        raise ValueError(f"{name} 必须 >= 1，收到 {k}")
    if k % 2 == 0:
        raise ValueError(f"{name} 必须是奇数，收到 {k}（中心像素要参与运算）")
    return k
