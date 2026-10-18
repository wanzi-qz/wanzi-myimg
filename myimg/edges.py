"""边缘检测算子。

Sobel 用可分离实现；Canny 是四步流水线（平滑 -> 梯度 -> 非极大值抑制 -> 双阈值滞后连接）。
"""

from __future__ import annotations

import numpy as np

from ._core import as_float64, pad
from .filters import gaussian_filter

__all__ = ["sobel", "sobel_naive", "canny"]


def sobel(img: np.ndarray, mode: str = "reflect") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sobel：返回 (gx, gy, magnitude)。

    核约定（OpenCV 口径，相关而非卷积）：
        gx = [[-1, 0, 1],      gy = gx.T = [[-1,-2,-1],
              [-2, 0, 2],                    [ 0, 0, 0],
              [-1, 0, 1]]                    [ 1, 2, 1]]
    可分离：gx = 垂直平滑[1,2,1] 后做水平差分[-1,0,1]；gy 反之。
    所以 gx[i,j] = SY[i, j+1] - SY[i, j-1]，其中 SY 是垂直平滑结果。

    中间结果必须是 float：uint8 上算会把负梯度截断成 0，这是最常见的坑。
    """
    src = as_float64(img)
    H, W = src.shape
    P = pad(src, 1, mode)

    # 垂直平滑（对行），覆盖所有 padded 列 -> (H, W+2)
    SY = P[0:H, :] + 2.0 * P[1:H + 1, :] + P[2:H + 2, :]
    # 水平平滑（对列），覆盖所有 padded 行 -> (H+2, W)
    SX = P[:, 0:W] + 2.0 * P[:, 1:W + 1] + P[:, 2:W + 2]

    gx = SY[:, 2:W + 2] - SY[:, 0:W]      # 右 - 左
    gy = SX[2:H + 2, :] - SX[0:H, :]      # 下 - 上
    mag = np.sqrt(gx * gx + gy * gy)
    return gx, gy, mag


def sobel_naive(img: np.ndarray, mode: str = "reflect") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """朴素版：显式写出 3x3 核双循环相关。慢，但和核的对应关系一目了然。"""
    KX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    KY = KX.T
    src = as_float64(img)
    H, W = src.shape
    P = pad(src, 1, mode)
    gx = np.empty((H, W)); gy = np.empty((H, W))
    for i in range(H):
        for j in range(W):
            win = P[i:i + 3, j:j + 3]
            gx[i, j] = (win * KX).sum()
            gy[i, j] = (win * KY).sum()
    return gx, gy, np.sqrt(gx * gx + gy * gy)


def _dilate8(m: np.ndarray) -> np.ndarray:
    """二值 mask 的 8 连通膨胀（纯切片实现，用于滞后连接的扩散）。"""
    H, W = m.shape
    P = np.pad(m, 1, mode="constant", constant_values=False)
    out = m.copy()
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            out |= P[1 + di:1 + di + H, 1 + dj:1 + dj + W]
    return out


def canny(img: np.ndarray, low: float, high: float, sigma: float = 0.0) -> np.ndarray:
    """Canny 边缘检测，输出 uint8 二值图（0 / 255）。

    四步：
      1. 高斯平滑（sigma>0 时；sigma=0 表示不平滑，对齐 cv2.Canny —— OpenCV 的 Canny
         内部不做独立的高斯预平滑，只靠 Sobel 核自带的 [1,2,1] 平滑。要对拍就传 0。）
      2. Sobel 求 gx/gy/幅值/方向
      3. 非极大值抑制：把梯度方向量化到 0/45/90/135 四个扇区，只保留沿梯度方向的局部极大。
         （OpenCV 用的是亚像素插值版 NMS，量化版会略粗，差异写进报告。）
      4. 双阈值 + 滞后连接：>high 为强边；low~high 为弱边，仅当 8 连通到强边时保留。
         滞后连接这里用「反复膨胀强边并与候选集求交」直到不动点实现，
         等价于 BFS 扩散但完全向量化。
    """
    src = as_float64(img)
    if sigma > 0:
        src = gaussian_filter(src, sigma)
    gx, gy, mag = sobel(src)
    H, W = mag.shape

    # --- 非极大值抑制 ---
    deg = np.degrees(np.arctan2(gy, gx)) % 180.0
    sector = ((deg + 22.5) // 45).astype(np.int64) % 4  # 0:0° 1:45° 2:90° 3:135°
    M = np.pad(mag, 1, mode="constant", constant_values=0.0)
    # 八个邻居（padded 坐标下中心为 (i+1, j+1)）
    E = M[1:H + 1, 2:W + 2]; W_ = M[1:H + 1, 0:W]
    N = M[0:H, 1:W + 1]; S = M[2:H + 2, 1:W + 1]
    NE = M[0:H, 2:W + 2]; SW = M[2:H + 2, 0:W]
    NW = M[0:H, 0:W]; SE = M[2:H + 2, 2:W + 2]

    n1 = np.empty_like(mag); n2 = np.empty_like(mag)
    m0 = sector == 0
    m1 = sector == 1
    m2 = sector == 2
    m3 = sector == 3
    n1[m0], n2[m0] = E[m0], W_[m0]
    n1[m1], n2[m1] = NE[m1], SW[m1]
    n1[m2], n2[m2] = N[m2], S[m2]
    n1[m3], n2[m3] = NW[m3], SE[m3]
    # 平局处理：n1 侧大于等于、n2 侧严格大于（一松一严）。
    # 完美阶跃边缘上相邻两列幅值相等：两侧都用 >= 会保留两像素宽的边；
    # 两侧都用 > 会整条边都被压掉；一松一严才细化成单像素。
    # 松紧放在哪一侧决定边落在阶跃的左列还是右列 —— 实测 cv2 保留左列（n1=E 侧松），
    # 放反了 step_edge 用例 IoU 直接是 0（像素数相同但位置错开一列）。
    keep = (mag >= n1) & (mag > n2) & (mag > 0)
    mag_nms = np.where(keep, mag, 0.0)

    # --- 双阈值 + 滞后连接 ---
    strong = mag_nms >= high
    weak = (mag_nms >= low) & (mag_nms < high)
    candidate = strong | weak
    reached = strong.copy()
    while True:
        grown = _dilate8(reached) & candidate
        if not grown.any() or np.array_equal(grown | reached, reached):
            reached = reached | grown
            break
        reached = reached | grown
    return (reached.astype(np.uint8)) * 255
