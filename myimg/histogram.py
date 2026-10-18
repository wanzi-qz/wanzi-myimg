"""直方图与阈值算子。

口径：float 输入一律先 to_uint8（四舍五入+裁剪）再统计，
和 benchmarks/refs.py 里的 cv2.calcHist(round(img)) 对齐。
"""

from __future__ import annotations

import numpy as np

from ._core import as_float64, to_uint8

__all__ = ["histogram", "otsu_threshold", "hist_equalize", "clahe"]


def histogram(img: np.ndarray, bins: int = 256) -> np.ndarray:
    """灰度直方图，返回 (bins,) int64，counts.sum() == H*W。

    用 np.bincount 而不是 np.histogram：bincount 是整数分桶，没有浮点边界问题，
    而且快一个数量级。np.histogram 的最后一个桶右闭、其余左闭右开，
    整数分桶下这个区别不存在，但 float 输入下会咬人 —— 所以这里强制先转 uint8。
    """
    if bins < 1:
        raise ValueError(f"bins 必须 >= 1，收到 {bins}")
    u = to_uint8(as_float64(img))
    if bins == 256:
        return np.bincount(u.ravel(), minlength=256).astype(np.int64)
    idx = np.clip((u.astype(np.int64) * bins) // 256, 0, bins - 1)
    return np.bincount(idx.ravel(), minlength=bins).astype(np.int64)


def otsu_threshold(img: np.ndarray) -> float:
    """大津法：遍历 256 个候选阈值，取类间方差最大者。

    类间方差 σ²_B(t) = w0(t) * w1(t) * (μ0(t) - μ1(t))²
      w0 = 前类像素数占比，μ0 = 前类灰度均值，w1/μ1 同理。
    最大化 σ²_B 等价于最小化类内方差，这个等价性面试要能推。

    实现用递推：w0、sum0 随 t 累加，w1 = total - w0，sum1 = sum_all - sum0，
    整体 O(256)，不是每个 t 重算前缀和的 O(256²)。
    """
    hist = histogram(img, 256).astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return 0.0
    sum_all = float((np.arange(256) * hist).sum())

    w0 = 0.0
    sum0 = 0.0
    best_var = -1.0
    best_t = 0
    for t in range(256):
        w0 += hist[t]
        sum0 += t * hist[t]
        w1 = total - w0
        if w0 <= 0 or w1 <= 0:
            continue  # 某一类为空，类间方差无定义，跳过
        sum1 = sum_all - sum0
        mu0 = sum0 / w0
        mu1 = sum1 / w1
        var = w0 * w1 * (mu0 - mu1) ** 2
        if var > best_var:  # 严格大于 => 平局时取最小的 t（OpenCV 同口径）
            best_var = var
            best_t = t
    return float(best_t)


def hist_equalize(img: np.ndarray) -> np.ndarray:
    """全局直方图均衡：s = round(255 * (CDF(r) - CDF_min) / (total - CDF_min))。

    CDF_min 是最低出现灰阶的累积计数（即该灰阶的像素数）。减掉它才能让
    最低灰阶映射到 0，否则暗部会被整体抬亮。这是 Gonzalez 教材和 OpenCV 共同的口径。
    2026-10-18 实测：不减 CDF_min 的版本在 microscope 用例上和 cv2 差到 9 灰阶，
    减了之后逐像素一致（maxdiff 0）。

    退化情况：整张图只有一个灰阶时 total - CDF_min == 0，公式无定义，
    OpenCV 的行为是原样返回（实测 equalizeHist(常数图) == 原图），这里照做。

    映射必须单调不减（CDF 单调），测试会查这条。
    取整用 floor(x+0.5)（四舍五入）对齐 OpenCV 的 cvRound，
    不要用 np.round —— 那是银行家舍入，整数灰阶上会差 ±1。
    """
    u = to_uint8(as_float64(img))
    hist = np.bincount(u.ravel(), minlength=256).astype(np.float64)
    cdf = hist.cumsum()
    total = float(u.size)
    nz = np.nonzero(hist)[0]
    if nz.size == 0:
        return u
    cmin = cdf[nz[0]]
    denom = total - cmin
    if denom <= 0:
        return u  # 单一灰阶，OpenCV 原样返回
    lut = np.floor((cdf - cmin) * 255.0 / denom + 0.5)
    lut = np.clip(lut, 0, 255).astype(np.uint8)
    return lut[u]


def clahe(img: np.ndarray, clip_limit: float = 2.0, grid: int = 8) -> np.ndarray:
    """对比度受限自适应直方图均衡。

    三步：
      1. 分 tile，每个 tile 算直方图；
      2. 削峰：超过 clip_limit * (tile像素数/256) 的计数切掉，
         切掉的总量平均 redistributed 回 256 个桶，循环直到没有桶超限（或轮数上限）；
      3. 每个像素的映射由周围 4 个 tile 中心的 LUT 双线性插值得到，消除块状接缝。
    第 3 步是 CLAHE 相对朴素分块均衡的全部优势所在。
    """
    u = to_uint8(as_float64(img))
    H, W = u.shape
    if grid < 1:
        raise ValueError(f"grid 必须 >= 1，收到 {grid}")

    # tile 边界：尽量均分，余数摊到前面的 tile
    def edges(n: int, g: int) -> list[int]:
        base, rem = divmod(n, g)
        out = [0]
        for i in range(g):
            out.append(out[-1] + base + (1 if i < rem else 0))
        return out

    ys = edges(H, grid)
    xs = edges(W, grid)
    centers_y = [(ys[i] + ys[i + 1]) / 2.0 for i in range(grid)]
    centers_x = [(xs[j] + xs[j + 1]) / 2.0 for j in range(grid)]

    # 每个 tile 的均衡 LUT
    luts = np.zeros((grid, grid, 256), dtype=np.float64)
    for i in range(grid):
        for j in range(grid):
            tile = u[ys[i]:ys[i + 1], xs[j]:xs[j + 1]]
            hist = np.bincount(tile.ravel(), minlength=256).astype(np.float64)
            n_pix = float(tile.size)
            limit = clip_limit * n_pix / 256.0
            # 削峰 - 重分配 循环
            for _ in range(8):
                excess = np.clip(hist - limit, 0, None).sum()
                if excess <= 1e-9:
                    break
                hist = np.minimum(hist, limit)
                hist += excess / 256.0
            cdf = hist.cumsum()
            luts[i, j] = np.clip(np.floor(cdf * 255.0 / n_pix + 0.5), 0, 255)

    # 双线性插值：每个像素找它落在哪四个 tile 中心围成的格子里
    out = np.empty_like(u, dtype=np.float64)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)

    def band(coord: np.ndarray, centers: list[float], g: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # 返回 (i0, i1, w)：i0/i1 是插值用到的两个 tile 下标，w 是偏向 i1 的权重
        c = np.array(centers)
        pos = np.searchsorted(c, coord, side="right") - 1
        pos = np.clip(pos, 0, g - 2)
        c0 = c[pos]
        c1 = c[pos + 1]
        w = (coord - c0) / np.maximum(c1 - c0, 1e-9)
        w = np.clip(w, 0.0, 1.0)
        # 坐标在最外圈 tile 中心之外时，退化为只用最近的那个 tile
        outside_lo = coord < c[0]
        outside_hi = coord > c[-1]
        w = np.where(outside_lo, 0.0, w)
        w = np.where(outside_hi, 1.0, w)
        pos = np.where(outside_lo, 0, pos)
        pos = np.where(outside_hi, g - 2, pos)
        w = np.where(outside_hi, 1.0, w)
        return pos, pos + 1, w

    iy0, iy1, wy = band(yy, centers_y, grid)
    ix0, ix1, wx = band(xx, centers_x, grid)

    def L(iy, ix):
        return luts[iy, ix, u]

    top = L(iy0, ix0) * (1 - wx) + L(iy0, ix1) * wx
    bot = L(iy1, ix0) * (1 - wx) + L(iy1, ix1) * wx
    out = top * (1 - wy) + bot * wy
    return np.clip(np.floor(out + 0.5), 0, 255).astype(np.uint8)
