"""形态学与连通域算子。

腐蚀 = 邻域取最小，膨胀 = 邻域取最大 —— 对二值图这就是全部定义，
而且天然推广到灰度形态学。实现直接用这个视角。

边界口径（重要）：cv2.erode 默认边界是 +inf、cv2.dilate 是 -inf，不是 0。
道理：腐蚀要求邻域全前景，边界外应假设「有利于保留」，否则贴边物体会被凭空腐蚀一圈。
所以腐蚀 pad True、膨胀 pad False。搞反了错误只出现在图像四周，很难发现。
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from ._core import check_gray, odd_kernel_size

__all__ = ["erode", "dilate", "morph_open", "morph_close", "label_components", "region_stats"]


def _to_bool(mask: np.ndarray) -> np.ndarray:
    check_gray(mask)
    if mask.dtype == bool:
        return mask.copy()
    if mask.dtype == np.uint8:
        return mask > 0
    return mask > 0


def _from_bool(b: np.ndarray) -> np.ndarray:
    return (b.astype(np.uint8)) * 255


def erode(mask: np.ndarray, ksize: int = 3, iterations: int = 1) -> np.ndarray:
    """腐蚀：邻域全前景才保留。边界 pad True（对应 cv2 的 +inf 边界）。"""
    b = _to_bool(mask)
    k = odd_kernel_size(ksize)
    r = k // 2
    for _ in range(max(1, iterations)):
        padded = np.pad(b, r, mode="constant", constant_values=True)
        b = sliding_window_view(padded, (k, k)).all(axis=(-2, -1))
    return _from_bool(b)


def dilate(mask: np.ndarray, ksize: int = 3, iterations: int = 1) -> np.ndarray:
    """膨胀：邻域有前景即置前景。边界 pad False（对应 cv2 的 -inf 边界）。"""
    b = _to_bool(mask)
    k = odd_kernel_size(ksize)
    r = k // 2
    for _ in range(max(1, iterations)):
        padded = np.pad(b, r, mode="constant", constant_values=False)
        b = sliding_window_view(padded, (k, k)).any(axis=(-2, -1))
    return _from_bool(b)


def morph_open(mask: np.ndarray, ksize: int = 3) -> np.ndarray:
    """开运算 = 先腐蚀后膨胀。去小于结构元的亮噪点，保持大物体尺寸。"""
    return dilate(erode(mask, ksize), ksize)


def morph_close(mask: np.ndarray, ksize: int = 3) -> np.ndarray:
    """闭运算 = 先膨胀后腐蚀。填小于结构元的孔洞和裂缝。"""
    return erode(dilate(mask, ksize), ksize)


# ---------------------------------------------------------------------------
# 连通域
# ---------------------------------------------------------------------------
class _UnionFind:
    """并查集，带路径压缩 + 按秩合并。两遍扫描连通域标记的核心数据结构。"""

    def __init__(self) -> None:
        self.parent: list[int] = []
        self.rank: list[int] = []

    def add(self) -> int:
        self.parent.append(len(self.parent))
        self.rank.append(0)
        return len(self.parent) - 1

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # 路径压缩
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> int:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return ra
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return ra


def label_components(mask: np.ndarray, connectivity: int = 8) -> tuple[np.ndarray, int]:
    """两遍扫描 + 并查集的连通域标记。

    第一遍：给前景像素编临时标签，记录等价关系；
    第二遍：用并查集合并等价标签并重编号为 1..n（背景 0）。

    编号顺序不保证和 cv2 一致（cv2 按扫描顺序），
    所以对拍比的是「个数」和「面积的多重集合」，不是 labels 数组本身。
    """
    if connectivity not in (4, 8):
        raise ValueError(f"connectivity 必须是 4 或 8，收到 {connectivity}")
    b = _to_bool(mask)
    H, W = b.shape
    labels = np.zeros((H, W), dtype=np.int32)
    uf = _UnionFind()

    if connectivity == 8:
        neigh = ((-1, -1), (-1, 0), (-1, 1), (0, -1))
    else:
        neigh = ((-1, 0), (0, -1))

    for i in range(H):
        for j in range(W):
            if not b[i, j]:
                continue
            found = []
            for di, dj in neigh:
                ii, jj = i + di, j + dj
                if 0 <= ii < H and 0 <= jj < W and labels[ii, jj] > 0:
                    found.append(labels[ii, jj])
            if not found:
                labels[i, j] = uf.add() + 1  # 标签从 1 开始，0 留给背景
            else:
                mn = min(found)
                labels[i, j] = mn
                for other in found:
                    uf.union(mn - 1, other - 1)

    # 第二遍：合并等价标签并重编号
    remap: dict[int, int] = {}
    out = np.zeros((H, W), dtype=np.int32)
    for i in range(H):
        row = labels[i]
        for j in range(W):
            v = row[j]
            if v == 0:
                continue
            root = uf.find(v - 1) + 1
            new = remap.get(root)
            if new is None:
                new = len(remap) + 1
                remap[root] = new
            out[i, j] = new
    return out, len(remap)


def region_stats(mask: np.ndarray, connectivity: int = 8) -> list[dict]:
    """连通域属性表：面积 / 外接框 / 质心，按面积降序。

    建立在 label_components 之上，不重复实现连通域逻辑。
    返回 list[dict] 是为了可读；几千个连通域的产线场景应换成结构化数组，
    这个局限写进报告比假装没这回事更可信。
    """
    labels, n = label_components(mask, connectivity)
    if n == 0:
        return []
    flat = labels.ravel()
    areas = np.bincount(flat, minlength=n + 1)
    ys, xs = np.nonzero(labels)
    labs = labels[ys, xs]
    out = []
    for lab in range(1, n + 1):
        sel = labs == lab
        yy = ys[sel]; xx = xs[sel]
        out.append({
            "label": int(lab),
            "area": int(areas[lab]),
            "x": int(xx.min()), "y": int(yy.min()),
            "w": int(xx.max() - xx.min() + 1), "h": int(yy.max() - yy.min() + 1),
            "cx": float(xx.mean()), "cy": float(yy.mean()),
        })
    return sorted(out, key=lambda d: -d["area"])
