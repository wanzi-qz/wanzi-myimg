"""形态学与连通域 vs OpenCV。

这一族要求逐像素精确：min/max 滤波没有浮点累加顺序问题，
对不上就是边界填充值或连通性口径错了。
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import assert_close, call

from benchmarks import cases, refs
from myimg import morphology as M

TOL = 0.0


def interior_hole_sizes(img: np.ndarray) -> list[int]:
    """数「内孔」：背景连通域里不贴图像边界的那些，返回面积降序。

    为什么要排除贴边的：把 255-img 当作前景去标记时，图像外圈那一大片背景
    自己就是一个连通域，直接数个数会得到「至少 1 个孔」的假结论。
    """
    lab, n = call(M.label_components, 255 - img, 8)
    H, W = lab.shape
    sizes = []
    for l in range(1, n + 1):
        ys, xs = np.nonzero(lab == l)
        if ys.min() == 0 or ys.max() == H - 1 or xs.min() == 0 or xs.max() == W - 1:
            continue  # 贴边 = 外部背景，不是孔
        sizes.append(int((lab == l).sum()))
    return sorted(sizes, reverse=True)


@pytest.mark.parametrize("k", [3, 5])
@pytest.mark.parametrize("op", ["erode", "dilate", "morph_open", "morph_close"])
def test_morphology_exact(binary, op, k):
    assert_close(call(getattr(M, op), binary, k), getattr(refs, op + "_ref")(binary, k), TOL, f"{op} k={k}")


@pytest.mark.parametrize("it", [1, 2, 3])
def test_iterations_match_cv2(binary, it):
    assert_close(call(M.erode, binary, 3, it), refs.erode_ref(binary, 3, it), TOL, f"erode it={it}")
    assert_close(call(M.dilate, binary, 3, it), refs.dilate_ref(binary, 3, it), TOL, f"dilate it={it}")


def test_open_removes_speckle():
    """开运算去掉孤立单像素噪点，同时保住大物体。"""
    b = cases.blobs()
    out = call(M.morph_open, b, 3)
    lab, n = call(M.label_components, out, 8)
    areas = sorted(np.bincount(lab.ravel())[1:].tolist(), reverse=True)
    assert areas[0] > 1000, "大物体被开运算破坏了"
    assert all(a > 5 for a in areas), f"开运算后仍有面积 {min(areas)} 的碎点"


def test_close_fills_holes_by_size():
    """闭运算填孔的判据：结构元装不进孔里，才填得掉这个孔。

    2026-10-18 在 holes 用例上实测（三个内孔面积 83 / 45 / 20）：
        k=3  一个都填不掉，只是把孔磨小了
        k=5  填掉最小的（20），另两个变成 79 / 35
        k=7  只剩最大的，63
        k=9  全填平
    这组数字直接说明「结构元尺寸怎么选」不是玄学，而是和缺陷尺寸挂钩的。
    """
    b = cases.holes()
    assert interior_hole_sizes(b) == [83, 45, 20], "用例本身应当有三个内孔"
    assert interior_hole_sizes(call(M.morph_close, b, 3)) == [83, 43, 16]
    assert interior_hole_sizes(call(M.morph_close, b, 5)) == [79, 35]
    assert interior_hole_sizes(call(M.morph_close, b, 7)) == [63]
    assert interior_hole_sizes(call(M.morph_close, b, 9)) == []


def test_morphology_inclusions(binary):
    """形态学的基本包含关系，对任何含原点的结构元都成立：
        erode ⊆ 原图 ⊆ dilate ，  open ⊆ 原图 ⊆ close
    这四条是开/闭运算「抗噪声但不改变物体大致形状」的形式化说法，
    边界口径错了一定会在贴边处露馅。
    """
    b = binary > 0
    assert set(np.unique(call(M.erode, binary, 3))) <= {0, 255}
    e = call(M.erode, binary, 3) > 0
    d = call(M.dilate, binary, 3) > 0
    o = call(M.morph_open, binary, 3) > 0
    c = call(M.morph_close, binary, 3) > 0
    assert not (e & ~b).any(), "腐蚀结果跑到了原图外面"
    assert not (b & ~d).any(), "膨胀没有覆盖住原图"
    assert not (o & ~b).any(), "开运算结果跑到了原图外面"
    assert not (b & ~c).any(), "闭运算没有覆盖住原图"


@pytest.mark.parametrize("conn", [4, 8])
def test_label_count_and_areas(binary, conn):
    lab, n = call(M.label_components, binary, conn)
    rlab, rn, _, _ = refs.label_ref(binary, conn)
    assert n == rn, f"连通域个数 {n} != cv2 {rn}"
    mine = sorted(np.bincount(lab.ravel())[1:].tolist(), reverse=True)
    theirs = sorted(np.bincount(rlab.ravel())[1:].tolist(), reverse=True)
    assert mine == theirs, "面积多重集合不一致"


def test_label_background_is_zero(binary):
    lab, n = call(M.label_components, binary, 8)
    assert lab[binary == 0].max() == 0


def test_checker_connectivity_difference():
    """棋盘格：4 连通和 8 连通必须给出不同结果，这是连通性口径的直接证据。"""
    c = cases.checker()
    _, n4 = call(M.label_components, c, 4)
    _, n8 = call(M.label_components, c, 8)
    assert n4 != n8, "4/8 连通在棋盘格上结果相同，连通性参数没生效"


def test_region_stats_match_cv2(binary):
    mine = call(M.region_stats, binary, 8)
    theirs = refs.region_stats_ref(binary, 8)
    assert len(mine) == len(theirs)
    for a, b in zip(mine, theirs):
        for key in ("area", "x", "y", "w", "h"):
            assert a[key] == b[key], f"{key}: {a[key]} != {b[key]}"
        assert abs(a["cx"] - b["cx"]) < 1e-6
        assert abs(a["cy"] - b["cy"]) < 1e-6


def test_region_stats_sorted_by_area(binary):
    stats = call(M.region_stats, binary, 8)
    areas = [s["area"] for s in stats]
    assert areas == sorted(areas, reverse=True)
