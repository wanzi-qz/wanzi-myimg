"""界面参数表：每个算子在 GUI 上暴露哪些可调项。

app.py 照着这张表生成控件，engine.py 照着它拼函数调用。

表里的 key 必须和 myimg 的函数签名一致 —— engine 是直接 **kwargs 传进去的，
名字写错不会在启动时报错，只会在点中那个算子的一瞬间炸。
tests/test_gui_engine.py 会把这张表和真实签名逐条对一遍，所以改这里记得跑测试。
"""

from __future__ import annotations

import dataclasses

# 高斯的 ksize 选这一项就是 None，交给算子按 ±4σ 自己算核宽
AUTO = "自动"

# myimg 支持的边界填充口径（_core.SUPPORTED_PAD_MODES）
MODES = ("reflect", "symmetric", "edge", "wrap", "constant")

_ODD_SIZES = tuple(str(k) for k in range(3, 16, 2))


@dataclasses.dataclass(frozen=True)
class Param:
    """一个可调项。kind 决定生成什么控件。"""

    key: str            # 传给算子的关键字参数名
    label: str          # 界面上的中文标签
    kind: str           # int / odd / float / choice
    default: object     # 默认值；choice 用字符串，其余用数字
    lo: float = 0.0
    hi: float = 1.0
    step: float = 1.0
    digits: int = 2     # float 在界面上显示几位小数
    options: tuple = ()  # choice 的候选（全用字符串存，coerce 时再转型）
    hint: str = ""      # 口径提醒，会拼进底部的说明区


# --- 算子分组（界面左上的树） -------------------------------------------------
OP_GROUPS: list[tuple[str, list[str]]] = [
    ("滤波 / 平滑", ["box_filter", "gaussian_filter", "median_filter"]),
    ("直方图 / 阈值", ["histogram", "otsu_threshold", "hist_equalize", "clahe"]),
    ("边缘", ["sobel", "canny"]),
    ("形态学 / 连通域", ["erode", "dilate", "morph_open", "morph_close",
                        "label_components", "region_stats"]),
]

ALL_OPS: list[str] = [op for _, ops in OP_GROUPS for op in ops]

# 这些算子把输入当二值 mask 处理（>0 为真），选灰度用例结果没有意义
BINARY_OPS = frozenset({"erode", "dilate", "morph_open", "morph_close",
                        "label_components", "region_stats"})

# 选中这类算子时界面自动切到这个用例，省得用户对着灰度图纳闷
DEFAULT_CASE_FOR_BINARY = "blobs"
DEFAULT_CASE_FOR_GRAY = "microscope"

PARAMS: dict[str, tuple[Param, ...]] = {
    "box_filter": (
        Param("ksize", "核宽 ksize", "choice", "5", options=_ODD_SIZES,
              hint="必须奇数。滑窗版复杂度 O(HW·k²)，积分图版 O(HW) 与 k 无关。"),
        Param("mode", "边界 mode", "choice", "reflect", options=MODES,
              hint="reflect=cv2 的 BORDER_REFLECT_101（不重复边界像素），"
                   "symmetric=BORDER_REFLECT，edge=REPLICATE。wrap 要 numpy 先补边，"
                   "cv2.boxFilter 不接受 BORDER_WRAP。"),
    ),
    "gaussian_filter": (
        Param("sigma", "σ", "float", 1.4, lo=0.1, hi=5.0, step=0.1, digits=2,
              hint="σ 决定核宽：自动模式下 ksize = 2*floor(4σ+0.5)+1，是 ±4σ 不是 ±3σ。"),
        Param("ksize", "核宽 ksize", "choice", AUTO,
              options=(AUTO,) + _ODD_SIZES + tuple(str(k) for k in range(17, 32, 2)),
              hint="选「自动」= 传 None，由算子按 ±4σ 算，和 cv2 的 ksize=0 口径一致。"),
        Param("mode", "边界 mode", "choice", "reflect", options=MODES),
    ),
    "median_filter": (
        Param("ksize", "核宽 ksize", "choice", "5", options=_ODD_SIZES,
              hint="cv2.medianBlur 只收 uint8（ksize<=5 时也收 float32）。"),
        Param("mode", "边界 mode", "choice", "edge", options=MODES,
              hint="cv2.medianBlur 的边界固定 REPLICATE，所以只有 mode='edge' 能对得上；"
                   "选别的必然出现差异，那是口径差异不是 bug。"),
    ),
    "histogram": (
        Param("bins", "分箱 bins", "choice", "256",
              options=("8", "16", "32", "64", "128", "256"),
              hint="np.bincount 整数分桶；bins<256 时 cv2.calcHist 的分桶边界口径要对齐。"),
    ),
    "otsu_threshold": (),
    "hist_equalize": (),
    "clahe": (
        Param("clip_limit", "削峰 clip_limit", "float", 2.0, lo=1.0, hi=10.0, step=0.5, digits=1,
              hint="myimg 与 cv2 在削峰后重分配几轮、余数怎么摊上口径不同，"
                   "clip 越大差得越多，报告里最大差 67 灰阶。"),
        Param("grid", "分块 grid", "int", 8, lo=2, hi=16, step=1,
              hint="tile 数 = grid²。不能整除图像尺寸时余数怎么摊，两家也不同。"),
    ),
    "sobel": (
        Param("mode", "边界 mode", "choice", "reflect", options=MODES,
              hint="可分离：先 [1,2,1] 平滑再 [-1,0,1] 差分。与 cv2.Sobel 精确一致。"),
    ),
    "canny": (
        Param("low", "低阈值 low", "float", 50.0, lo=0.0, hi=255.0, step=1.0, digits=0),
        Param("high", "高阈值 high", "float", 150.0, lo=0.0, hi=255.0, step=1.0, digits=0,
              hint="high < low 时滞后连接直接找不到强边缘，结果全空。"),
        Param("sigma", "预平滑 σ", "float", 0.0, lo=0.0, hi=3.0, step=0.1, digits=1,
              hint="σ=0 表示不预平滑，基准是 cv2.Canny；σ>0 时基准换成"
                   "「cv2.GaussianBlur(uint8) + cv2.Canny」，两边口径不同，IoU 只记录不判定。"),
    ),
    "erode": (
        Param("ksize", "结构元 ksize", "choice", "3", options=_ODD_SIZES,
              hint="边界 pad True（等价 cv2 的 +inf 边界）：贴边的亮块不会被腐蚀掉。"),
        Param("iterations", "迭代次数", "int", 1, lo=1, hi=5, step=1),
    ),
    "dilate": (
        Param("ksize", "结构元 ksize", "choice", "3", options=_ODD_SIZES,
              hint="边界 pad False（等价 cv2 的 -inf 边界）。"),
        Param("iterations", "迭代次数", "int", 1, lo=1, hi=5, step=1),
    ),
    "morph_open": (
        Param("ksize", "结构元 ksize", "choice", "3", options=_ODD_SIZES,
              hint="先腐蚀后膨胀：断细连接、去孤立噪点，结果一定是原图的子集。"),
    ),
    "morph_close": (
        Param("ksize", "结构元 ksize", "choice", "5", options=_ODD_SIZES,
              hint="先膨胀后腐蚀：按尺寸填内孔。holes 用例实测 k=3→[83,43,16]、"
                   "k=5→[79,35]、k=7→[63]、k=9→[]。"),
    ),
    "label_components": (
        Param("connectivity", "连通性", "choice", "8", options=("4", "8"),
              hint="checker 用例上 4 连通和 8 连通给出的域数完全不同。"
                   "编号顺序不保证和 cv2 一致，比的是个数与面积多重集合。"),
    ),
    "region_stats": (
        Param("connectivity", "连通性", "choice", "8", options=("4", "8"),
              hint="输出按面积降序，cv2 按标签号排；对比前先排序。"),
    ),
}

# --- 实现变体 ----------------------------------------------------------------
# 同一个算子的不同写法。naive 是逐像素双重循环，用来量化向量化到底赚了多少；
# integral 是 box filter 的积分图版。
VARIANTS: dict[str, tuple[tuple[str, str], ...]] = {
    "box_filter": (("vector", "向量化滑窗"), ("naive", "朴素双重循环"),
                   ("integral", "积分图 O(HW)")),
    "gaussian_filter": (("vector", "可分离两次一维"), ("naive", "朴素二维卷积")),
    "median_filter": (("vector", "向量化窗口排序"), ("naive", "朴素逐像素")),
    "sobel": (("vector", "可分离向量化"), ("naive", "朴素逐像素")),
}

DEFAULT_VARIANT = ("vector", "向量化")


def variants(op: str) -> tuple[tuple[str, str], ...]:
    """这个算子有几种实现可选。"""
    return VARIANTS.get(op, (DEFAULT_VARIANT,))


def defaults(op: str) -> dict:
    """界面上的初始参数值（已按 key 存成字符串/数字）。"""
    return {p.key: p.default for p in PARAMS.get(op, ())}


def coerce(op: str, raw: dict) -> dict:
    """把界面上的值转成算子真正要的类型。

    界面上的控件只会给字符串或数字，算子要的可能是 None / int / float，
    转换全部集中在这里，engine 拿到的就是能直接 **kwargs 的干净字典。
    """
    out: dict = {}
    for p in PARAMS.get(op, ()):
        v = raw.get(p.key, p.default)
        if p.kind == "choice":
            s = str(v)
            if s == AUTO:
                out[p.key] = None
            else:
                # 数字型的选项（bins / connectivity / ksize）转成 int，
                # 文字型的（mode）原样保留
                out[p.key] = int(s) if s.lstrip("-").isdigit() else s
        elif p.kind == "float":
            out[p.key] = float(v)
        elif p.kind == "odd":
            k = int(round(float(v)))
            out[p.key] = k + 1 if k % 2 == 0 else k
        else:  # int
            out[p.key] = int(round(float(v)))
    return out


def hints(op: str) -> list[str]:
    """当前算子所有参数的口径提醒，界面底部逐条显示。"""
    return [f"{p.label}：{p.hint}" for p in PARAMS.get(op, ()) if p.hint]
