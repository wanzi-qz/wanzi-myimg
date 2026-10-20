"""GUI 的计算层：喂输入、收数字。

关键设计：**判定口径不重写**。误差、IoU、连通域的面积多重集合比较，
全部走 benchmarks.run_compare.compare_one —— 界面上显示的「逐像素精确 / 超出容差 /
仅记录」必须和 compare.md 是同一套代码算出来的。否则迟早会出现
「报告说通过、界面说不通过」，那种不一致比慢十倍还致命。

gui/ 属于应用层，可以 import cv2 / skimage（都封在 benchmarks.refs 里）。
"""

from __future__ import annotations

import dataclasses
import importlib
import pathlib
import sys
from typing import Any, Callable

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import cases as C                    # noqa: E402
from benchmarks import refs as R                     # noqa: E402
from benchmarks.run_compare import SPECS, Spec, compare_one, timeit   # noqa: E402
from myimg import edges as E                         # noqa: E402
from myimg import filters as F                       # noqa: E402
from myimg import morphology as M                    # noqa: E402

from . import params as P                            # noqa: E402

# 包级导出把模块名遮蔽了（myimg.histogram 是函数不是模块），只能这么取
H = importlib.import_module("myimg.histogram")

SPEC_BY_OP: dict[str, Spec] = {s.op: s for s in SPECS}

# 用例的中文说明，全部照抄 benchmarks/cases.py 的 docstring，不加料
CASE_LABELS: dict[str, str] = {
    "flat": "常数图 128（退化用例，Otsu/均衡化容易出 nan）",
    "ramp": "水平线性渐变 0..255（测边界填充与梯度方向）",
    "step_edge": "左 40 / 右 200 阶跃（测中值的保边性、Sobel 方向）",
    "impulse": "ramp 叠椒盐噪声（中值滤波的标准考题）",
    "gauss_noise": "结构 + 高斯噪声 σ=20（高斯滤波的标准考题）",
    "uneven_light": "强光照梯度 + 小目标（Otsu 在这里会失效）",
    "microscope": "合成荧光显微图：暗背景 + 颗粒 + 光照不均 + 噪声",
    "random_uint8": "纯随机图（最严苛的对拍输入，没有结构可藏）",
    "blobs": "圆斑 + 细连接 + 孤立噪点（腐蚀膨胀、连通域主用例）",
    "holes": "实心块内挖小孔（闭运算主用例）",
    "checker": "棋盘格（4 连通和 8 连通结果完全不同）",
    "two_touching": "两个相切圆（连通域计数的边界用例）",
}

GRAY_CASES = list(C.GRAY_CASES)
BINARY_CASES = list(C.BINARY_CASES)

REF_BY_OP: dict[str, Callable] = {
    "box_filter": R.box_ref,
    "gaussian_filter": R.gaussian_ref,
    "median_filter": R.median_ref,
    "histogram": R.histogram_ref,
    "otsu_threshold": R.otsu_ref,
    "hist_equalize": R.hist_equalize_ref,
    "clahe": R.clahe_ref,
    "sobel": R.sobel_ref,
    "erode": R.erode_ref,
    "dilate": R.dilate_ref,
    "morph_open": R.morph_open_ref,
    "morph_close": R.morph_close_ref,
    "label_components": R.label_ref,
    "region_stats": R.region_stats_ref,
}


# ---------------------------------------------------------------------------
# 图像源
# ---------------------------------------------------------------------------
def synth_case(name: str, size: int) -> np.ndarray:
    """合成用例。固定随机种子，任何时候跑出来都一样。"""
    return C.make(name, size)


def real_images(max_side: int = 512) -> dict[str, np.ndarray]:
    """data/ 下的真实图片（目录为空就返回空 dict）。"""
    return C.load_real_images(max_side=max_side)


def open_file(path: str | pathlib.Path, max_side: int = 1024) -> np.ndarray:
    """用户选的本地图片 → 灰度 ndarray。PIL 只做 IO，碰不到算子数值。"""
    from PIL import Image

    im = Image.open(str(path))
    im = im.convert("L")
    if max(im.size) > max_side:
        s = max_side / max(im.size)
        im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))),
                       Image.Resampling.LANCZOS)
    return np.asarray(im, dtype=np.uint8)


def is_binary(img: np.ndarray) -> bool:
    """是不是只有 0 / 255 两个值（形态学算子的合法输入口径）。"""
    vals = np.unique(img)
    return bool(vals.size <= 2 and np.all(np.isin(vals, (0, 255))))


# ---------------------------------------------------------------------------
# 算子解析
# ---------------------------------------------------------------------------
def mine_fn(op: str, variant: str) -> Callable:
    """按算子名 + 变体取 myimg 里的实现。"""
    table: dict[str, dict[str, Callable]] = {
        "box_filter": {"vector": F.box_filter, "naive": F.box_filter_naive,
                       "integral": F.box_filter_integral},
        "gaussian_filter": {"vector": F.gaussian_filter, "naive": F.gaussian_filter_naive},
        "median_filter": {"vector": F.median_filter, "naive": F.median_filter_naive},
        "sobel": {"vector": E.sobel, "naive": E.sobel_naive},
        "histogram": {"vector": H.histogram},
        "otsu_threshold": {"vector": H.otsu_threshold},
        "hist_equalize": {"vector": H.hist_equalize},
        "clahe": {"vector": H.clahe},
        "canny": {"vector": E.canny},
        "erode": {"vector": M.erode},
        "dilate": {"vector": M.dilate},
        "morph_open": {"vector": M.morph_open},
        "morph_close": {"vector": M.morph_close},
        "label_components": {"vector": M.label_components},
        "region_stats": {"vector": M.region_stats},
    }
    if op not in table:
        raise KeyError(f"未知算子 {op!r}")
    if variant not in table[op]:
        raise KeyError(f"{op} 没有 {variant!r} 这个实现，可选 {sorted(table[op])}")
    return table[op][variant]


def spec_for(op: str, variant: str, kw: dict) -> Spec:
    """该用哪条对拍口径。

    两处需要分流，都是事实而不是偷懒：
      * canny 的 sigma>0 时，cv2.Canny 没有 sigma 参数，基准只能是
        「GaussianBlur(uint8) + Canny」，对应报告里的 canny_smoothed（record_only）
      * box_filter 的积分图版对应报告里的 box_filter_integral
    """
    if op == "canny":
        name = "canny_smoothed" if float(kw.get("sigma", 0.0) or 0.0) > 0 else "canny"
        return SPEC_BY_OP[name]
    if op == "box_filter" and variant == "integral":
        return SPEC_BY_OP["box_filter_integral"]
    return SPEC_BY_OP[op]


def ref_fn(op: str, kw: dict) -> Callable:
    """基准函数。canny 要按 sigma 分流，其余直接查表。"""
    if op == "canny":
        if float(kw.get("sigma", 0.0) or 0.0) > 0:
            return lambda i: R.canny_ref_smoothed(i, kw["low"], kw["high"], kw["sigma"])
        return lambda i: R.canny_ref(i, kw["low"], kw["high"])
    return REF_BY_OP[op]


# ---------------------------------------------------------------------------
# 结果容器
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class RunResult:
    op: str
    variant: str
    variant_label: str
    case: str               # 用例名；真实图/本地文件时是它的标签
    size: int               # 输入长边（真实图用它自己的尺寸）
    params: dict            # coerce 之后、真正传进算子的参数
    spec_op: str            # 用的是报告里哪条对拍口径
    kind: str               # image / counts / scalar / labels / stats / tuple3 / iou
    img: np.ndarray         # 输入
    mine: Any               # myimg 的原始返回（可能是 tuple / list[dict] / float）
    ref: Any                # 基准的原始返回
    metrics: dict           # compare_one 的输出，界面所有数字都从这里取
    t_mine: float           # 毫秒，repeats 次取最小
    t_ref: float
    repeats: int
    note: str               # 口径说明（来自 SPECS，和报告里同一句话）
    verdict: str
    warnings: list[str] = dataclasses.field(default_factory=list)

    @property
    def slowdown(self) -> float:
        """myimg 比 OpenCV 慢几倍。小于 1 说明这次比基准还快。"""
        return self.t_mine / self.t_ref if self.t_ref > 0 else float("inf")


def headline(res: "RunResult | None") -> tuple[str, Any]:
    """这一帧最该盯的那个数：不同 kind 的算子主指标不一样。

    桌面版拿它刷状态区，web 版的扫描折线也是把这个值逐帧收下来，
    所以放 engine 而不是 app。
    """
    if res is None:
        return "指标", None
    m = res.metrics or {}
    k = res.kind
    if k == "iou":
        return "IoU", m.get("iou")
    if k == "tuple3":
        pc = (m.get("per_component") or {}).get("mag") or {}
        return "mag max|err|", pc.get("max_abs")
    if k == "labels":
        try:
            return "个数差", abs(int(m.get("mine_n", 0)) - int(m.get("ref_n", 0)))
        except (TypeError, ValueError):
            return "个数差", None
    if k == "stats":
        return "外接框最大差", m.get("max_box_err")
    return "max|err|", m.get("max_abs")


def _warnings(op: str, variant: str, kw: dict, img: np.ndarray,
              case: str, spec: Spec) -> list[str]:
    """界面上要提前说清楚的坑，别等用户看到「超出容差」再猜。"""
    w: list[str] = []
    if op == "median_filter" and str(kw.get("mode", "edge")) != "edge":
        w.append(f"cv2.medianBlur 的边界固定 REPLICATE，当前 mode={kw.get('mode')!r} "
                 f"必然出现差异，这是口径差异不是实现错。")
    if op in P.BINARY_OPS and not is_binary(img):
        w.append("形态学 / 连通域把输入当二值 mask（>0 为真）。当前是灰度图，"
                 "结果只反映「阈值 0」的切分，建议换 blobs / holes / checker。")
    if op == "histogram" and int(kw.get("bins", 256)) != 256:
        w.append(f"bins={kw.get('bins')}：myimg 用 (v*bins)//256 整数分桶；"
                 f"报告里那 112 组对拍用的是 bins=256。")
    if op == "canny" and float(kw.get("sigma", 0.0) or 0.0) > 0:
        w.append("σ>0：基准已切成「cv2.GaussianBlur(uint8) + cv2.Canny」。"
                 "uint8 取整 + 亚像素 NMS 两处口径都不同，IoU 只记录不判定。")
    if op == "canny":
        low, high = float(kw.get("low", 0.0)), float(kw.get("high", 0.0))
        if high < low:
            # 实测：microscope 128² 上 low=200/high=50 与 low=high=50 的边缘点数
            # 完全相同（都是 5846），因为扩展条件比种子条件更严，等于没扩展。
            w.append("high < low：种子条件是 mag≥high，扩展条件是 mag≥low。"
                     "high 比 low 还小时扩展几乎不发生，结果退化成「所有 mag≥high 的点」，"
                     "边缘会变多而不是变空。")
        if low <= 0:
            w.append("low=0：扩展条件恒成立，只要有一个强边缘种子，"
                     "它所在的整个梯度连通区域都会被标记（128² 上实测直接铺满 16384 点）。")
    if spec.record_only:
        w.append("这条口径在报告里就是「仅记录」，不参与一致 / 不一致判定。")
    if variant == "naive" and img.size > 128 * 128:
        w.append(f"朴素版是逐像素 Python 循环，{img.shape[0]}×{img.shape[1]} 会明显卡；"
                 f"报告里的 naive 计时是在 64×64 上跑的。")
    if case.startswith("file:") or case.startswith("data:"):
        w.append("这张图不在报告的 112 组对拍里。真实图一换文件，误差数字就得重跑。")
    return w


def run(op: str, variant: str, raw_params: dict, img: np.ndarray,
        case: str = "", repeats: int = 3) -> RunResult:
    """跑一次对比。异常直接往外抛，由界面负责显示成人话。"""
    if variant == "naive":
        # 朴素版是逐像素 Python 循环，128² 上一次就要 40~75ms，
        # 再取 3 次最小值只会把界面卡住；取 min 的意义本来也不大。
        repeats = 1
    kw = P.coerce(op, raw_params)
    fn_mine = mine_fn(op, variant)
    fn_ref = ref_fn(op, kw)
    spec_src = spec_for(op, variant, kw)

    call_mine: Callable = lambda i: fn_mine(i, **kw)   # noqa: E731
    if op == "canny":
        # ref_fn 已经把 low/high/sigma 绑进闭包了（sigma>0 时基准整个换掉）
        call_ref: Callable = fn_ref
    else:
        call_ref = lambda i: fn_ref(i, **kw)           # noqa: E731

    spec = Spec(op=spec_src.op, mine=call_mine, ref=call_ref,
                kind=spec_src.kind, tol=spec_src.tol,
                record_only=spec_src.record_only, note=spec_src.note,
                cases="binary" if op in P.BINARY_OPS else "gray")

    metrics = compare_one(spec, img)
    mine_out = call_mine(img)
    ref_out = call_ref(img)
    t_mine = timeit(call_mine, img, repeats)
    t_ref = timeit(call_ref, img, repeats)

    label = dict(P.variants(op)).get(variant, variant)
    return RunResult(
        op=op, variant=variant, variant_label=label, case=case,
        size=int(max(img.shape[:2])), params=kw, spec_op=spec_src.op,
        kind=spec_src.kind, img=img, mine=mine_out, ref=ref_out,
        metrics=metrics, t_mine=t_mine, t_ref=t_ref, repeats=repeats,
        note=spec_src.note, verdict=metrics.get("verdict", "?"),
        warnings=_warnings(op, variant, kw, img, case, spec_src),
    )
