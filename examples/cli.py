"""命令行入口：把 myimg 的算子接到真实图片上。

    python examples/cli.py list
    python examples/cli.py demo gaussian --case microscope -o out/g.png --sigma 1.4
    python examples/cli.py run median 照片.jpg -o out/m.png --k 5
    python examples/cli.py run otsu 照片.jpg            # 分析类算子直接打印结果
    python examples/cli.py run region_stats 照片.jpg --conn 8

约定：这个文件只做 IO 和展示（PIL 读图存图），算法一律走 myimg。
PIL / matplotlib 不算「实现依赖」——它们碰不到任何一个算子的数值。
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import myimg                                     # noqa: E402
from benchmarks import cases as C               # noqa: E402

# 每个算子接受哪些命令行参数（名字 -> 默认值）。None 表示必填由算子自己决定。
PARAMS = {
    "box_filter": {"k": 5, "mode": "reflect"},
    "box_filter_integral": {"k": 5, "mode": "reflect"},
    "gaussian_filter": {"sigma": 1.4, "ksize": None, "mode": "reflect"},
    "median_filter": {"k": 5, "mode": "edge"},
    "histogram": {},
    "otsu_threshold": {},
    "hist_equalize": {},
    "clahe": {"clip": 2.0, "grid": 8},
    "sobel": {"mode": "reflect"},
    "canny": {"low": 50.0, "high": 150.0, "sigma": 0.0},
    "erode": {"k": 3, "iters": 1},
    "dilate": {"k": 3, "iters": 1},
    "morph_open": {"k": 3},
    "morph_close": {"k": 5},
    "label_components": {"conn": 8},
    "region_stats": {"conn": 8},
}


def load_gray(path: pathlib.Path) -> np.ndarray:
    from PIL import Image
    im = Image.open(path).convert("L")
    return np.asarray(im, dtype=np.uint8)


def save_gray(arr: np.ndarray, path: pathlib.Path) -> None:
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    if arr.dtype == np.bool_:
        arr = (arr.astype(np.uint8)) * 255
    if arr.dtype != np.uint8:
        a = np.clip(arr, 0, 255)
        arr = np.floor(a + 0.5).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(path)


def build_kwargs(op: str, args) -> dict:
    spec = PARAMS[op]
    kw = {}
    for name, default in spec.items():
        val = getattr(args, name, None)
        kw[name] = default if val is None else val
    # 参数名和函数签名对齐
    if "k" in kw:
        kw["ksize"] = kw.pop("k")
    if "clip" in kw:
        kw["clip_limit"] = kw.pop("clip")
    if "iters" in kw:
        kw["iterations"] = kw.pop("iters")
    if "conn" in kw:
        kw["connectivity"] = kw.pop("conn")
    if kw.get("ksize") is None and op == "gaussian_filter":
        kw.pop("ksize")
    return kw


def present(op: str, out, out_path: pathlib.Path | None) -> None:
    """图像类存盘，分析类打印。"""
    if op == "histogram":
        top = np.argsort(out)[::-1][:8]
        print("灰度直方图（前 8 个桶）：")
        for t in top:
            print(f"  灰阶 {t:3d}: {int(out[t]):8d}  {'#' * min(60, int(out[t] * 60 / out.max()))}")
        return
    if op == "otsu_threshold":
        print(f"Otsu 阈值 = {out:.0f}")
        return
    if op == "label_components":
        labels, n = out
        print(f"连通域个数 = {n}")
        if out_path:
            save_gray((labels % 251).astype(np.uint8) * (labels > 0), out_path)
            print(f"标签图（取模上色）-> {out_path}")
        return
    if op == "region_stats":
        print(f"{'面积':>8} {'外接框(x,y,w,h)':>22} {'质心(cx,cy)':>20}")
        for d in out[:20]:
            print(f"{d['area']:8d} {d['x']:6d},{d['y']:4d},{d['w']:4d},{d['h']:4d}   "
                  f"{d['cx']:9.2f},{d['cy']:9.2f}")
        if len(out) > 20:
            print(f"... 还有 {len(out) - 20} 个")
        return
    if op == "sobel":
        gx, gy, mag = out
        arr = mag
        print(f"梯度幅值 max={mag.max():.1f} mean={mag.mean():.2f}")
    else:
        arr = out
        print(f"输出 dtype={arr.dtype} 值域 [{arr.min()}, {arr.max()}]")
    if out_path:
        save_gray(arr, out_path)
        print(f"已保存 -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="myimg 命令行")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="列出全部算子")

    p_demo = sub.add_parser("demo", help="用合成用例跑一个算子")
    p_demo.add_argument("op", choices=sorted(PARAMS))
    p_demo.add_argument("--case", default="microscope",
                        help=f"用例名，可选 {sorted(list(C.GRAY_CASES) + list(C.BINARY_CASES))}")
    p_demo.add_argument("-o", "--out", type=pathlib.Path, default=None)
    _add_common(p_demo)

    p_run = sub.add_parser("run", help="对一张真实图片跑一个算子")
    p_run.add_argument("op", choices=sorted(PARAMS))
    p_run.add_argument("image", type=pathlib.Path)
    p_run.add_argument("-o", "--out", type=pathlib.Path, default=None)
    _add_common(p_run)

    args = ap.parse_args()

    if args.cmd == "list":
        for name in sorted(PARAMS):
            params = ", ".join(f"{k}={v}" for k, v in PARAMS[name].items()) or "无参数"
            print(f"{name:22s} {params}")
        return

    img = C.make(args.case) if args.cmd == "demo" else load_gray(args.image)
    kw = build_kwargs(args.op, args)
    out = myimg.OPS[args.op](img, **kw)
    present(args.op, out, args.out)


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--k", type=int, default=None, help="核尺寸（奇数）")
    p.add_argument("--sigma", type=float, default=None)
    p.add_argument("--ksize", type=int, default=None, help="高斯显式核尺寸，缺省按 ±4σ 推")
    p.add_argument("--mode", default=None, help="边界模式 reflect/symmetric/edge/wrap/constant")
    p.add_argument("--low", type=float, default=None)
    p.add_argument("--high", type=float, default=None)
    p.add_argument("--iters", type=int, default=None)
    p.add_argument("--conn", type=int, default=None, choices=(4, 8))
    p.add_argument("--clip", type=float, default=None)
    p.add_argument("--grid", type=int, default=None)


if __name__ == "__main__":
    main()
