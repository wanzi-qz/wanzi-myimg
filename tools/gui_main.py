r"""打包用的入口：exe 双击之后跑的就是这个文件。

平时开发用 `python -m gui`（走 gui/__main__.py）；打包时 PyInstaller 需要一个
明确的脚本文件当入口，就放这儿。它只做两件事：把仓库根塞进 sys.path，
然后调 gui.app.main()。

冻结（frozen）之后 cwd 不再是仓库根，__file__ 也指向临时解包目录，
所以路径得从 sys.frozen / sys._MEIPASS 这一侧推——好在包本身已经被打进去了，
_MEIPASS 就是那个根。

另外认一个 --selftest：不开窗口，把 engine / render / GIF 三条链路各跑一遍就退出，
返回码 0 = 打包没漏模块。windowed 的 exe 崩了没有控制台可看，这个就是给打包做
体检用的：`dist\myimg-gui.exe --selftest`，看返回码（或 --console 版看输出）。
"""

from __future__ import annotations

import pathlib
import sys


def _root() -> pathlib.Path:
    if getattr(sys, "frozen", False):
        return pathlib.Path(getattr(sys, "_MEIPASS",
                                    pathlib.Path(sys.executable).parent))
    return pathlib.Path(__file__).resolve().parents[1]


ROOT = _root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _log(msg: str) -> None:
    """windowed 模式下 stdout 可能是空壳，打印失败也不能把自检带崩。"""
    try:
        print(msg, flush=True)
    except Exception:
        pass


# 自检挑这四个：滤波 / 阈值 / 边缘 / 连通域各一个，正好把打包最容易漏的四条路
# 都踩一遍——histogram 是 importlib 动态 import，canny 要 cv2，region_stats
# 要 skimage.measure，最后渲染 + GIF 再验 matplotlib 字体和 PIL。
SELFTEST_OPS = (
    ("gaussian_filter", "vector", "microscope"),
    ("otsu_threshold", "vector", "blobs"),
    ("canny", "vector", "microscope"),
    ("region_stats", "vector", "blobs"),
)


def selftest() -> int:
    import io
    import traceback

    import matplotlib

    matplotlib.use("Agg")

    try:
        from gui import engine, params as P, render
        from gui.anim import assemble_gif, grab_frame, prepare_gif_frames

        render.setup_style()   # 中文字体；不先调会刷一屏 glyph 警告

        res = None
        for op, variant, case in SELFTEST_OPS:
            img = engine.synth_case(case, 96)
            raw = {p.key: p.default for p in P.PARAMS[op]}
            res = engine.run(op, variant, raw, img, case=case, repeats=1)
            _log(f"{op:18s} {res.verdict:10s} mine={res.t_mine:6.2f}ms "
                 f"ref={res.t_ref:6.2f}ms")

        fig = render.make_figure(dpi=80)
        render.draw(fig, res, render.default_view(res))
        sel, ms = prepare_gif_frames([grab_frame(fig)] * 2, 480, 1, 120)
        buf = io.BytesIO()
        n = assemble_gif(sel, buf, duration_ms=ms)
        _log(f"render + gif ok: {len(sel)} 帧 / {n} 字节")
        _log("SELFTEST OK")
        return 0
    except Exception:
        _log("SELFTEST FAILED")
        _log(traceback.format_exc())
        return 1


def main() -> None:
    if "--selftest" in sys.argv[1:]:
        raise SystemExit(selftest())

    from gui.app import main as gui_main

    gui_main()


if __name__ == "__main__":
    main()