"""把桌面版对拍台打成 Windows 可执行文件。

    python tools/build_exe.py              # 单文件：dist/myimg-gui.exe
    python tools/build_exe.py --onedir     # 单目录：启动快，但要整个文件夹一起发
    python tools/build_exe.py --console    # 留个控制台窗口，崩了能看见 traceback

为什么写成脚本而不是直接敲 pyinstaller 命令：参数里有几处是踩出来的——
    * Anaconda 环境里 matplotlib / PIL 会把 PyQt5、IPython、pandas 这些一起拖进来，
      第一次打出来 pkg 有 600+ MB。界面只用 tkinter + Agg，EXCLUDES 那一批直接掐掉，
      需要时用 --no-exclude 还原；
    * myimg.histogram 在 engine / run_compare 里是 importlib.import_module 动态取的，
      静态分析扫不到，得 --collect-submodules 兜住；
    * 界面要的是 tkinter 后端 + matplotlib 的 mpl-data 字体/配置，靠 PyInstaller
      自带 hook 带上，不用手写 --add-data。
    * pandas 只有 web 版（web/app.py）用得到，桌面版不 import，可以排掉；
      scipy 不能排——skimage 依赖它，而 skimage 是基准。
散在命令行里下次就忘了，放这儿至少能读到注释。

产物：dist/myimg-gui.exe（onefile 启动要先解包，第一次点开等两三秒是正常的）。
build/ 和 dist/ 都在 .gitignore 里，不入库。
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
NAME = "myimg-gui"
ENTRY = ROOT / "tools" / "gui_main.py"

# 桌面版只用 tkinter + matplotlib 的 Agg/TkAgg 后端，下面这些是 Anaconda 环境里
# 被间接拖进来的大块头（Qt 绑定动辄上百 MB）。排除后如果 exe 起不来，先怀疑这里，
# 用 --console 看 traceback，或 --no-exclude 打一份对照。
EXCLUDES = (
    # Qt 绑定：界面是 tkinter + Agg，Qt 一个字节都不需要（Anaconda 里光
    # Qt5WebEngineCore 一个 dll 就 107MB）
    "PyQt5", "PyQt6", "PySide2", "PySide6", "qtpy",
    # 交互 / notebook 全家桶：matplotlib 的 hook 会顺手把它们拽进来
    "IPython", "jupyter", "jupyterlab", "notebook", "nbformat", "nbconvert",
    "ipykernel", "ipywidgets", "tornado", "zmq",
    # 只有 web 版（web/app.py）用得到，桌面版不 import
    "pandas", "streamlit", "altair",
    # Anaconda 里被间接拖进来的可视化 / 科学计算大块头，跟对拍无关
    "torch", "torchvision", "torchaudio", "tensorflow",
    "panel", "bokeh", "holoviews", "hvplot", "datashader", "plotly",
    "seaborn", "sklearn", "astropy", "numba", "llvmlite",
    "dask", "distributed", "botocore", "boto3", "s3transfer",
    "imageio_ffmpeg", "tables", "h5py",
    # 杂项：桌面版读写只走 PIL，不碰 excel / parquet / 数据库
    "sympy", "sqlalchemy", "openpyxl", "pyarrow", "pytest",
)


def build(onedir: bool = False, console: bool = False,
          clean: bool = True, exclude: bool = True) -> int:
    cmd = [sys.executable, "-m", "PyInstaller",
           "--onedir" if onedir else "--onefile",
           "--console" if console else "--windowed",
           "--name", NAME,
           "--noconfirm",
           "--collect-submodules", "myimg",
           "--collect-submodules", "benchmarks",
           "--hidden-import", "myimg.histogram",
           *(x for e in (EXCLUDES if exclude else ())
             for x in ("--exclude-module", e)),
           "--distpath", str(ROOT / "dist"),
           "--workpath", str(ROOT / "build"),
           "--specpath", str(ROOT / "build"),
           str(ENTRY)]
    if clean:
        cmd.insert(3, "--clean")
    print("$ " + " ".join(cmd))
    t0 = time.time()
    rc = subprocess.call(cmd, cwd=str(ROOT))
    dt = time.time() - t0
    out = ROOT / "dist" / (NAME if onedir else f"{NAME}.exe")
    if rc == 0 and out.exists():
        if out.is_dir():
            size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
        else:
            size = out.stat().st_size
        print(f"\n打包完成，用时 {dt:.0f}s：{out}（{size / 1024 / 1024:.1f} MB）")
    else:
        print(f"\n打包失败，返回码 {rc}")
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(description="打包 myimg 桌面版对拍台")
    ap.add_argument("--onedir", action="store_true", help="打成目录而不是单文件")
    ap.add_argument("--console", action="store_true", help="保留控制台窗口")
    ap.add_argument("--no-clean", action="store_true", help="不清 PyInstaller 缓存")
    ap.add_argument("--no-exclude", action="store_true",
                    help="不排除 Qt / IPython / pandas 等（体积会大很多）")
    a = ap.parse_args()
    return build(onedir=a.onedir, console=a.console, clean=not a.no_clean,
                 exclude=not a.no_exclude)


if __name__ == "__main__":
    raise SystemExit(main())