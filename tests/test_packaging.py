"""打包链路的回归：exe 入口的路径推断 + PyInstaller 命令行本身。

exe 打不出来往往不是代码错，而是命令行少了一条参数：`myimg.histogram` 是
`importlib.import_module` 动态取的，静态分析扫不到；`benchmarks` 整个包也得
`--collect-submodules`，否则打完一调判定就 ModuleNotFoundError。这类"忘了就 silently
少一个模块"的东西，靠人记不住，钉成测试。

另一条是 frozen 之后的路径：exe 双击时 cwd 不是仓库根，`__file__` 指向临时解包目录，
`_root()` 必须从 `sys.frozen` / `sys._MEIPASS` 这一侧推。这里在子进程里把这两个属性
伪造出来对拍，不真的去打包。
"""

from __future__ import annotations

import importlib.util
import pathlib
import subprocess
import sys
import textwrap

ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"


def _load(name: str, path: pathlib.Path):
    """按路径加载 tools/ 下的脚本（tools 不是包，没有 __init__.py）。"""
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # __name__ != "__main__"，不会真的开窗口
    return mod


def _run_snippet(snippet: str, *args: str) -> str:
    code = textwrap.dedent(snippet)
    out = subprocess.run([sys.executable, "-X", "utf8", "-c", code, *args],
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    return out.stdout.strip()


# ---------------------------------------------------------------- exe 入口路径

_ROOT_SNIPPET = """
    import importlib.util as u, sys
    sys.frozen = True
    if sys.argv[1] != "-":
        sys._MEIPASS = sys.argv[1]
    spec = u.spec_from_file_location("gui_main", sys.argv[2])
    m = u.module_from_spec(spec)
    spec.loader.exec_module(m)
    print(m._root())
"""


def test_root_frozen_uses_meipass(tmp_path):
    """frozen + _MEIPASS：根就是解包目录（包已经被打进去了）。"""
    got = _run_snippet(_ROOT_SNIPPET, str(tmp_path), str(TOOLS / "gui_main.py"))
    assert pathlib.Path(got) == tmp_path


def test_root_frozen_without_meipass_falls_back_to_exe_dir():
    """frozen 但没有 _MEIPASS（onedir 之类）：退回 exe 所在目录。"""
    got = _run_snippet(_ROOT_SNIPPET, "-", str(TOOLS / "gui_main.py"))
    assert pathlib.Path(got) == pathlib.Path(sys.executable).parent


def test_root_unfrozen_is_repo_root():
    """开发时（python tools/gui_main.py）根就是仓库根，且它已经被塞进 sys.path。"""
    m = _load("gui_main_dev", TOOLS / "gui_main.py")
    assert m.ROOT == ROOT
    assert str(ROOT) in sys.path
    assert (ROOT / "myimg" / "__init__.py").exists()
    assert (ROOT / "gui" / "app.py").exists()


# ------------------------------------------------------- PyInstaller 命令行

class _FakeCall:
    """接住 subprocess.call，只记命令，不真打包。"""

    def __init__(self, rc: int = 0):
        self.rc = rc
        self.cmd: list[str] = []
        self.cwd = None

    def __call__(self, cmd, cwd=None):
        self.cmd = list(cmd)
        self.cwd = cwd
        return self.rc


def _capture(monkeypatch, **kwargs):
    m = _load("build_exe_mod", TOOLS / "build_exe.py")
    fake = _FakeCall(kwargs.pop("rc", 0))
    monkeypatch.setattr(m.subprocess, "call", fake)
    m.build(**kwargs)
    return m, fake.cmd, fake.cwd


def _flag_value(cmd: list[str], flag: str, i: int = 0) -> str:
    idx = [k for k, x in enumerate(cmd) if x == flag]
    assert idx, f"命令行里没有 {flag}：{cmd}"
    return cmd[idx[i] + 1]


def test_default_is_onefile_windowed(monkeypatch):
    _, cmd, _ = _capture(monkeypatch)
    assert "--onefile" in cmd and "--onedir" not in cmd
    assert "--windowed" in cmd and "--console" not in cmd
    assert _flag_value(cmd, "--name") == "myimg-gui"
    assert "--noconfirm" in cmd


def test_collects_dynamic_imports(monkeypatch):
    """两条踩出来的参数：动态 import 的 histogram + 整个 benchmarks 包。"""
    _, cmd, _ = _capture(monkeypatch)
    collected = [_flag_value(cmd, "--collect-submodules", i) for i in (0, 1)]
    assert sorted(collected) == ["benchmarks", "myimg"]
    assert _flag_value(cmd, "--hidden-import") == "myimg.histogram"


def test_paths_and_entry_stay_in_repo(monkeypatch):
    m, cmd, cwd = _capture(monkeypatch)
    assert pathlib.Path(_flag_value(cmd, "--distpath")) == m.ROOT / "dist"
    assert pathlib.Path(_flag_value(cmd, "--workpath")) == m.ROOT / "build"
    assert pathlib.Path(_flag_value(cmd, "--specpath")) == m.ROOT / "build"
    assert pathlib.Path(cmd[-1]) == m.ROOT / "tools" / "gui_main.py"
    assert pathlib.Path(cwd) == m.ROOT          # 在仓库根跑，spec 里的相对路径才对
    assert cmd[0] == sys.executable
    assert "PyInstaller" in cmd


def test_clean_by_default_and_can_be_dropped(monkeypatch):
    _, cmd, _ = _capture(monkeypatch)
    assert "--clean" in cmd
    _, cmd2, _ = _capture(monkeypatch, clean=False)
    assert "--clean" not in cmd2


def test_onedir_and_console_flags(monkeypatch):
    _, cmd, _ = _capture(monkeypatch, onedir=True, console=True)
    assert "--onedir" in cmd and "--onefile" not in cmd
    assert "--console" in cmd and "--windowed" not in cmd


def test_excludes_qt_and_pandas_by_default(monkeypatch):
    """Anaconda 里 Qt 绑定 + pandas 能把 pkg 顶到 600MB+，默认必须排掉。"""
    m, cmd, _ = _capture(monkeypatch)
    excluded = [cmd[i + 1] for i, x in enumerate(cmd) if x == "--exclude-module"]
    for mod in ("PyQt5", "pandas", "IPython"):
        assert mod in excluded, mod
    assert set(excluded) == set(m.EXCLUDES)
    # scipy / skimage / cv2 是基准链路，绝不能被排掉
    for mod in ("scipy", "skimage", "cv2", "numpy", "matplotlib", "PIL"):
        assert mod not in excluded, mod


def test_no_exclude_keeps_everything(monkeypatch):
    _, cmd, _ = _capture(monkeypatch, exclude=False)
    assert "--exclude-module" not in cmd


def test_build_returns_subprocess_rc(monkeypatch):
    """打包失败要把返回码透出去，CI / 脚本才拦得住。"""
    m = _load("build_exe_rc", TOOLS / "build_exe.py")
    monkeypatch.setattr(m.subprocess, "call", _FakeCall(rc=3))
    assert m.build() == 3