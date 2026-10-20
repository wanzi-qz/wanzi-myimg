"""主题：wanzi.me 的 :root 设计变量，色值全部集中在这一个字典里。

从 app.py 抽出来是为了 web 版：Streamlit 那边不能 import gui.app——
tkinter 在没有显示环境的机器上一 import 就炸。配色和 GIF 导出是桌面版、
web 版共用的纯值，放这里两边都能拿。
"""

from __future__ import annotations

# --- wanzi.me 的 :root 设计变量 ---------------------------------------------
THEME = {
    "bg": "#070707",          # --bg：页面底
    "bg_soft": "#0a0a0b",     # --bg-soft：树、状态区、输入框底
    "panel": "#0d0d0e",       # --panel：卡片 / 控件底
    "hover": "#161617",
    "line": "#262626",        # --line：常规描边
    "line_soft": "#1b1b1c",   # --line-soft：卡片描边
    "hover_line": "#3b3b3c",
    "ink": "#f3f3f1",         # --ink：主文字
    "ink_soft": "#ededeb",
    "muted": "#9b9ea2",       # --muted：次要文字
    "faint": "#6b6d70",       # --faint：最弱
    "gold": "#d8b46a",        # 强调金（active tab 下划线、渐变标题都用它）
    "gold_dim": "#b98f3e",
    "gold_pale": "#f6e7c4",
    "gold_soft": "#1a1610",   # 金色按钮 / 选中项的底
    "gold_line": "#5d4d2b",
    "ok": "#86c991",
    "warn": "#d8b46a",
    "bad": "#e0707a",
    "info": "#c9cbcc",
    "note": "#6b6d70",
}


# --- 判定文本 -> 语义色 -------------------------------------------------------
# 名单来自 benchmarks 的判定口径；新增判定时在这里加一行，两个界面同时生效。
VERDICT_OK = ("逐像素精确", "一致", "浮点噪声级一致", "IoU 合格")
VERDICT_WARN = ("仅记录", "两边都无边缘")
VERDICT_BAD = ("超出容差", "不一致", "IoU 偏低")


def verdict_tone(verdict: str) -> str:
    """判定文本 -> ok / warn / bad。名单外的按 bad 处理，宁可扎眼也别糊过去。"""
    if verdict in VERDICT_OK:
        return "ok"
    if verdict in VERDICT_WARN:
        return "warn"
    return "bad"


def verdict_color(verdict: str) -> str:
    """判定文本 -> 主题色值（web 版直接拿去写 inline style）。"""
    return THEME[verdict_tone(verdict)]


def _hex_rgb(c: str) -> tuple[int, int, int]:
    c = c.lstrip("#")
    return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)


def _blend(fg: str, bg: str, a: float) -> str:
    """把 fg 按透明度 a 混到 bg 上，返回十六进制色。"""
    f, b = _hex_rgb(fg), _hex_rgb(bg)
    return "#%02x%02x%02x" % tuple(int(round(b[i] + (f[i] - b[i]) * a)) for i in range(3))