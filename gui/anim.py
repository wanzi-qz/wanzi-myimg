"""GIF 导出：把参数扫描攒下的帧序列写成循环动图。

调色板那套取舍（全帧马赛克 + 标准色块，不抖动）是被 PIL 10.4 坑出来的，
原因写在 assemble_gif 的 docstring 里，别改回「只取首帧调色板」。

target 可以是路径，也可以是任何带 tell() 的可写缓冲（BytesIO）：
web 版要把 GIF 直接交给 st.download_button，中间不落盘。
"""

from __future__ import annotations

import pathlib

DEFAULT_FRAME_MS = 150      # 参数扫描的帧间隔：比防抖稍慢一点，肉眼跟得上

GIF_WIDTHS = ("480", "720", "960")   # 导出宽度：桌面版缓存帧就是 960，只往下缩不放大
GIF_SKIPS = ("1", "2", "3", "4")     # 抽帧：隔 k 帧取一帧，帧间隔同步乘 k


def _swatches() -> list[str]:
    """取色用的标准色块：主题色 + 渲染层强调色。

    render 延迟 import——它要拉 matplotlib，而 assemble_gif 只需要那几个色值，
    没必要在导入本模块时就把整条绘图链带上。
    """
    from gui import render
    from gui.theme import THEME

    return [THEME[k] for k in ("bg", "panel", "ink", "muted", "gold")] + [
        render._C_MINE, render._C_REF, render._C_INPUT] + [
        "#%02x%02x%02x" % tuple(round(c * 255) for c in rgb)
        for rgb in (render._C_EDGE_MINE, render._C_EDGE_REF, render._C_EDGE_BOTH)]


def assemble_gif(frames, target, duration_ms: int = DEFAULT_FRAME_MS) -> int:
    """把一列 PIL 帧写成循环 GIF，返回字节数。

    调色板取自「全部帧缩略图拼成的马赛克」：只用第一帧的话，万一首帧色少
    （比如纯色用例），后面帧的颜色会被量化碾平、帧间全同被 PIL 并帧；
    各帧独立调色板又会闪。马赛克用最近邻取样、再贴上主题与渲染强调色的
    标准色块，1px 细彩线也不会在取色时丢。不加抖动——界面本来就是大平色块，
    抖出来的噪点只会把体积撑大。
    """
    from PIL import Image

    thumbs = [f.convert("RGB").resize((64, 40), Image.NEAREST) for f in frames]
    sw = _swatches()
    strip = Image.new("RGB", (64 * len(thumbs) + 16 * len(sw), 40))
    x = 0
    for t in thumbs:
        strip.paste(t, (x, 0))
        x += 64
    for c in sw:
        strip.paste(Image.new("RGB", (16, 40), c), (x, 0))
        x += 16
    pal = strip.quantize(colors=256, dither=Image.NONE)
    q = [f.convert("RGB").quantize(palette=pal, dither=Image.NONE) for f in frames]
    kw = dict(save_all=True, append_images=q[1:], duration=int(duration_ms),
              loop=0, optimize=True)
    if hasattr(target, "tell"):            # BytesIO：交给 web 下载按钮
        q[0].save(target, format="GIF", **kw)
        return int(target.tell())
    q[0].save(target, **kw)
    return pathlib.Path(target).stat().st_size


def prepare_gif_frames(frames, width: int, skip: int,
                       base_ms: int = DEFAULT_FRAME_MS):
    """导出前的帧处理：隔 skip 帧取一帧，再缩到 width 宽。

    返回 (帧列表, 每帧毫秒)。帧间隔乘 skip，这样抽帧之后播放速度仍接近
    扫描时的实时速度，而不是变成快进。不改动传入的缓存帧。
    """
    from PIL import Image

    k = max(1, int(skip))
    sel = list(frames[::k])
    w = max(1, int(width))
    if sel and sel[0].width != w:
        h = max(1, round(sel[0].height * w / sel[0].width))
        sel = [f.resize((w, h), Image.LANCZOS) for f in sel]
    return sel, base_ms * k

def grab_frame(fig):
    """把 Figure 当前内容抓成一张 PIL RGB 图，就是 GIF 的一帧。

    桌面版扫描缓存帧、web 版下载按钮、exe 自检都要这个，放这儿三处共用。
    两个坑，都是踩过的：
      * render.make_figure() 给的是裸 Figure，canvas 是 FigureCanvasBase，
        没有渲染器，直接 buffer_rgba() 会 AttributeError——桌面版没这问题，
        因为它那张 Figure 早就绑了 FigureCanvasTkAgg。这里换成 Agg canvas。
      * render.draw() 结尾是 draw_idle()，异步的；抓像素前必须 canvas.draw()
        强制同步渲染一次，否则拿到的是上一帧。
    """
    import numpy as np
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from PIL import Image

    if not isinstance(fig.canvas, FigureCanvasAgg):
        FigureCanvasAgg(fig)
    fig.canvas.draw()
    return Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert("RGB")