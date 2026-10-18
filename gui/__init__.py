"""myimg 的可视化交互界面（Tkinter + matplotlib）。

跑法
    python -m gui                # 或 python gui\app.py

分工
    params.py  每个算子在界面上暴露哪些可调项（唯一事实源）
    engine.py  调 myimg / OpenCV，算误差与耗时（判定逻辑复用 benchmarks）
    render.py  把结果画成 2x2 四格
    app.py     窗口、控件、事件

纪律：gui/ 和 benchmarks/ 一样属于「应用层」，允许 import cv2 / skimage；
myimg/ 包里依然不许出现任何第三方实现，那条规矩没破。
"""
