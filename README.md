# myimg

纯 NumPy 手写常用图像处理算子，逐个与 OpenCV / scikit-image 对拍，把误差、耗时、差异原因都记下来。

## 为什么做这个

硕士课题的图像预处理流程我走通了，但有些算子属于「会调、说不清里面在算什么」。
这个仓库就是把它们拆开重写：高斯滤波到底怎么卷的、Otsu 的类间方差在遍历什么、
Canny 的非极大值抑制为什么要对幅值插值。写完再跟工业标准库对拍，
误差对不上就必须找到原因，不许用「差不多就行」糊过去。

不是想替代 OpenCV——手写版慢一个数量级是常态，报告里的耗时表就是用来记录这个差距的。
目的是把每个算子的原理和实现细节钉死，被追问的时候答得上来。

## 目录

```
myimg/            算子实现（手写区，只允许 numpy + 标准库）
  _core.py        dtype 归一、边界填充、输入校验（基础设施，已写好）
  filters.py      高斯 / 均值 / 中值滤波
  histogram.py    Otsu 阈值 / 直方图均衡 / CLAHE
  edges.py        Sobel / Canny 简化版
  morphology.py   腐蚀膨胀开闭 / 连通域标记
tests/            pytest：契约测试 + 与 OpenCV 的数值对拍
benchmarks/       对拍运行器与报告模板（cv2 / skimage 只在这里当基准）
  cases.py        12 个固定种子合成用例（含荧光显微合成图）
  refs.py         基准封装，所有口径换算集中在这一个文件
  run_compare.py  跑对拍 -> out/compare.md + compare.json + 对比图
  report_template.md  机器出数字、人写归因，各占一半
examples/cli.py   命令行入口（PIL 只读写图片，碰不到任何算子数值）
notes/            报错本、默写记录
data/             真实图片放这里（已 gitignore，只留 README）
PROGRESS.md       验收清单 + 面试复述清单
```

## 手写纪律

1. `myimg/` 下的算子文件**禁止** import `cv2` / `scipy.ndimage` / `skimage`。库只允许出现在
   `tests/` 和 `benchmarks/` 里当基准（`examples/cli.py` 的 PIL 只负责读写图片，不参与数值）。
   这一条破了，「手写 NumPy 实现」这个说法就不成立了。
2. 每个函数**先写能跑的版本**（双层 for 循环也行），测试转绿之后再改写向量化。
   两版都保留，耗时差写进报告——这本身就是报告里的一节。
3. 对不上的误差不跳过，去 `benchmarks/report_template.md` 的「差异归因」里写清楚原因。
4. 每写完一个算子，48 小时内合上资料默写一遍核心实现，记到 `notes/默写记录.md`。
   默不出来 = 没会，回去重写。

## 快速开始

```bash
pip install -e ".[dev]"

pytest -q                       # 全部：未实现的显示 xfail，实现错的显示 failed
pytest tests/test_filters.py -k gaussian   # 单个算子
python benchmarks/run_compare.py --all --figures   # 误差表+耗时表+对比图
python examples/cli.py demo canny --case microscope --sigma 1.4 -o out.png
```

报告在 `benchmarks/out/compare.md`：误差表、耗时表、差异归因、已核对的口径事实四节。
每个数字都能在 `compare.json` 里找到出处；人写的那一半用 `benchmarks/report_template.md`。

测试结果的三种状态是有意的：

| 状态 | 含义 |
|---|---|
| `xfailed` | 函数体还是 `NotImplementedError`，没开始写 |
| `failed` | 写了，但和基准对不上——去查边界处理、核归一化、dtype |
| `passed` | 对上了，可以去默写一遍 |

## 对标口径（2026-10-18 本机实测，numpy 1.26.4 / opencv-python 4.10.0 / scikit-image 0.25.2）

对拍之前先把口径对齐，否则误差全是自己造的：

**边界模式三家命名不一致**

| `np.pad(mode=)` | `cv2` borderType | `scipy` / `skimage` mode |
|---|---|---|
| `reflect` | `BORDER_REFLECT_101` | `mirror` |
| `symmetric` | `BORDER_REFLECT` | `reflect` |
| `edge` | `BORDER_REPLICATE` | `nearest` |
| `wrap` | `BORDER_WRAP` | `wrap` |
| `constant` | `BORDER_CONSTANT` | `constant` |

scipy 的 `reflect` 指的是**含边缘**的对称填充，跟 numpy 的 `reflect` 不是一回事。踩过一次就记住了。

**已实测的换算关系**

- 可分离高斯（`np.convolve` 时核要反转）对 `cv2.GaussianBlur(..., borderType=BORDER_REFLECT_101)`：最大绝对误差 `5.68e-14`，可以当完全一致。
- `cv2.getGaussianKernel(7, 2.0)` = `[0.070159, 0.131075, 0.190713, 0.216106, 0.190713, 0.131075, 0.070159]`，不是直接套一维高斯公式，OpenCV 内部有自己的归一化和取整。
- `skimage.sobel(img / 255.0)` == `cv2 幅值 / 255.0 / (4 * sqrt(2))`，精确到 `1.8e-16`。skimage 把平滑核除了 4、求导核除了 √2。
- `cv2.equalizeHist` 与 `skimage.exposure.equalize_hist * 255` 最大差 `1.657` 灰阶 → 对拍取 ±2 容差。
- `cv2.medianBlur` 的边界是 REPLICATE，不是 reflect。
- Otsu：`cv2.threshold(..., THRESH_OTSU)` 与 `skimage.filters.threshold_otsu` 在随机图上都给 125，但两者平局时的取值约定不同，对拍留 ±1 灰阶。
- `cv2.erode` 默认边界值是 `+inf`，`cv2.dilate` 默认是 `-inf`（不是 0）。手写时腐蚀要 pad 大值、膨胀要 pad 小值，否则边缘一圈必错。
- `cv2.boxFilter` **不接受** `BORDER_WRAP`（实测抛 cv2.error）。要对拍环绕边界只能 numpy 先补再常数边界均值。
- OpenCV 高斯的自动核尺寸是 `2 * floor(4σ + 0.5) + 1`（±4σ），不是网传的 ±3σ；`_core.py` 和 `filters.py` 的注释里有命中表。
- `cv2.Canny` 只收 CV_8U 且没有 sigma 参数，工业链路是 `GaussianBlur(uint8) -> Canny`，
  那条链路的 uint8 取整让「带预平滑的 Canny」没法逐像素对拍，只记录 IoU。
- Otsu 在全部 12 个用例上和 cv2 **完全相等**；测试仍留 ±1 容差是给平局约定分叉留的余量。

## 进度

见 [PROGRESS.md](PROGRESS.md)。

## License

MIT
