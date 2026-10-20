# myimg 图像算子对拍台 · 产品使用说明

| 项 | 值 |
|---|---|
| 适用代码 | GitHub `wanzi-qz/wanzi-myimg`，commit `111bc52` 及以后 |
| 文档日期 | 2026-10-21 |
| 测试基线 | 全仓 796 passed / 0 failed |
| 对拍基线 | numpy 1.26.4 / opencv-python 4.10.0 / scikit-image 0.25.2（2026-10-18 本机实测） |
| 对拍规模 | 15 个算子、17 条对拍口径（含实现变体）× 12 个合成用例，共 112 组 |

---

## 1. 背景与目的

### 1.1 要解决的问题

手写图像算子（学习、科研、面试作品都算）永远面对同一个问题：**怎么证明我写的是对的？**
OpenCV 和 scikit-image 是行业事实基准，但拿它们当"标准答案"时有三个坑：

1. **肉眼不可靠**。两幅图看起来一样，逐像素可能差 30 个灰度级；反过来，差 2.8e-14 的浮点噪声
   在视觉上完全无感，却可能被误判成 bug。
2. **口径差异普遍存在**。边界处理（REPLICATE / pad True / pad False）、取整方式
   （floor(x+0.5) 还是 round）、Otsu 平局取最小阈值、Canny 边缘半个像素的位置偏移——
   这些都不是谁写错了，而是"两家定义本来就不一样"。不区分"真 bug"和"口径差异"，
   对拍就会天天狼来了。
3. **判定逻辑容易漂移**。如果命令行报告写一套判定、界面写另一套，迟早出现
   "报告说通过、界面说不通过"。

### 1.2 产品定位

一句话：**以 OpenCV / scikit-image 为基准，对手写 NumPy 图像算子做逐像素对拍，
并把"差在哪、差多少"画出来。** 它同时是三样东西：

- **验收台**：每个算子 × 每个用例给出判定（逐像素精确 / 浮点噪声级一致 / 超出容差 / 仅记录），
  回归测试 796 条守着不让它退化。
- **教学台**：参数滑块 + 2×2 画布 + 信息格（边缘叠加、差值热图、直方图、CDF、连通域柱状），
  改一个参数立刻看到算子行为怎么变。
- **交付物**：桌面 exe 双击即用、Web 版公网链接可分享、CLI 一键出对拍报告。

### 1.3 设计目标与硬规矩

四条目标：判定可追溯（所有数字来自同一次 compare_one 调用）、用例可复现（合成用例固定种子）、
三端不漂移（GUI / Web / CLI 共用同一套判定与渲染）、交付工程化（exe / Docker 友好的依赖口径 / CI）。

三条硬规矩：

1. **判定口径不重写**。界面上每个判定字都走 `benchmarks.run_compare.compare_one`，
   和 CLI 报告是同一套代码算出来的。
2. **口径差异只记录不判定**。`tol=None` 或 `record_only=True` 的口径（clahe、canny_smoothed）
   显示数字但标 🟡，不进入通过/失败。
3. **myimg 包内只准用 numpy**。基准实现（cv2 / skimage）全部隔离在 `benchmarks/refs.py`，
   保证"被验收的代码"和"当裁判的代码"物理分开。

### 1.4 适用场景

- 自己验收新算子：写完跑一下，判定 + 信息格直接告诉你差在哪。
- 学习算子行为：扫一个参数看动画，比看公式直观。
- 给别人演示 / 面试：exe 或公网链接，对方零环境成本。
- 简历叙事素材：对拍方法论（含部署一致性对拍的扩展用法）是可讲的项目亮点。

---

## 2. 程序构成与工作原理

### 2.1 分层结构

```
┌─ 呈现层 ─ gui/app.py (Tkinter 桌面)   web/app.py (Streamlit)   run_compare  CLI 报告
│           ↓ 只喂控件值 / 摆结果，不写业务
├─ 编排层 ─ gui/engine.py   run() → RunResult（取图、取函数、计时、调判定、组装警告）
├─ 判定层 ─ benchmarks/run_compare.py   Spec / compare_one / timeit / SPECS（唯一判定源）
├─ 基准层 ─ benchmarks/refs.py（cv2·skimage 参照）   benchmarks/cases.py（合成用例）
└─ 算法层 ─ myimg/（filters·histogram·edges·morphology，纯 numpy，15 算子）
横切：gui/params.py 参数schema　gui/render.py 信息格　gui/theme.py 配色　gui/anim.py GIF
```

### 2.2 目录职责

| 目录 / 文件 | 职责 |
|---|---|
| `myimg/` | 被验收的手写算子，纯 numpy，禁 import cv2/skimage |
| `benchmarks/cases.py` | 12 个合成用例生成器（固定种子，可复现） |
| `benchmarks/refs.py` | OpenCV / scikit-image 参照实现（裁判） |
| `benchmarks/run_compare.py` | Spec 定义、compare_one 判定、timeit 计时、SPECS 口径表、CLI 入口 |
| `gui/engine.py` | 编排：取图 / 取函数 / 计时 / 调 compare_one / 组装 RunResult 与警告 |
| `gui/params.py` | 算子分组 OP_GROUPS、参数 schema PARAMS、coerce、sweep_sequence |
| `gui/render.py` | matplotlib 画布与信息格（暗色 rcParams，导出与屏幕同色） |
| `gui/theme.py` | 主题色板（对齐 wanzi.me：黑底 #070707 + 强调金 #d8b46a） |
| `gui/anim.py` | grab_frame 抓帧、prepare_gif_frames、assemble_gif |
| `gui/app.py` | Tkinter 桌面版（对共用模块只做 re-export） |
| `web/app.py` | Streamlit Web 版（控件层，零业务逻辑） |
| `tools/` | build_exe.py 打包、start_demo / stop_demo 一键启停 |
| `tests/` | 796 条回归（含"掐掉 tkinter 再 import web.app"的隔离测试） |

### 2.3 一次交互的完整数据流

控件值 → `params.coerce`（类型/范围/奇偶规整）→ `engine.run(op, variant, raw_params, img, case, repeats)`
→ 内部取 `mine_fn` / `ref_fn` → `compare_one(spec, img)` 算出 metrics（含 verdict）
→ `timeit` 各跑 repeats 次取最小 → 组装 `RunResult` → `render.views_for` 决定信息格候选
→ 画布 2×2 + 信息格 → `engine.headline` / `render.summary_line` 出状态行文案。
GUI 参数改完 150ms 防抖自动重跑这条链；Web 版每次控件变化重跑同一条链。

### 2.4 关键对象

**Spec（一条对拍口径）**：`op` 算子名、`mine`/`ref` 两侧取函数、`naive` 朴素变体、
`kind` 结果形态、`tol` 容差（None=仅记录）、`cases` 灰度/二值、`record_only`、`note` 口径说明
（界面上那句"口径"文案就来自这里，和报告同一句话）。

**kind 七种结果形态与比对方式**：

| kind | 含义 | 比对方式 |
|---|---|---|
| image | 输出同尺寸图像 | 逐像素 max/mean 绝对误差 + 超容差像素数 |
| counts | 直方图计数向量 | 逐 bin 误差 |
| scalar | 单值（Otsu 阈值） | 绝对差，tol=1.0（阈值取整级） |
| tuple3 | sobel 的 (gx, gy, mag) | 三分量分别比，取最差 |
| labels | 连通域编号图 | 个数 + 排序后面积序列全等（编号顺序不算口径） |
| stats | 连通域属性列表 | 逐个比外接框与质心 |
| iou | canny 边缘 | 边缘集合 IoU（边缘位置差半像素，不比逐像素） |

**RunResult（一次运行的全部事实）**：op/variant/case/size/params（coerce 后真传进去的值）、
spec_op（用的哪条口径）、kind、img、mine、ref、metrics（界面所有数字的唯一来源）、
t_mine/t_ref（毫秒，repeats 次取最小）、repeats、note、verdict、warnings。
属性 `slowdown = t_mine / t_ref`，小于 1 表示这次比基准还快。

**verdict 判定措辞（_verdict 原文逻辑）**：

| 措辞 | 条件 | 含义 |
|---|---|---|
| 逐像素精确 | max_abs == 0 | 比特级一致 |
| 浮点噪声级一致 | 0 < max_abs ≤ tol | 浮点结合序差异，视为通过 |
| 超出容差 | max_abs > tol | 真差异：查 note 与信息格定位 |
| 一致 / 不一致 | labels / stats 专用 | 个数+面积序列 / 框+质心 全等与否 |
| 仅记录 | tol=None 或 record_only | 🟡 口径差异，显示数字不判定 |
### 2.5 用例库

合成用例固定种子、可复现；`data/` 里有真实图时优先用真实图；GUI/Web 也能上传本地图片
（长边压到 1024 以内）。

| 组 | 用例 | 专门压什么 |
|---|---|---|
| 灰度 | flat | 常数图：边界与除零路径 |
| 灰度 | ramp | 线性梯度：平滑算子的保真度 |
| 灰度 | step_edge | 阶跃边缘：边缘算子的定位 |
| 灰度 | impulse | 脉冲噪声：中值滤波的主场 |
| 灰度 | gauss_noise | 高斯噪声：滤波前后信噪比 |
| 灰度 | uneven_light | 光照不均：直方图均衡 / clahe 的主场 |
| 灰度 | microscope | 显微纹理：贴近真实科研图像 |
| 灰度 | random_uint8 | 全随机：无结构的压力测试 |
| 二值 | blobs | 分离连通域 |
| 二值 | holes | 带孔洞：连通域编号与面积 |
| 二值 | checker | 棋盘格：高频边界 |
| 二值 | two_touching | 两个相切目标：4/8 连通性的分水岭 |

形态学 / 连通域 6 个算子把输入当二值 mask（>0 为真），选灰度用例没有意义——
engine 会给出 warnings 提示，界面上以警告条呈现。

### 2.6 参数与扫描

`Param(key, label, kind, default, lo, hi, step, digits, options, hint)`：
kind 分 slider / choice 等；`coerce` 负责类型转换、夹范围、奇数核保奇。
`OP_GROUPS` 四个分组：滤波/平滑、直方图/阈值、边缘、形态学/连通域。

参数扫描取值走 `sweep_sequence`，四条规则：choice 按原序、浮点舍入去重、
奇数核保奇、点数不足时按 SWEEP_STEPS=(12,24,36,48) 兜底插值。
扫描时画布逐帧刷新放动画，帧顺路按 960 宽缓存；导出 GIF 时宽度（480/720/960）
与抽帧（隔 k 帧取一帧，k=1..4）可选，帧间隔随抽帧倍数放大，播放速度仍接近扫描实时。
调色板由全帧马赛克 + 主题标准色块统一量化，避免帧间闪色。

### 2.7 三端为什么不漂移

GUI、Web、CLI 共用：判定 `compare_one`、编排 `engine.run`、渲染 `render.*`、
扫描 `params.sweep_sequence`、组帧 `anim.*`、文案 `engine.headline` / `render.summary_line`。
Web 层只做三件事：把控件值喂给 engine、把 Figure 转位图（`anim.grab_frame`，
内部换 FigureCanvasAgg 才有 buffer_rgba）、把结果摆上页面。
桌面版修好一个口径 bug，Web 版自动跟着好。防退化靠两条回归：
参数表 key 与 myimg 函数签名逐条对齐；子进程掐掉 tkinter 再 import `web.app`。

### 2.8 计时口径

`timeit(fn, img, repeats)` 跑 repeats 次（界面默认 3，扫描时降为 1）取**最小值**，
减少调度抖动；CLI 的 `--naive-size`（默认 64）单独给朴素变体计时用，
因为朴素版 O(HW·k²) 在大图上慢到没有演示价值。slowdown 是验收副产物，
**本产品不是性能基准台**：纯 numpy 对 C/SIMD 慢几十到几百倍属预期，见 5.5。

---

## 3. 安装与启动

### 3.1 环境基线

Python ≥ 3.9（全仓 `from __future__ import annotations`）。对拍基线版本：
numpy 1.26.4 / opencv-python 4.10.0 / scikit-image 0.25.2；换版本可能引入浮点噪声级差异，
判定容差已留余量，但报告数字会变。

### 3.2 形态一：CLI 对拍报告

```bash
pip install -e ".[dev]"
python -m benchmarks.run_compare --all                 # 全算子 × 全用例
python -m benchmarks.run_compare --op canny --case step_edge --figures
```

产物在 `benchmarks/out/`：`compare.json`（全部原始数字，可追溯可 diff）、
`compare.md`（人读的误差表 + 耗时表 + 差异归因）、`fig_*.png`（--figures 时）。
常用参数：`--size`（用例边长，默认 128）、`--repeats`、`--naive-size`、`--out`。

### 3.3 形态二：桌面 GUI

```bash
pip install -e ".[dev]"
python -m gui
```

或直接双击 `dist/myimg-gui.exe`（对方机器无需 Python；onefile 首次启动要解包，等几秒正常）。
体检：`dist\myimg-gui.exe --selftest`，不开窗口跑滤波/阈值/边缘/连通域四条链路 +
渲染 + GIF，返回码 0 = 打包没漏模块。

### 3.4 形态三：Web 对拍台

```bash
pip install -e ".[web]"
streamlit run web/app.py --server.port 8501
```

Windows 上更省事：双击 `tools/start_demo.bat`——8501 没起就拉 streamlit，
再起 cloudflared 公网隧道，最后打印本机 / 同局域网 / 公网三条链接并把公网链接复制进剪贴板；
`tools/stop_demo.bat` 按端口和进程名收掉两者。日志在 `%TEMP%\myimg-*.log`。
同局域网访问需防火墙放行 8501（规则名 `myimg-streamlit-8501`）。

> 坑：**改了 `gui/` 下的共用模块必须重启 streamlit 服务**。streamlit 刷新只重跑
> `web/app.py`，不会重新 import 已加载进内存的模块，旧进程会拿着搬家前的模块报
> ImportError，看着像代码坏了，重启即好。

### 3.5 打包 exe（独立 venv，别用 Anaconda 直打）

Anaconda 的 hook 会把 Qt / torch / panel / imageio-ffmpeg 全拖进来：第一版 759 MB、
解包一分多钟。独立 venv 实测 105.9 MB、打包 107 秒、双击 4.8 秒出窗口：

```bash
W:\Anaconda\python.exe -m venv .venv-pack
.venv-pack\Scripts\python -m pip install numpy==1.26.4 opencv-python-headless==4.10.0.84 scikit-image==0.25.2 matplotlib==3.9.2 pillow==10.4.0 pyinstaller==6.22.3
.venv-pack\Scripts\python tools\build_exe.py
```

可选：`--onedir`（启动快，整文件夹分发）、`--console`（留控制台看 traceback）。

### 3.6 CI：push 之后谁在替你跑

`.github/workflows/ci.yml`，ubuntu-latest + Python 3.12，push / PR / 手动都触发。
两道门串在同一个 job 里：

| 门 | 命令 | 拦什么 |
|---|---|---|
| 回归 | `python -X utf8 -m pytest` | 796 条测试，代码写错就红 |
| 对拍 | `run_compare.py --all --figures` → `tools/ci_check.py` | 112 组判定有没有变红、组数够不够、依赖基线有没有漂 |

`ci_check.py` 存在的理由有两条，都是这个项目最容易 silently 坏掉的地方：

1. **`run_compare` 是报告工具，不是裁判。** 出现「超出容差」它只打印一行 `[待查]`
   然后 exit 0。人在终端前会看见，CI 里没人看见，等于没跑。红牌得有人举。
2. **判定强依赖上游库的实现细节。** numpy 从 1.26 升到 2.x，浮点结合序一变，
   「浮点噪声级一致」可能就成了「超出容差」，而这跟代码对错毫无关系。
   `compare.json` 里存了 env，正好拿来和 `requirements-ci.txt` 的钉子逐个对
   （opencv 特殊：pip 报四段 `4.10.0.84`，`cv2.__version__` 只报三段 `4.10.0`，
   取前三段比，不然会因为构建号误报）。

另外还数一下 rows：`--op` / `--case` 名字打错时，`run_compare` 会安静地跑出 0 组、
exit 0，看着像全绿。低于 100 组就报错。

依赖钉死版本而不是 `>=`，同一个道理：对拍项目的 CI 一旦因为上游升级天天报红，
红就没人再看了，等于没有 CI。要升基线，按 `requirements-ci.txt` 顶上那段流程走
（本机先升 → 全量 pytest + run_compare → 确认判定措辞没变 → 同步改钉子和文档基线表）。

产物 `compare-report` 保留 14 天，`if: always()`——红了也要传，出问题时第一手要看的
就是这份报告。门禁脚本自己也有 19 条回归（`tests/test_ci_check.py`），
防止它坏了却一路 exit 0、CI 天天绿其实什么都没查。

**首跑实测**（run `39473982917`，ubuntu-latest / Python 3.12.15，job 68 秒）：
`796 passed`；112 组对拍的判定分布与 Windows 本机**逐字一致**——逐像素精确 60、
浮点噪声级一致 20、一致 8、IoU 合格 5、两边都无边缘 3、仅记录 16。
这条比「CI 绿了」更值钱：它说明手写算子与基准库之间的口径差异不是平台相关的，
不是「只在我这台机器上对得上」。产物 `compare-report` 2.64 MB（报告 + 15 张对比图）。

---

## 4. 使用方法

### 4.1 桌面 GUI 走一遍

1. 左栏算子树选分组 → 算子；需要时切实现变体（如 box_filter 的 vector / naive / 积分图）。
2. 顶部工具条选用例与尺寸，或「打开本地图片」用自己的图。
3. 左栏参数滑块拖动；停手 150ms 防抖后自动重跑。
4. 右侧 2×2 画布：输入 / 我的输出 / 基准输出 / 信息格；信息格内容用下拉切换
   （候选由 `views_for` 按 kind 给出，见 5.2）。
5. 底部状态区读判定：误差 / 耗时 / 口径说明 / 坑提示，一行 headline 概括。
6. 左栏「参数扫描」卡：选一个参数与帧数 → 开始 → 画布放动画；扫完或中途停止后
   「存 GIF」导出循环动图（宽度与抽帧可选）。
7. 工具条「导出」存当前画布 PNG（与屏幕同色）。

### 4.2 Web 版走一遍

侧栏：算子分组 → 算子 → 变体 → 用例 → 尺寸，或上传本地图。
主区：信息格下拉 + 2×2 画布，PNG 直接下载；「参数扫描」展开是折线图 + 明细表 + 循环 GIF。
其余与桌面版一致；判定字与桌面版逐字相同（同一 compare_one）。

### 4.3 给别人演示的推荐动线

median_filter + impulse（逐像素精确，先立信任）→ canny + step_edge 看边缘叠加
（红/青/白三色最直观）→ 参数扫描 σ 放动画存 GIF → region_stats + blobs 看面积柱状 + 质心误差。
四步覆盖判定、可视化、动画、结构化输出四种能力。
---

## 5. 结果解释

### 5.1 判定措辞怎么读、读到之后做什么

| 措辞 | 读法 | 动作 |
|---|---|---|
| 逐像素精确 | 比特级一致，最强结论 | 无需动作 |
| 浮点噪声级一致 | 差在 1e-8 量级，浮点结合序差异 | 无需动作；换 numpy 版本可能微调 |
| 超出容差 | 真差异 | 先看 note 的口径说明，再看信息格定位区域；确认是 bug 还是口径 |
| 一致 / 不一致 | 连通域个数 + 排序面积序列（或框+质心）全等与否 | 不一致时看面积柱状找是哪个域 |
| 仅记录 🟡 | 口径与两家库都不同，数字展示但不判定 | 读 PROGRESS.md 对应行的归因，别当 bug 修 |

### 5.2 信息格视图指南

| 视图 | 看什么 | 典型算子 |
|---|---|---|
| 差值热图 | 差异的空间分布；全黑 = 无差异 | 滤波 / 均衡 / clahe |
| 边缘叠加 | 红 = 仅我的、青 = 仅 cv2、白 = 重合；看边缘定位偏移 | canny / sobel |
| 直方图 + 阈值线 | 两侧直方图叠画 + 各自 Otsu 阈值线位置 | otsu_threshold |
| 累积分布 CDF | 均衡化映射曲线的形状差异 | hist_equalize / counts |
| 面积柱状 + 质心误差 | 连通域面积排序对比 + 每个域的质心偏移 | region_stats |
| 面积分布对比 / 编号差值 | 编号顺序不属于口径，编号差值**仅供参考** | label_components |
| mag / gx / gy | sobel 三分量分别看 | sobel |

### 5.3 误差数字怎么读

`max_abs` 最坏单像素差、`mean_abs` 平均差、`n_over_tol` / `pct_over_tol` 超容差像素数与占比、
`exact` 是否比特级全等。判断顺序：先看 verdict，再看 max_abs 量级（0 / 1e-14 / 1 / 30 是完全不同的故事），
最后用信息格把差异定位到空间区域。max_abs 在 1e-14 量级 = 浮点噪声；在 1 量级 = 取整口径；
在几十量级 = 算法口径不同（如 clahe 的 3.5e+01）。

### 5.4 IoU 怎么读（canny）

边缘图比 IoU 不比逐像素：边缘差半个像素逐像素就全错，但工程上算"同一条边缘"。
当前实测 IoU 0.63~1.00（按用例分布）：step_edge 这类干净边缘接近 1，
纹理复杂用例低一些。canny_smoothed（预平滑 σ=1.4）对两家库的 IoU 都只有 0.00~0.20——
因为三家定义都不同，所以这条标 🟡 仅记录，**不是我的实现坏了**。

### 5.5 耗时怎么读

t_mine / t_ref 都是 repeats 次取最小；slowdown = t_mine / t_ref。
纯 numpy 循环/向量化对 OpenCV 的 C+SIMD，慢 2x~225x 都在当前实测表里（见 PROGRESS.md）。
这张表的意义是"慢得明明白白"（哪个算子慢、慢在哪个复杂度），**不是性能竞赛成绩**；
面试讲法：先讲判定与口径，再讲复杂度归因（如 median 滑窗排序 O(HW·k²) vs cv2 直方图法），
最后讲优化路线（积分图 / 分离核 / 直方图更新），比背数字有价值。

### 5.6 六个"看着像 bug 其实不是"

1. Web 报 `ImportError: cannot import name ... from 'gui.anim'` → 旧服务缓存旧模块，重启 streamlit。
2. 灰度用例 + 形态学算子出警告条 → 二值算子吃灰度图本就没意义，是提示不是错误。
3. 真实图 / 本地图的判定与合成用例不同 → 口径在合成用例上标定，真实图只看可视化与量级。
4. onefile exe 第一次点开慢 → 解包到临时目录；换 --onedir 可解。
5. 扫描 GIF 颜色与单帧略不同 → 全帧统一量化调色板，防闪色设计。
6. 公网隧道链接打不开 → 免注册快速隧道随进程活着，机器休眠/进程被杀即失效，重跑 start_demo 拿新链接。

---

## 6. FAQ

1. **能加自己的算子吗？** 能。myimg 里写纯 numpy 实现 → benchmarks/refs.py 写参照 →
   SPECS 加一条 Spec（定 kind / tol / note）→ params.py 加 PARAMS 与分组 → 补测试。五步，判定自动接入三端。
2. **为什么 clahe 不修到一致？** 三家 tile 插值与 clip  redistribute 口径都不同，追平一家就背离另一家；
   按硬规矩 2 只记录。归因写在 PROGRESS.md 第 4 节。
3. **判定容差能改吗？** 能，SPECS 里每条 Spec 的 tol 就是唯一入口；改完跑全量测试与 CLI 报告核对。
4. **Web 和 GUI 数字会不一样吗？** 不会。两侧都从同一次 compare_one 的 metrics 取数；
   若发现不一样，是 bug，提 issue。
5. **能在没有显示环境的服务器跑吗？** 能。Web 版 matplotlib 走 Agg；CLI 纯计算；
   GUI 才需要显示环境（或 exe）。
6. **测试 796 条都测什么？** 算子数值回归、参数表与签名对齐、engine 全算子不抛异常、
   render 全信息格不崩、扫描取值四规则、主题色 token、GIF 组帧、web 隔离 tkinter、
   打包路径四态与 build_exe 命令行捕获等。
7. **换 OpenCV 版本会怎样？** 浮点噪声级差异可能变化（verdict 仍在容差内）；
   大版本行为变化会体现在 compare.md 的 diff 里，这正是留 json 的原因。
8. **引用/转载？** MIT License（见 README）。

---

## 7. 附录

### 7.1 算子总表（2026-10-18 实测口径）

| 算子 | 分组 | 变体 | kind | 容差 | 当前判定 | 慢几倍 |
|---|---|---|---|---|---|---|
| box_filter | 滤波 | vector / naive | image | 1e-8 | 浮点噪声级一致 (2.8e-14) | 30x |
| box_filter_integral | 滤波 | 积分图 | image | 1e-8 | 浮点噪声级一致 (2.8e-14) | 12x |
| gaussian_filter | 滤波 | vector / naive | image | 1e-8 | 浮点噪声级一致 (2.3e-13) | 2x |
| median_filter | 滤波 | vector / naive | image | 0 | 逐像素精确 | 28x |
| histogram | 直方图 | - | counts | 0 | 逐像素精确 | 7x |
| otsu_threshold | 直方图 | - | scalar | 1.0 | 逐像素精确 | 22x |
| hist_equalize | 直方图 | - | image | 0 | 逐像素精确 | 4x |
| clahe | 直方图 | - | image | None | 🟡 仅记录 (3.5e+01) | 94x |
| sobel | 边缘 | vector / naive | tuple3 | 1e-8 | 逐像素精确 | 1x |
| canny | 边缘 | - | iou | 按 IoU 判定 | ✅ IoU 0.63~1.00 | 13x |
| canny_smoothed | 边缘 | σ=1.4 | iou | None | 🟡 仅记录 (IoU 0.00~0.20) | 54x |
| erode | 形态学 | - | image | 0 | 逐像素精确 | 32x |
| dilate | 形态学 | - | image | 0 | 逐像素精确 | 35x |
| morph_open | 形态学 | - | image | 0 | 逐像素精确 | 46x |
| morph_close | 形态学 | - | image | 0 | 逐像素精确 | 81x |
| label_components | 连通域 | 4/8 连通 | labels | 0 | 一致 | 225x |
| region_stats | 连通域 | 4/8 连通 | stats | 0 | 一致 | 212x |

### 7.2 术语表

- **对拍**：同一输入喂两套实现，逐像素/逐结构比输出。
- **口径**：一家实现对该算子的定义细节（边界、取整、平局、插值），差异不等于错误。
- **仅记录 🟡**：口径差异已归因存档，展示数字但不进入通过/失败。
- **变体**：同一算子的不同实现路线（滑窗 / 积分图 / 朴素循环），数值应一致、耗时不同。
- **信息格**：2×2 画布右下那一格，专门回答"差在哪、差多少"。
- **扫描**：某参数按序列取值连跑，画动画并可导出 GIF。

### 7.3 文档维护约定

判定口径（SPECS / tol / note）任何改动，同步更新本文 5.1、7.1 与 PROGRESS.md；
新增算子按 FAQ 1 的五步走并在 7.1 补行；三端任一端新增展示能力，在 2.7 补共用模块说明。