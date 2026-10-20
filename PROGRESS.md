# 进度与验收

更新：2026-10-20 22:42　|　测试：749 passed / 0 failed　|　对拍：112 组（1.26.4 / cv2 4.10.0 / skimage 0.25.2）

图例：✅ 对拍通过　🟡 口径不同、差异已记录不追求一致　❌ 待查

| 算子 | 文件 | 对拍 | 误差 / IoU | myimg ms / OpenCV ms (192×192) | 慢几倍 | 关键点 |
|---|---|---|---|---|---|---|
| `box_filter` | filters.py | ✅ 通过 | max|err| = 2.8e-14 | 1.52 / 0.05 | 30x | 滑窗 / 积分图两版结果相同 |
| `box_filter_integral` | filters.py | ✅ 通过 | max|err| = 2.8e-14 | 0.73 / 0.06 | 12x | O(HW) 与核尺寸无关 |
| `gaussian_filter` | filters.py | ✅ 通过 | max|err| = 2.3e-13 | 0.36 / 0.16 | 2x | 自动核 ±4σ（不是 ±3σ） |
| `median_filter` | filters.py | ✅ 通过 | max|err| = 0 | 13.26 / 0.48 | 28x | 边界 REPLICATE 对齐 cv2 |
| `histogram` | histogram.py | ✅ 通过 | max|err| = 0 | 0.09 / 0.01 | 7x | bincount 整数分桶 |
| `otsu_threshold` | histogram.py | ✅ 通过 | max|err| = 0 | 0.26 / 0.01 | 22x | 递推 O(256)，平局取最小 t |
| `hist_equalize` | histogram.py | ✅ 通过 | max|err| = 0 | 0.17 / 0.04 | 4x | 减 CDF_min + floor(x+0.5) |
| `clahe` | histogram.py | 🟡 仅记录 | max|err| = 3.5e+01 | 5.73 / 0.06 | 94x | 口径不同，只记录（见报告第 4 节） |
| `sobel` | edges.py | ✅ 通过 | max|err| = 0 | 0.21 / 0.15 | 1x | 可分离，与 cv2 精确一致 |
| `canny` | edges.py | ✅ 通过 | IoU 0.63~1.00 | 4.57 / 0.34 | 13x | 4 扇区量化 NMS + 迭代膨胀滞后 |
| `canny_smoothed` | edges.py | 🟡 仅记录 | IoU 0.00~0.20 | 4.73 / 0.09 | 54x | 预平滑版，IoU 只记录不判定 |
| `erode` | morphology.py | ✅ 通过 | max|err| = 0 | 0.74 / 0.02 | 32x | 边界 pad True（cv2 +inf） |
| `dilate` | morphology.py | ✅ 通过 | max|err| = 0 | 0.80 / 0.02 | 35x | 边界 pad False（cv2 -inf） |
| `morph_open` | morphology.py | ✅ 通过 | max|err| = 0 | 1.51 / 0.03 | 46x | 先腐蚀后膨胀 |
| `morph_close` | morphology.py | ✅ 通过 | max|err| = 0 | 2.87 / 0.04 | 81x | 先膨胀后腐蚀 |
| `label_components` | morphology.py | ✅ 通过 | 个数+面积/属性 全中 | 25.42 / 0.11 | 225x | 两遍扫描 + 并查集 |
| `region_stats` | morphology.py | ✅ 通过 | 个数+面积/属性 全中 | 26.33 / 0.12 | 212x | 面积/外接框/质心，按面积降序 |

耗时说明：192×192 上取 3 次最小值；「慢几倍」是向量化版相对 OpenCV 的倍数，
手写 NumPy 慢是预期结果，这张表的意义是量化差距而不是追平。

## 已完成

- [x] 基础设施：`_core.py`（dtype 归一 / 边界填充 / 输入校验）
- [x] 15 个算子 + 4 个朴素对照实现（box/gaussian/median/sobel 各一份 naive）+ 积分图版 box
- [x] 12 个合成用例（含 microscope 荧光显微合成图）+ cv2/skimage 基准封装
- [x] 契约测试 + 数值对拍测试共 663 条，全绿
- [x] 对拍运行器 `benchmarks/run_compare.py`：误差表 / 耗时表 / 差异归因 / 对比图
- [x] 命令行 `examples/cli.py`（list / demo / run）
- [x] 报告模板 `benchmarks/report_template.md`
- [x] 可视化对拍台 `gui/`（Tkinter + matplotlib）：算子树 / 参数滑块 / 实现变体 /
      信息格视图，判定口径复用 run_compare；另加 86 条 GUI 回归测试
      （参数表 vs 签名逐条对齐、engine 全算子、render 全视图、扫描序列、主题色、
      GIF 组装）
- [x] GUI 暗金主题：ttk 全控件 + matplotlib rcParams 对齐 wanzi.me 设计变量
      （`#070707` 底 / `#d8b46a` 强调金），导出 PNG 与屏幕同色
- [x] 参数扫描动画：左栏选参数点 + 帧数，画布逐帧刷新，状态区给区间与空帧说明；
      `params.sweep_sequence` 为纯函数，测试里直接对拍
- [x] HiDPI 布局修复：200% 缩放下 matplotlib 画布裁切（Map 时 device_pixel_ratio
      翻倍撑爆 grid）与 Treeview 行高截字，均已在真机截图确认
- [x] 扫描 GIF 导出：帧顺路缩 960 宽缓存，「存 GIF」导出循环动图；调色板取
      全帧马赛克 + 主题/渲染标准色块，避开帧间闪与量化碾色（PIL 并帧坑：帧间
      全同会被合并，纯色首帧做调色板会把后续帧碾平）
- [x] GIF 导出规格可选：宽度 480/720/960、抽帧 1~4（帧间隔乘抽帧倍数，
      播放速度不掉成快进）；`prepare_gif_frames` 纯函数带回归

## 已知差异（不粉饰，全部写在 compare.md 第 4 节）

- **CLAHE**：最大灰阶差 67（uneven_light）。削峰重分配轮数、tile 余数、插值取样点三处口径
  和 OpenCV 不同，只查自洽性 + 暗块对比度提升。
- **Canny（sigma=0）**：量化 NMS vs OpenCV 亚像素 NMS，IoU 0.63~1.00；干净阶跃 1.000。
  NMS 平局必须一松一严（`mag >= n1 & mag > n2`），放反了 step_edge IoU 直接 0。
- **Canny（sigma=1.4）**：cv2.Canny 没有 sigma 参数，基准链路带 uint8 取整，IoU 无判定意义，
  只记录；验收标准改成「预平滑把噪声边缘压掉几个数量级」。

## 待办

- [ ] 每算子一份 `notes/报告_<算子>.md`（模板在 benchmarks/report_template.md）
- [ ] 48 小时默写：gaussian_kernel1d / otsu / canny NMS / label_components 两遍扫描
      （记录表在 notes/默写记录.md）
- [ ] 中值滤波的 Huang 直方图法（O(HW·256) 与 k 无关）作为进阶练习
- [ ] CLAHE 与 OpenCV 源码逐行对齐（可选，只为把 67 灰阶差归因到具体某一行）
- [ ] 项目 B：DSB2018 细胞核分割预处理流水线；项目 C：NEU 钢材表面缺陷检测
      （见 ../图像算法岗转向_学习与项目计划_2026-10.md）

## 面试复述清单（合上资料能讲清楚才算过）

1. 高斯自动核尺寸为什么是 `2*floor(4σ+0.5)+1`，±3σ 的说法错在哪，差多少。
2. Otsu 类间方差最大化 ⇔ 类内方差最小化的推导；递推怎么做到 O(256)。
3. Canny 四步；NMS 平局为什么要一松一严；滞后连接为什么等价于「膨胀 ∩ 候选」的不动点。
4. 腐蚀 pad True、膨胀 pad False 的道理（cv2 的 ±inf 边界）。
5. 积分图怎么把 box filter 做到 O(HW)，dtype 为什么必须 float64。
6. 两遍扫描连通域：等价表 + 并查集，为什么编号顺序和 cv2 不同也能对拍。
7. 可分离卷积为什么只对目标轴 pad（两个方向都 pad 尺寸就回不去了）。
