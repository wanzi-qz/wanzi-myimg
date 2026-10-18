# myimg 对拍报告

生成时间：2026-10-18 23:39:01  
命令：`python benchmarks/run_compare.py --all --figures --size 192`  
环境：Python 3.12.3 / numpy 1.26.4 / opencv-python 4.10.0 / scikit-image 0.25.2 / Windows-11-10.0.26300-SP0

## 1. 结论速览

- **逐像素精确一致**（max|err| = 0，全部用例全中）：dilate, erode, hist_equalize, histogram, label_components, median_filter, morph_close, morph_open, otsu_threshold, region_stats, sobel
- **浮点噪声级一致**（max|err| ≤ 容差，非零）：box_filter, box_filter_integral, gaussian_filter
- **按 IoU 判定**（边缘位置允许半像素偏移）：`canny`，IoU 0.635 ~ 1.000
- **只记录不判定**（IoU 在这里没有判定意义，归因见第 4 节）：`canny_smoothed`，IoU 0.000 ~ 0.197
- **口径不同，只记录差异不追求一致**：canny_smoothed, clahe
- **超出容差 / 待查**：无

## 2. 误差表

| 算子 | 用例 | max\|err\| | mean\|err\| | 关键数字 | 判定 |
|---|---|---|---|---|---|
| `box_filter` | flat | 0 | 0 |  | 逐像素精确 |
| `box_filter` | ramp | 1.42e-14 | 1.19e-15 |  | 浮点噪声级一致 |
| `box_filter` | step_edge | 0 | 0 |  | 逐像素精确 |
| `box_filter` | impulse | 2.84e-14 | 2.13e-15 |  | 浮点噪声级一致 |
| `box_filter` | gauss_noise | 2.84e-14 | 2.73e-15 |  | 浮点噪声级一致 |
| `box_filter` | uneven_light | 2.84e-14 | 2.62e-15 |  | 浮点噪声级一致 |
| `box_filter` | microscope | 2.84e-14 | 1.09e-15 |  | 浮点噪声级一致 |
| `box_filter` | random_uint8 | 2.84e-14 | 2.59e-15 |  | 浮点噪声级一致 |
| `box_filter_integral` | flat | 0 | 0 |  | 逐像素精确 |
| `box_filter_integral` | ramp | 1.42e-14 | 1.19e-15 |  | 浮点噪声级一致 |
| `box_filter_integral` | step_edge | 0 | 0 |  | 逐像素精确 |
| `box_filter_integral` | impulse | 2.84e-14 | 2.13e-15 |  | 浮点噪声级一致 |
| `box_filter_integral` | gauss_noise | 2.84e-14 | 2.73e-15 |  | 浮点噪声级一致 |
| `box_filter_integral` | uneven_light | 2.84e-14 | 2.62e-15 |  | 浮点噪声级一致 |
| `box_filter_integral` | microscope | 2.84e-14 | 1.09e-15 |  | 浮点噪声级一致 |
| `box_filter_integral` | random_uint8 | 2.84e-14 | 2.59e-15 |  | 浮点噪声级一致 |
| `gaussian_filter` | flat | 4.26e-14 | 4.26e-14 |  | 浮点噪声级一致 |
| `gaussian_filter` | ramp | 1.71e-13 | 5.73e-14 |  | 浮点噪声级一致 |
| `gaussian_filter` | step_edge | 1.71e-13 | 1.75e-14 |  | 浮点噪声级一致 |
| `gaussian_filter` | impulse | 2.27e-13 | 5.91e-14 |  | 浮点噪声级一致 |
| `gaussian_filter` | gauss_noise | 1.99e-13 | 6.27e-14 |  | 浮点噪声级一致 |
| `gaussian_filter` | uneven_light | 1.99e-13 | 5.97e-14 |  | 浮点噪声级一致 |
| `gaussian_filter` | microscope | 1.99e-13 | 2.49e-14 |  | 浮点噪声级一致 |
| `gaussian_filter` | random_uint8 | 1.71e-13 | 6.12e-14 |  | 浮点噪声级一致 |
| `median_filter` | flat | 0 | 0 |  | 逐像素精确 |
| `median_filter` | ramp | 0 | 0 |  | 逐像素精确 |
| `median_filter` | step_edge | 0 | 0 |  | 逐像素精确 |
| `median_filter` | impulse | 0 | 0 |  | 逐像素精确 |
| `median_filter` | gauss_noise | 0 | 0 |  | 逐像素精确 |
| `median_filter` | uneven_light | 0 | 0 |  | 逐像素精确 |
| `median_filter` | microscope | 0 | 0 |  | 逐像素精确 |
| `median_filter` | random_uint8 | 0 | 0 |  | 逐像素精确 |
| `histogram` | flat | 0 | 0 |  | 逐像素精确 |
| `histogram` | ramp | 0 | 0 |  | 逐像素精确 |
| `histogram` | step_edge | 0 | 0 |  | 逐像素精确 |
| `histogram` | impulse | 0 | 0 |  | 逐像素精确 |
| `histogram` | gauss_noise | 0 | 0 |  | 逐像素精确 |
| `histogram` | uneven_light | 0 | 0 |  | 逐像素精确 |
| `histogram` | microscope | 0 | 0 |  | 逐像素精确 |
| `histogram` | random_uint8 | 0 | 0 |  | 逐像素精确 |
| `otsu_threshold` | flat | 0 | 0 | mine=0 / cv2=0 | 逐像素精确 |
| `otsu_threshold` | ramp | 0 | 0 | mine=126 / cv2=126 | 逐像素精确 |
| `otsu_threshold` | step_edge | 0 | 0 | mine=40 / cv2=40 | 逐像素精确 |
| `otsu_threshold` | impulse | 0 | 0 | mine=126 / cv2=126 | 逐像素精确 |
| `otsu_threshold` | gauss_noise | 0 | 0 | mine=130 / cv2=130 | 逐像素精确 |
| `otsu_threshold` | uneven_light | 0 | 0 | mine=125 / cv2=125 | 逐像素精确 |
| `otsu_threshold` | microscope | 0 | 0 | mine=83 / cv2=83 | 逐像素精确 |
| `otsu_threshold` | random_uint8 | 0 | 0 | mine=127 / cv2=127 | 逐像素精确 |
| `hist_equalize` | flat | 0 | 0 |  | 逐像素精确 |
| `hist_equalize` | ramp | 0 | 0 |  | 逐像素精确 |
| `hist_equalize` | step_edge | 0 | 0 |  | 逐像素精确 |
| `hist_equalize` | impulse | 0 | 0 |  | 逐像素精确 |
| `hist_equalize` | gauss_noise | 0 | 0 |  | 逐像素精确 |
| `hist_equalize` | uneven_light | 0 | 0 |  | 逐像素精确 |
| `hist_equalize` | microscope | 0 | 0 |  | 逐像素精确 |
| `hist_equalize` | random_uint8 | 0 | 0 |  | 逐像素精确 |
| `clahe` | flat | 2.0000 | 2.0000 |  | 仅记录 |
| `clahe` | ramp | 10.0000 | 6.5687 |  | 仅记录 |
| `clahe` | step_edge | 2.0000 | 1.4688 |  | 仅记录 |
| `clahe` | impulse | 11.0000 | 7.5562 |  | 仅记录 |
| `clahe` | gauss_noise | 28.0000 | 14.7619 |  | 仅记录 |
| `clahe` | uneven_light | 35.0000 | 16.7771 |  | 仅记录 |
| `clahe` | microscope | 26.0000 | 7.6797 |  | 仅记录 |
| `clahe` | random_uint8 | 3.0000 | 5.14e-01 |  | 仅记录 |
| `sobel` | flat | 0 | 0 | gx 0 / gy 0 / mag 0 | 逐像素精确 |
| `sobel` | ramp | 0 | 0 | gx 0 / gy 0 / mag 0 | 逐像素精确 |
| `sobel` | step_edge | 0 | 0 | gx 0 / gy 0 / mag 0 | 逐像素精确 |
| `sobel` | impulse | 0 | 0 | gx 0 / gy 0 / mag 0 | 逐像素精确 |
| `sobel` | gauss_noise | 0 | 0 | gx 0 / gy 0 / mag 0 | 逐像素精确 |
| `sobel` | uneven_light | 0 | 0 | gx 0 / gy 0 / mag 0 | 逐像素精确 |
| `sobel` | microscope | 0 | 0 | gx 0 / gy 0 / mag 0 | 逐像素精确 |
| `sobel` | random_uint8 | 0 | 0 | gx 0 / gy 0 / mag 0 | 逐像素精确 |
| `canny` | flat | 0 | 0 | IoU=1.000，边缘点 0 vs cv2 0 | 两边都无边缘 |
| `canny` | ramp | 0 | 0 | IoU=1.000，边缘点 0 vs cv2 0 | 两边都无边缘 |
| `canny` | step_edge | 0 | 0 | IoU=1.000，边缘点 192 vs cv2 192 | IoU 合格 |
| `canny` | impulse | 255.0000 | 9.0340 | IoU=0.823，边缘点 6715 vs cv2 6749 | IoU 合格 |
| `canny` | gauss_noise | 255.0000 | 37.6095 | IoU=0.642，边缘点 11922 vs cv2 13021 | IoU 合格 |
| `canny` | uneven_light | 0 | 0 | IoU=1.000，边缘点 0 vs cv2 0 | 两边都无边缘 |
| `canny` | microscope | 255.0000 | 38.5779 | IoU=0.635，边缘点 12181 vs cv2 12774 | IoU 合格 |
| `canny` | random_uint8 | 255.0000 | 35.0085 | IoU=0.688，边缘点 13622 vs cv2 13765 | IoU 合格 |
| `canny_smoothed` | flat | 0 | 0 | IoU=1.000，边缘点 0 vs cv2 0 | 仅记录 |
| `canny_smoothed` | ramp | 0 | 0 | IoU=1.000，边缘点 0 vs cv2 0 | 仅记录 |
| `canny_smoothed` | step_edge | 255.0000 | 2.6562 | IoU=0.000，边缘点 192 vs cv2 192 | 仅记录 |
| `canny_smoothed` | impulse | 0 | 0 | IoU=1.000，边缘点 0 vs cv2 0 | 仅记录 |
| `canny_smoothed` | gauss_noise | 0 | 0 | IoU=1.000，边缘点 0 vs cv2 0 | 仅记录 |
| `canny_smoothed` | uneven_light | 0 | 0 | IoU=1.000，边缘点 0 vs cv2 0 | 仅记录 |
| `canny_smoothed` | microscope | 255.0000 | 1.5495 | IoU=0.197，边缘点 114 vs cv2 220 | 仅记录 |
| `canny_smoothed` | random_uint8 | 255.0000 | 47.4320 | IoU=0.119，边缘点 1111 vs cv2 7592 | 仅记录 |
| `erode` | blobs | 0 | 0 |  | 逐像素精确 |
| `erode` | holes | 0 | 0 |  | 逐像素精确 |
| `erode` | checker | 0 | 0 |  | 逐像素精确 |
| `erode` | two_touching | 0 | 0 |  | 逐像素精确 |
| `dilate` | blobs | 0 | 0 |  | 逐像素精确 |
| `dilate` | holes | 0 | 0 |  | 逐像素精确 |
| `dilate` | checker | 0 | 0 |  | 逐像素精确 |
| `dilate` | two_touching | 0 | 0 |  | 逐像素精确 |
| `morph_open` | blobs | 0 | 0 |  | 逐像素精确 |
| `morph_open` | holes | 0 | 0 |  | 逐像素精确 |
| `morph_open` | checker | 0 | 0 |  | 逐像素精确 |
| `morph_open` | two_touching | 0 | 0 |  | 逐像素精确 |
| `morph_close` | blobs | 0 | 0 |  | 逐像素精确 |
| `morph_close` | holes | 0 | 0 |  | 逐像素精确 |
| `morph_close` | checker | 0 | 0 |  | 逐像素精确 |
| `morph_close` | two_touching | 0 | 0 |  | 逐像素精确 |
| `label_components` | blobs | 0 | 0 | n=13 vs cv2 13 | 一致 |
| `label_components` | holes | 0 | 0 | n=1 vs cv2 1 | 一致 |
| `label_components` | checker | 0 | 0 | n=1 vs cv2 1 | 一致 |
| `label_components` | two_touching | 0 | 0 | n=2 vs cv2 2 | 一致 |
| `region_stats` | blobs | 0 | 0 | 13 个域，质心最大差 0.0e+00 | 一致 |
| `region_stats` | holes | 0 | 0 | 1 个域，质心最大差 0.0e+00 | 一致 |
| `region_stats` | checker | 0 | 0 | 1 个域，质心最大差 0.0e+00 | 一致 |
| `region_stats` | two_touching | 0 | 0 | 2 个域，质心最大差 0.0e+00 | 一致 |

## 3. 耗时表

向量化与 OpenCV 都在 192×192 上计时（取 3 次最小值）；朴素版在 64×64 上跑一次，所以两者比的是**单像素耗时**，不是绝对毫秒数。

| 算子 | 朴素 (ms) | 向量化 (ms) | OpenCV (ms) | 朴素→向量化 | 向量化 / OpenCV |
|---|---|---|---|---|---|
| `box_filter` | 13.3 | 1.55 | 0.06 | 77.3x | 24.4x 慢 |
| `box_filter_integral` | — | 1.01 | 0.07 | — | 15.5x 慢 |
| `gaussian_filter` | 11.9 | 0.36 | 0.17 | 294.1x | 2.2x 慢 |
| `median_filter` | 43.6 | 15.38 | 0.48 | 25.5x | 32.1x 慢 |
| `histogram` | — | 0.09 | 0.01 | — | 6.8x 慢 |
| `otsu_threshold` | — | 0.26 | 0.01 | — | 24.7x 慢 |
| `hist_equalize` | — | 0.16 | 0.03 | — | 5.0x 慢 |
| `clahe` | — | 6.26 | 0.06 | — | 103.9x 慢 |
| `sobel` | 26.0 | 0.20 | 0.16 | 1176.1x | 1.2x 慢 |
| `canny` | — | 4.83 | 0.29 | — | 16.6x 慢 |
| `canny_smoothed` | — | 5.10 | 0.08 | — | 60.2x 慢 |
| `erode` | — | 0.76 | 0.03 | — | 26.5x 慢 |
| `dilate` | — | 0.91 | 0.02 | — | 36.8x 慢 |
| `morph_open` | — | 1.68 | 0.03 | — | 51.5x 慢 |
| `morph_close` | — | 2.81 | 0.04 | — | 75.8x 慢 |
| `label_components` | — | 26.11 | 0.09 | — | 274.8x 慢 |
| `region_stats` | — | 26.29 | 0.13 | — | 200.7x 慢 |

手写版比 OpenCV 慢是预期结果：OpenCV 走 SIMD、有定点优化路径，中值滤波还可能用 Huang 直方图法。这张表的作用是把这个差距量化，而不是假装手写版能打赢。

## 4. 差异归因

### `clahe`

**现象**：最大灰阶差 35.0，平均灰阶差 7.17，所有用例都对不上，但 myimg 自己两次跑结果完全一样。

**原因**：CLAHE 的规范实现里有三处自由度，OpenCV 的选择没有写在文档里，只能从代码反推：
1. 削峰阈值 `clip_limit * tile_pix / 256` 之后，超出部分怎么摊回去 —— OpenCV 是「一次性把超出总量平均分到 256 个桶，再重复几轮」，每轮的整数取整位置不同，结果就不同；
2. tile 划分不整除时的余数摊给谁；
3. 双线性插值的取样点算在 tile 中心还是 tile 角点。

**处理**：不追求逐像素一致，测试只查自洽性（dtype / 形状 / 确定性 / 暗块局部对比度确实被拉开）。这条差异如实记在这里，不粉饰。

### `canny`

**现象**：IoU 逐用例为 flat 1.000，ramp 1.000，step_edge 1.000，impulse 0.823，gauss_noise 0.642，uneven_light 1.000，microscope 0.635，random_uint8 0.688。干净阶跃边缘能到 1.000，带噪声的用例只有 0.6 左右。

**原因**：非极大值抑制的实现不同。OpenCV 沿梯度方向对相邻两个像素的幅值做线性插值再比大小（亚像素 NMS），myimg 把方向量化到 0/45/90/135 四个扇区后直接比邻居。量化会让 22.5° 附近的边缘选错比较对象，结果是边缘位置整体偏移半个到一个像素 —— IoU 对偏移极其敏感，两个像素宽、错开一列的两条边，IoU 就是 0。

**另一个坑（已修）**：NMS 的平局处理。阶跃边缘上相邻两列幅值完全相等，两侧都用 `>=` 会留下两像素宽的边，都用 `>` 会整条边被压掉，必须一松一严：`keep = (mag >= n1) & (mag > n2)`。松紧放在哪一侧决定边缘落在阶跃的左列还是右列，放反了 step_edge 的 IoU 直接是 0。

### `canny_smoothed`

**现象**：边缘像素数 myimg {'flat': 0, 'ramp': 0, 'step_edge': 192, 'impulse': 0, 'gauss_noise': 0, 'uneven_light': 0, 'microscope': 114, 'random_uint8': 1111} vs cv2 流水线 {'flat': 0, 'ramp': 0, 'step_edge': 192, 'impulse': 0, 'gauss_noise': 0, 'uneven_light': 0, 'microscope': 220, 'random_uint8': 7592}，IoU 在带噪声的用例上只有 0.1 ~ 0.3。

**为什么不对拍**：`cv2.Canny` 没有 sigma 参数，工业用法是`GaussianBlur(uint8) -> Canny(uint8)`。这条链路上有两处口径和 myimg 不同：一是平滑结果被取整回 uint8（梯度被量化），二是 OpenCV 的 NMS 沿梯度方向做亚像素插值、myimg 用 4 扇区量化。平滑后的噪声场梯度很弱，这两处差异会被阈值放大成「检不检得出边缘」的差别，IoU 失去判定意义。

**这个算子真正的验收标准**（在 tests/test_edges.py 里）：microscope 用例（128×128）上 sigma=0 检出 5327 个噪声边缘像素 / 318 个碎片，sigma=1.4 只剩 186 个 / 31 个连通块；纯噪声图（gauss_noise）上 sigma=1.4 一个都检不出。预平滑的价值用这组数字说，不用 IoU 说。

## 5. 已核对的口径事实

- `np.pad(mode='reflect')` == `cv2.BORDER_REFLECT_101`；`'symmetric'` == `BORDER_REFLECT`；`'edge'` == `BORDER_REPLICATE`。而 scipy / skimage 的 `'reflect'` 指含边缘的对称填充（= `BORDER_REFLECT`），`'mirror'` 才是 `REFLECT_101` —— 三家命名互不兼容。
- OpenCV 高斯的自动核尺寸是 `ksize = 2 * floor(4σ + 0.5) + 1`（±4σ 截断），**不是**网上流传的 ±3σ。实测命中表：σ = 0.3/0.5/0.8/1.0/1.4/1.5/2.0/2.5/3.0 → ksize = 3/5/7/9/13/13/17/21/25。用 ±3σ 时 σ=1.0 会得到 7 而 OpenCV 是 9，对拍直接不过。
- `cv2.boxFilter` 不接受 `BORDER_WRAP`（4.10.0 实测抛 cv2.error）。要对拍环绕边界，只能用 numpy 先把环绕 padding 补出来，再让 cv2 做常数边界均值后裁回去。
- `cv2.erode` 默认边界是 `+inf`、`cv2.dilate` 是 `-inf`，不是 0。腐蚀 pad True、膨胀 pad False；搞反了错误只出现在图像四周一圈，很难发现。
- `cv2.medianBlur` 的边界固定 REPLICATE，且只接受 uint8（ksize ≤ 5 时也收 float32）。
- `np.round` 是银行家舍入（round(0.5)=0），`cvRound` 是四舍五入。整数值上会差 ±1 灰阶，直方图均衡取整统一用 `floor(x + 0.5)`。
- 直方图均衡必须减 `CDF_min`（最低出现灰阶的累积计数），否则暗部被整体抬亮。实测不减的版本在 microscope 用例上和 cv2 差到 9 灰阶，减了之后逐像素一致。
- `skimage.sobel(img / 255)` == `cv2 幅值 / 255 / (4√2)`，精确到 1.8e-16 —— skimage 把平滑核除了 4、求导核除了 √2。
- Otsu 阈值在全部 12 个用例上和 `cv2.threshold(THRESH_OTSU)` **完全相等**（不是差 1 灰阶）。测试仍保留 ±1 容差，因为类间方差出现并列最大值时两家的平局约定不同，换一批图就可能分叉；另外单独写了一条「递推实现 vs 暴力遍历定义式」的测试兜底。
- `cv2.GaussianBlur` 不接受 `ddepth` 参数，输出深度跟随输入，要拿无损结果必须喂 float64（喂 uint8 会先被截断成整数）。
