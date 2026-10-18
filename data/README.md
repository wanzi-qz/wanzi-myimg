# data/

真实图片放这个目录，`benchmarks/cases.load_real_images()` 会自动收进来。

约定：

1. **数值对拍不用真实图。** 合成用例固定种子、可复现；真实图一换文件，
   compare.md 里所有误差数字都得重跑，报告就失去可比性。
2. 真实图只用于两件事：报告里的效果对比图、examples/cli.py 的演示。
3. 支持格式：png / jpg / jpeg / bmp / tif / tiff，读入自动转灰度并缩到长边 512 以内。
4. 本目录除本文件外全部 gitignore，不跟仓库上传（版权和体积）。

想加图就把文件丢进来，不用改任何代码。
