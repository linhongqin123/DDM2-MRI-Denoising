# DDM²: 基于扩散概率模型的自监督 MRI 图像去噪

本项目实现了 **DDM² (Diffusion Denoising Model for MRI)**，这是一个专为医学图像设计的自监督生成式去噪框架。该框架无需干净的 Ground Truth 图像作为标签，通过扩散模型（Diffusion Model）的生成能力和马尔可夫链状态匹配（Markov Chain State Matching）机制，直接从带噪图像中学习如何恢复高保真的解剖结构。

##  项目亮点

* **全流程自监督**：解决了医学影像中难以获取配对干净数据（Noise-free data）的痛点。
* **三阶段训练机制**：
    1.  **Stage 1 (Noise Model)**: 训练噪声估计网络，学习数据的噪声先验。
    2.  **Stage 2 (Markov Chain Matching)**: 精确匹配每一张切片的初始去噪时间步 $T$。
    3.  **Stage 3 (Joint Training)**: 扩散模型与噪声模型联合优化，实现结构还原。
* **工程优化**：
    * 修复了原代码在测试阶段对 `matched_state` 的逻辑处理漏洞（`KeyError` 修复）。
    * 开发了针对 4D 医学数据集（NIfTI 格式）的**暴力搜索匹配脚本**，解决了评估时的索引对齐难题。

##  环境配置

* **硬件**: NVIDIA GeForce RTX 5070 Laptop GPU (8GB VRAM)
* **系统/环境**: Windows 11 / CUDA 12.8 / PyTorch (Nightly)
* **依赖库**: `numpy`, `torch`, `nibabel`, `scikit-image`, `matplotlib`

```bash
# 创建并激活环境
conda create -n ddm2_env python=3.11
conda activate ddm2_env
# 安装核心依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
pip install nibabel scikit-image matplotlib
```

##  实验数据 (Stanford HARDI150)

经过 10,000 次迭代训练（占预设总进度的 10%），模型在验证集上展现了显著的去噪趋势：

| 评估指标 | 加噪输入 (Noisy) | DDM² 处理后 (Denoised) |
| :--- | :--- | :--- |
| **PSNR** | 12.92 dB | **17.79 dB** |
| **SSIM** | 0.0600 | **0.1727** |

> **注**：在测试阶段，通过暴力搜索校准，确认测试切片对应原始数据的 `Volume 15, Slice 8`。

##  快速开始

### 1. 模型测试
在完成训练阶段后，运行以下命令进行验证：
```bash
python test.py -p val -c config/hardi_150.json
```

### 2. 结果评估与可视化
运行自定义评估脚本，自动对齐原图并生成对比图：
```bash
python final_evaluate.py
```

##  训练过程
训练过程中 $L_1$ Loss 从初始的 $5.20 \times 10^{-2}$ 平滑下降至 $3.84 \times 10^{-3}$，证明了扩散模型对 MRI 图像全局结构捕捉的有效性。



##  源码修复说明
针对 `model/mri_modules/diffusion.py` 中的 Bug，本项目对 `denoise` 函数进行了逻辑容错处理：
```python
if 'matched_state' in x_in:
    matched_state = int(x_in['matched_state'][0].item())
else:
    # 针对测试阶段未提供状态文件时的默认处理
    matched_state = self.num_timesteps - 1 
```
