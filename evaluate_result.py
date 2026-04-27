import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import os
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

# 1. 路径配置 (请核对这些路径是否正确)
# 原始干净数据集路径
path_raw_data = r"C:\Users\ASUS\.dipy\stanford_hardi\HARDI150.nii.gz"
# 你刚才跑出来的去噪结果文件 (0.npy)
path_denoised_npy = r"D:\PPT\DDM²\DDM2\experiments\hardi150_260427_181651\results\0\0.npy"

def normalize(data):
    """归一化到 0-1"""
    return (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)

def main():
    print("正在读取原始数据集和去噪结果...")
    
    # 1. 加载原始干净数据 (Ground Truth)
    # 根据你的日志，数据维度是 (81, 106, 76, 160)
    raw_img = nib.load(path_raw_data).get_fdata()
    
    # 2. 加载你的去噪结果
    denoised_data = np.load(path_denoised_npy)
    
    # 3. 提取对应的 Ground Truth 切片
    # 这里的关键是找到对应关系。通常 test.py 的第一个结果 (0.npy) 
    # 对应的是测试集的第一张图。在你的配置中是 volume_idx=10, slice_idx=0 左右开始
    # 我们假设它是你配置文件里指定的那个位置 (Volume 40, Slice 40)
    # 如果不准，你可以微调这两个数字
    gt_slice = raw_img[:, :, 8, 15] 
    
    # 4. 统一尺寸和归一化
    # 原始是 (81, 106)，你的结果也是 (81, 106)
    img_gt = normalize(gt_slice)
    img_denoised = normalize(denoised_data)
    
    # 5. 模拟一个加噪图 (用于对比显示)
    # 扩散模型的输入通常带有强噪声
    img_noisy = img_gt + np.random.normal(0, 0.1, img_gt.shape)
    img_noisy = normalize(img_noisy)

    # ==========================================
    # 6. 计算真实指标
    # ==========================================
    psnr_val = compare_psnr(img_gt, img_denoised, data_range=1.0)
    ssim_val = compare_ssim(img_gt, img_denoised, data_range=1.0)

    print(f"\n✅ 真实评估完成！")
    print(f"PSNR: {psnr_val:.2f} dB")
    print(f"SSIM: {ssim_val:.4f}")

    # ==========================================
    # 7. 绘制三联对比图
    # ==========================================
    plt.figure(figsize=(15, 5), facecolor='black')
    
    plt.subplot(1, 3, 1)
    plt.imshow(img_noisy, cmap='gray')
    plt.title("Noisy Input (Simulated)", color='white')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(img_denoised, cmap='gray')
    plt.title(f"DDM² Denoised\nPSNR: {psnr_val:.2f}dB", color='springgreen', fontsize=14)
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(img_gt, cmap='gray')
    plt.title("Ground Truth (Original)", color='white')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig("Real_Evaluation_Result.png", dpi=300)
    print("\n对比图已保存为 Real_Evaluation_Result.png")
    plt.show()

if __name__ == "__main__":
    main()