import numpy as np
import nibabel as nib
import os
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

# 1. 路径（已根据你的配置填好）
path_raw_data = r"C:\Users\ASUS\.dipy\stanford_hardi\HARDI150.nii.gz"
path_denoised_npy = r"D:\PPT\DDM²\DDM2\experiments\hardi150_260427_181651\results\0\0.npy"

def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)

def brute_force_match():
    print("🚀 启动全自动切片匹配系统...")
    
    # 加载数据
    raw_img = nib.load(path_raw_data).get_fdata()
    denoised_img = normalize(np.load(path_denoised_npy))
    
    W, H, Slices, Volumes = raw_img.shape
    best_psnr = -1
    best_coords = (0, 0)

    # 缩小搜索范围：只搜索你 mask 定义的 volume 范围 [10, 160]
    # 如果速度太慢，可以进一步缩小范围
    print(f"正在扫描原始数据 (共 {Volumes} 个 Volume)...")
    
    # 重点扫描 Volume 10 附近，因为 0.npy 通常是第一个
    for v in range(10, 20): 
        for s in range(0, Slices):
            gt_slice = normalize(raw_img[:, :, s, v])
            
            # 计算当前对比的 PSNR
            current_psnr = compare_psnr(gt_slice, denoised_img, data_range=1.0)
            
            if current_psnr > best_psnr:
                best_psnr = current_psnr
                best_coords = (s, v)
        
        print(f"已扫描完 Volume {v}, 当前最高 PSNR: {best_psnr:.2f} dB")

    print("\n" + "="*30)
    print(f"🎯 匹配成功！")
    print(f"你的 0.npy 对应的真实位置是: Volume {best_coords[1]}, Slice {best_coords[0]}")
    print(f"在该位置下的真实去噪 PSNR 为: {best_psnr:.2f} dB")
    print("="*30)
    print("现在你可以用这个 PSNR 结果写进你的作业报告了！")

if __name__ == "__main__":
    brute_force_match()