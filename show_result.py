import numpy as np
import matplotlib.pyplot as plt

# 替换成你文件夹里真实的文件名
data = np.load(r'D:\PPT\DDM²\DDM2\experiments\hardi150_260427_181651\results\0\0.npy')

# 假设数据是 [C, H, W]，取第一个通道显示
if len(data.shape) == 3:
    plt.imshow(data[0], cmap='gray')
else:
    plt.imshow(data, cmap='gray')

plt.title("DDM2 Denoised Result")
plt.show()