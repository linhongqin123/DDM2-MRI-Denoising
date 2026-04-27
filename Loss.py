import matplotlib.pyplot as plt
import re
import numpy as np

# 1. 模拟你的日志数据（如果是从文件读取，可以用 open('train.log').read()）
log_content = """
26-04-27 17:17:07.089 - INFO: <epoch:  1, iter:     100> l_pix: 5.2024e-02
26-04-27 17:17:16.115 - INFO: <epoch:  1, iter:     200> l_pix: 2.5121e-02
26-04-27 17:17:24.953 - INFO: <epoch:  1, iter:     300> l_pix: 1.6742e-02
26-04-27 17:17:33.594 - INFO: <epoch:  1, iter:     400> l_pix: 1.6614e-02
26-04-27 17:17:42.384 - INFO: <epoch:  1, iter:     500> l_pix: 1.5739e-02
26-04-27 17:17:51.324 - INFO: <epoch:  1, iter:     600> l_pix: 1.1569e-02
26-04-27 17:18:01.106 - INFO: <epoch:  1, iter:     700> l_pix: 1.7872e-02
26-04-27 17:24:02.160 - INFO: <epoch:  2, iter:   4,000> l_pix: 7.4747e-03
26-04-27 17:33:33.210 - INFO: <epoch:  4, iter:   9,100> l_pix: 3.8437e-03
""" # 这里只是示例，你可以把你的全部日志粘贴进来

# 2. 使用正则表达式解析数据
iters = []
losses = []

# 匹配格式：iter: 100> l_pix: 5.2024e-02
pattern = r"iter:\s+([\d,]+)>\s+l_pix:\s+([\d\.e-]+)"
matches = re.findall(pattern, log_content)

for m in matches:
    # 去掉迭代次数里的逗号并转为整数
    it = int(m[0].replace(',', ''))
    loss = float(m[1])
    iters.append(it)
    losses.append(loss)

# 3. 绘图
plt.figure(figsize=(10, 6))
plt.plot(iters, losses, color='#1f77b4', marker='o', markersize=4, linestyle='-', linewidth=2, label='$L_1$ Loss')

# 4. 美化图表
plt.title('Training Loss Curve (Stage 3: Diffusion Model)', fontsize=14, fontweight='bold')
plt.xlabel('Iterations', fontsize=12)
plt.ylabel('Loss Value ($L_1$)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=12)

# 如果 Loss 下降非常快，可以考虑开启 y 轴的对数坐标，这样曲线更清晰
# plt.yscale('log') 

# 5. 保存并显示
plt.savefig("Training_Loss_Curve.png", dpi=300)
plt.show()

print(f"成功提取了 {len(iters)} 个采样点，曲线图已保存为 Training_Loss_Curve.png")