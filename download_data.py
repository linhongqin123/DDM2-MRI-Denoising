from dipy.data import get_fnames
from dipy.io.image import load_nifti

print("开始下载 Stanford HARDI 数据集...")
# 这会自动下载到你电脑的默认 dipy 缓存目录（通常是 C盘的用户目录下 ~/.dipy/）
hardi_fname, hardi_bval_fname, hardi_bvec_fname = get_fnames('stanford_hardi')
print(f"\n=======================")
print(f"数据下载成功！请务必复制并记住下面这行路径：")
print(f"{hardi_fname}")
print(f"=======================")