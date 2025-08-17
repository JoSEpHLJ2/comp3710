import torch
import numpy as np

print("PyTorch Version:", torch.__version__)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# grid for computing image, subdivide the space
X, Y = np.mgrid[-4.0:4:0.01, -4.0:4:0.01]

# load into PyTorch tensors
x = torch.Tensor(X)
y = torch.Tensor(Y)

# transfer to the GPU device
x = x.to(device)
y = y.to(device)
# 高斯函数
gaussian = torch.exp(-(x**2 + y**2) / 2.0)

# 正弦函数
sine = torch.sin(5 * x + 5 * y)

# 余弦函数
cosine = torch.cos(5 * x + 5 * y)

# Gabor 滤波器 = 高斯 × 正弦 / 余弦
gabor_sin = gaussian * sine
gabor_cos = gaussian * cosine

#plot
import matplotlib.pyplot as plt

# 绘制结果
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(gabor_sin.cpu().numpy(), extent=[-4, 4, -4, 4], cmap='seismic')
plt.title("Gabor Filter (Gaussian × Sine)")

plt.subplot(1, 2, 2)
plt.imshow(gabor_cos.cpu().numpy(), extent=[-4, 4, -4, 4], cmap='seismic')
plt.title("Gabor Filter (Gaussian × Cosine)")

plt.tight_layout()
plt.show()
plt.savefig("test1.3.png", dpi=300, bbox_inches="tight")  # 保存到文件