import torch
import numpy as np
import matplotlib.pyplot as plt

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create a grid in the complex plane
Y, X = np.mgrid[-1.3:1.3:0.005, -2:1:0.005]

# Load into PyTorch tensors
x = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(Y, dtype=torch.float32)
z = torch.complex(x, y)  # constant c
zs = z.clone()           # z_0 = c
ns = torch.zeros_like(z) # divergence counter

# Transfer to GPU if available
z, zs, ns = z.to(device), zs.to(device), ns.to(device)

# Mandelbrot Set iterations
for i in range(200):
    # Compute the new values of z: z^2 + c
    zs_ = zs * zs + z
    # Have we diverged with this new value?
    not_diverged = torch.abs(zs_) < 4.0
    # Update divergence counter
    ns += not_diverged
    # Prepare for next iteration
    zs = zs_

# Color mapping function
def processFractal(a):
    a_cyclic = (6.28 * a / 20.0).reshape(list(a.shape) + [1])
    img = np.concatenate([
        10 + 20 * np.cos(a_cyclic),
        30 + 50 * np.sin(a_cyclic),
        155 - 80 * np.cos(a_cyclic)
    ], 2)
    img[a == a.max()] = 0
    a = img
    a = np.uint8(np.clip(a, 0, 255))
    return a

# Plot
# 生成处理后的图像
img = processFractal(ns.cpu().numpy())

# 画图
plt.figure(figsize=(16, 10))
plt.imshow(img)
plt.axis("off")
plt.tight_layout(pad=0)

# 保存文件
plt.savefig("test2.1.png", dpi=300, bbox_inches="tight")  # 保存到文件