import torch
import numpy as np
import matplotlib.pyplot as plt

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create a grid in the complex plane
Y, X = np.mgrid[-1.3:1.3:0.005, -2:1:0.005]

eps = 1e-6

# Load into PyTorch tensors
x = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(Y, dtype=torch.float32)
z = torch.complex(x, y)  # constant c
zs = z.clone()           # z_0 = c
ns = torch.zeros_like(z) # divergence counter

# Transfer to GPU if available
z, zs, ns = z.to(device), zs.to(device), ns.to(device)
# Update divergence counter
ns = torch.zeros(z.shape, dtype=torch.int32, device=device)          # 迭代次数
converged_root = torch.full(z.shape, -1, dtype=torch.int32, device=device)  # 收敛到哪个根

# Roots of z^3 = 1
roots = torch.tensor([
    1+0j,
    -0.5 + 0.8660254j,   # e^(2πi/3)
    -0.5 - 0.8660254j    # e^(4πi/3)
], dtype=torch.complex64, device=device) #定义三个根

# Mandelbrot Set iterations
for i in range(200):

    zs = zs - (zs**3 - 1) / (3 * zs**2 + eps) # 避免3z=0
    # Have we diverged with this new value?

    for j, r in enumerate(roots):
        mask = (converged_root == -1) & (torch.abs(zs - r) < 1e-3)
        converged_root[mask] = j
        ns[mask] = i
        


    # Prepare for next iteration


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
plt.show()

# 保存文件
plt.savefig("test2.1.png", dpi=300, bbox_inches="tight")  # 保存到文件