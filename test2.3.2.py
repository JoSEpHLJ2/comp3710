import torch
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# 设备配置
# -------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# -------------------------------
# 高分辨率 Mandelbrot 集（可放大区域）
# -------------------------------
# 减小步长提高分辨率，同时放大某个区域
Y, X = np.mgrid[-0.75:0.75:0.002, -1.25:-0.25:0.002]

# 转换为 PyTorch 张量
x = torch.tensor(X, dtype=torch.float32).to(device)
y = torch.tensor(Y, dtype=torch.float32).to(device)
z = torch.complex(x, y)  # 常数 c
zs = z.clone()
ns = torch.zeros_like(z, dtype=torch.int32)

max_iter = 300  # 增加迭代次数，提高细节
for i in range(max_iter):
    zs_ = zs * zs + z
    not_diverged = torch.abs(zs_) < 4.0
    ns += not_diverged
    zs = zs_

# -------------------------------
# Julia 集生成
# -------------------------------
c = torch.tensor(complex(-0.7, 0.27015), dtype=torch.complex64, device=device)
zs_julia = z.clone()  # Julia 集从 z_0 = 网格开始
ns_julia = torch.zeros_like(zs_julia, dtype=torch.int32)

for i in range(max_iter):
    zs_ = zs_julia * zs_julia + c
    not_diverged = torch.abs(zs_) < 4.0
    ns_julia += not_diverged
    zs_julia = zs_

# -------------------------------
# 颜色映射函数
# -------------------------------
def processFractal(a):
    a_cyclic = (6.28 * a / 20.0).reshape(list(a.shape) + [1])
    img = np.concatenate([
        10 + 20 * np.cos(a_cyclic),
        30 + 50 * np.sin(a_cyclic),
        155 - 80 * np.cos(a_cyclic)
    ], 2)
    img[a == a.max()] = 0
    a = np.uint8(np.clip(img, 0, 255))
    return a

# -------------------------------
# 绘图
# -------------------------------
plt.figure(figsize=(16, 8))

# Julia 集
plt.subplot(1, 2, 2)
plt.imshow(processFractal(ns_julia.cpu().numpy()))
plt.axis("off")
plt.title("Julia Set (c = -0.7 + 0.27015i)")

plt.tight_layout()
plt.show()

# 保存图像
plt.savefig("test2.3.2.png", dpi=300, bbox_inches="tight")
