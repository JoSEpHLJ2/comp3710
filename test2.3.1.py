import torch
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# 设备配置
# -------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("使用设备:", device)

# -------------------------------
# 创建复平面网格
# -------------------------------
# 增加分辨率可以稍微加快或降低计算时间，可调整0.005
Y, X = np.mgrid[-1.3:1.3:0.005, -2:1:0.005]

# 转换为 PyTorch 张量
x = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(Y, dtype=torch.float32)
z = torch.complex(x, y)  # 常数 c
zs = z.clone()           # 初始 z_0 = c
ns = torch.zeros_like(z, dtype=torch.int32) # 发散计数器

# 转到 GPU
z, zs, ns = z.to(device), zs.to(device), ns.to(device)

# -------------------------------
# Mandelbrot 集迭代
# -------------------------------
max_iter = 200
for i in range(max_iter):
    zs_ = zs * zs + z
    not_diverged = torch.abs(zs_) < 4.0
    ns += not_diverged
    zs = zs_

# -------------------------------
# 颜色映射函数
# -------------------------------
def processFractal(a):
    # 将迭代次数映射到 RGB
    a_cyclic = (6.28 * a / 20.0).reshape(list(a.shape) + [1])
    img = np.concatenate([
        10 + 20 * np.cos(a_cyclic),
        30 + 50 * np.sin(a_cyclic),
        155 - 80 * np.cos(a_cyclic)
    ], 2)
    img[a == a.max()] = 0  # Mandelbrot 集内部设为黑色
    a = np.uint8(np.clip(img, 0, 255))
    return a

# -------------------------------
# 生成图像并显示
# -------------------------------
img = processFractal(ns.cpu().numpy())

plt.figure(figsize=(16, 10))
plt.imshow(img)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()

# 保存图像
plt.savefig("mandelbrot_gpu.png", dpi=300, bbox_inches="tight")