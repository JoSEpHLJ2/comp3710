# 用 PyTorch 创建二维正弦函数的“条纹”图像
import torch
import matplotlib.pyplot as plt
import numpy as np

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 创建坐标网格
x_np, y_np = np.mgrid[-4.0:4.0:0.01, -4.0:4.0:0.01]
x = torch.tensor(x_np, dtype=torch.float32).to(device)
y = torch.tensor(y_np, dtype=torch.float32).to(device)

# 生成正弦函数图案，例如 sin(5x + 5y)
z = torch.sin(5 * x + 5 * y)

# 绘制正弦条纹
plt.imshow(z.cpu().numpy(), extent=[-4, 4, -4, 4], cmap='seismic')
plt.title("2D Sine Function (Stripes)")
plt.tight_layout()
plt.show()
plt.savefig("test1.2.png", dpi=300, bbox_inches="tight")