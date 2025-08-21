import torch
import numpy as np
import matplotlib.pyplot as plt

print("PyTorch Version:", torch.__version__)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# grid for computing image
X, Y = np.mgrid[-4.0:4:0.01, -4.0:4:0.01]

# load into PyTorch tensors
x = torch.Tensor(X).to(device)
y = torch.Tensor(Y).to(device)

# Compute sine pattern (change here)
z = torch.sin(x) * torch.sin(y)

# plot
plt.imshow(z.cpu().numpy(), cmap='viridis')  # 可以换 cmap 看效果
plt.tight_layout()
plt.show()
plt.savefig("test1.1.png", dpi=300, bbox_inches="tight")
