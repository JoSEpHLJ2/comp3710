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

# Gaussian
gaussian = torch.exp(-(x**2 + y**2) / 2.0)

# 1) Gaussian × sin(x)
z = gaussian * torch.sin(x)

# 2) Gaussian × cos(y)
# z = gaussian * torch.cos(y)


# plot
plt.imshow(z.cpu().numpy(), cmap='viridis')
plt.colorbar()
plt.tight_layout()
plt.show()
plt.savefig("test1.1.png", dpi=300, bbox_inches="tight")
