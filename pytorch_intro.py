import torch

print(f"PyTorch版本：{torch.__version__}")
print(f"CUDA可用：{torch.cuda.is_available()}")

# 1. tensor 创建
a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6], [7, 8]])
print(f"\na:\n{a}")
print(f"b:\n{b}")

# 2. tensor 运算
print(f"\na + b:\n{a + b}")
print(f"a * b（逐元素乘）:\n{a * b}")
print(f"a @ b（矩阵乘）:\n{a @ b}")

# 3. 自动求导
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2 + 2 * x + 1
y.backward()
print(f"\nx = 3, y = x² + 2x + 1 = {y.item()}")
print(f"dy/dx = {x.grad.item()}")  # 应该是 2*3 + 2 = 8

# 4. 从 NumPy 转 tensor
import numpy as np
np_arr = np.array([[1.0, 2.0], [3.0, 4.0]])
tensor_from_np = torch.from_numpy(np_arr)
print(f"\n从NumPy转tensor:\n{tensor_from_np}")

# 5. tensor 转 NumPy
back_to_np = tensor_from_np.numpy()
print(f"\n转回NumPy:\n{back_to_np}")
