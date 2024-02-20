import torch
import time

# 检查是否有可用的GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    raise SystemError("GPU not found. This program needs a GPU to run.")

# 创建两个随机矩阵
matrix_size = 100  # 可以根据需要调整矩阵的大小
A = torch.randn(matrix_size, matrix_size, device=device)
B = torch.randn(matrix_size, matrix_size, device=device)

while True:
    # 执行矩阵乘法
    C = torch.matmul(A, B)
    
    # 稍作等待，以避免完全占据GPU
    time.sleep(0.1)  # 调整等待时间可以控制占用的GPU资源量
