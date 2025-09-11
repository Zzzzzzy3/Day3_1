import torch
import time

# 确保 GPU 可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 生成随机矩阵
size = 10000  # 矩阵大小
A_cpu = torch.rand(size, size) # 默认在CPU上创建tensor
B_cpu = torch.rand(size, size)

start_cpu = time.time()
C_cpu = torch.mm(A_cpu, B_cpu)  # 矩阵乘法
end_cpu = time.time()
cpu_time = end_cpu - start_cpu

# 在 GPU 上计算
A_gpu = A_cpu.to(device) # 将tensor转移到GPU上
B_gpu = B_cpu.to(device)

start_gpu = time.time()
C_gpu = torch.mm(A_gpu, B_gpu)
torch.cuda.synchronize()  # 确保GPU计算完成
end_gpu = time.time()
gpu_time = end_gpu - start_gpu

print(f"CPU time: {cpu_time:.6f} sec")
if torch.cuda.is_available():
    print(f"GPU time: {gpu_time:.6f} sec")
else:
    print("GPU not available, skipping GPU test.")