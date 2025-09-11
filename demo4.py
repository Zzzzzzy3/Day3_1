import torch

shape = (2,3) #形状（2行3列）
rand_tensor = torch.rand(shape) # 生成一个从[0,1]均匀抽样的tensor。
randn_tensor = torch.randn(shape) # 生成一个从标准正态分布抽样的tensor。
ones_tensor = torch.ones(shape) #生成一个值全为1的tensor。
zeros_tensor = torch.zeros(shape) # 生成一个值全为0的tensor。
twos_tensor = torch.full(shape, 2) #  生成一个值全为2的tensor。
print(rand_tensor)
print(randn_tensor)
print(ones_tensor)
print(zeros_tensor)
print(twos_tensor)

tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")