import torch

x = torch.tensor([[1,2,3],[4,5,6]])
#扩展第0维
x_0 = x.unsqueeze(0)
print(x_0.shape,x_0)
#扩展第1维
x_1 = x.unsqueeze(1)
print(x_1.shape,x_1)
#扩展第2维
x_2 = x.unsqueeze(2)
print(x_2.shape,x_2)


# 创建一个随机张量作为例子
x = torch.randn(2, 3, 4)  # 创建一个形状为 (2, 3, 4) 的3D张量

# 获取张量的形状
size_obj = x.size()   # 使用 .size() 方法
shape_obj = x.shape   # 使用 .shape 属性

print(size_obj)   # 输出: torch.Size([2, 3, 4])
print(shape_obj)  # 输出: torch.Size([2, 3, 4])