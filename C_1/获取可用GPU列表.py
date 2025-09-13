import torch

# 检查是否有可用的GPU
if torch.cuda.is_available():
    # 获取GPU数量
    gpu_count = torch.cuda.device_count()
    print(f"Number of available GPUs: {gpu_count}")

    # 获取当前使用的GPU索引
    current_device = torch.cuda.current_device()
    print(f"Currently using GPU {current_device}: {torch.cuda.get_device_name(current_device)}")
else:
    print("CUDA is not available. Using CPU.")

