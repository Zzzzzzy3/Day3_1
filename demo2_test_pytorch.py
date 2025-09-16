import torch
print(torch.cuda.is_available())
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
