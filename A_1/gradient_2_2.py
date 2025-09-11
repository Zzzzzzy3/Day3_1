import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
inputs = torch.tensor([[2, 1000], [3, 2000], [2, 500], [1, 800], [4, 3000]], dtype=torch.float, device=device)
labels = torch.tensor([[19], [31], [14], [15], [43]], dtype=torch.float, device=device)

#进行归一化
inputs = inputs / torch.tensor([4, 3000], device=device)


w = torch.ones(2, 1, requires_grad=True, device=device)
b = torch.ones(1, requires_grad=True, device=device)

epoch = 1000
lr = 0.5

for i in range(epoch):
    outputs = inputs @ w + b
    loss = torch.mean(torch.square(outputs - labels))
    print("loss", loss.item())
    loss.backward()
    print("w.grad", w.grad.tolist())
    with torch.no_grad():
        w -= w.grad * lr
        b -= b.grad * lr

    w.grad.zero_()
    b.grad.zero_()