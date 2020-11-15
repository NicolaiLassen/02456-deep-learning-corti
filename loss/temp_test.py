import torch
import torch.nn as nn

if __name__ == '__main__':
    def my_loss(output, target):
        loss = torch.mean((output - target) ** 2)
        return loss


    model = nn.Linear(2, 2)
    x = torch.randn(1, 2)
    target = torch.randn(1, 2)
    output = model(x)
    print(output)
    loss = my_loss(output, target)
    print(loss)
    loss.backward()
    print(model.weight.grad)