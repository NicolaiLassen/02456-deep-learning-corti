import torch

if __name__ == '__main__':
    x = torch.randn(2, 1, 3)
    print(x)
    x = torch.transpose(x, 0, 1)
    print(x)