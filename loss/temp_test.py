import torch
import torch.nn as nn



if __name__ == '__main__':
    t = torch.rand(1, requires_grad=True)
    t = t.detach().clone()
    print(t)