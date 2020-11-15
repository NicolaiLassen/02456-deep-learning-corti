import torch
import torch.nn as nn

if __name__ == '__main__':
    a = torch.rand(1, 2)
    print(a)
    sum = torch.einsum('ij,ij->ij', a, a)
    print(sum)