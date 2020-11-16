import torch
import torch.nn as nn

def buffered_arange(max):
    if not hasattr(buffered_arange, "buf"):
        buffered_arange.buf = torch.LongTensor()
    if max > buffered_arange.buf.numel():
        buffered_arange.buf.resize_(max)
        torch.arange(max, out=buffered_arange.buf)
    return buffered_arange.buf[:max]

if __name__ == '__main__':
    t = torch.rand(1, requires_grad=True)
    t = t.detach().clone()
    print(t)