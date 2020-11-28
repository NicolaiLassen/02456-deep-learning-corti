import torch


def collate(batch):
    data = [torch.Tensor(t[0]).transpose(0, 1) for t in batch]
    data = torch.nn.utils.rnn.pad_sequence(data, batch_first=True)
    return [data]
