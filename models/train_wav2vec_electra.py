import torch
import torch.nn as nn
from transformers import ElectraModel, ElectraTokenizer

if __name__ == '__main__':
    text_1 = "SHE HAD THIN AWKWARD FIGURE".lower()
    text_2 = "HAD SHE THIN AWKWARD FIGURE".lower()
    text_3 = "HE HAD THIN AWKWARD FIGURE".lower()

    tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')
    model = ElectraModel.from_pretrained('google/electra-small-discriminator', return_dict=True)

    inputs_1 = torch.tensor(tokenizer.encode(text_1, return_tensors="pt"))
    print(inputs_1.shape)
    inputs_2 = torch.tensor(tokenizer.encode(text_2, return_tensors="pt"))
    inputs_3 = torch.tensor(tokenizer.encode(text_3, return_tensors="pt"))

    outputs_1 = model(inputs_1)[0]
    outputs_2 = model(inputs_2)[0]
    outputs_3 = model(inputs_3)[0]

    print(outputs_1.shape)
    print(outputs_2.shape)
    print(outputs_3.shape)

    cos = nn.CosineSimilarity()
    output_1 = (1 - cos(outputs_1, outputs_2)).sum()
    print(output_1)
    output_2 = (1 - cos(outputs_1, outputs_3)).sum()
    print(output_2)
