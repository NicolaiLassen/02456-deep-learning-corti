import torch
import torch.nn.functional as F
from transformers import ElectraModel, ElectraTokenizer

if __name__ == '__main__':
    text_1 = "SHE HAD THIN AWKWARD FIGURE".lower()
    text_2 = "HAD SHE THIN AWKWARD FIGURE".lower()
    text_3 = "HE HAD THIN AWKWARD FIGURE".lower()

    tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')
    model = ElectraModel.from_pretrained('google/electra-small-discriminator', return_dict=True)

    inputs_1 = torch.tensor(tokenizer.encode(text_2, return_tensors="pt"))
    print("tok", inputs_1)

    inputs_2 = torch.tensor(tokenizer.encode(text_2, return_tensors="pt"))
    inputs_3 = torch.tensor(tokenizer.encode(text_3, return_tensors="pt"))

    inputs_in = torch.stack([inputs_2.squeeze(0), inputs_3.squeeze(0)]).squeeze(0)

    print(inputs_in)

    outputs_1 = model(inputs_in)[0]
    outputs_2 = model(inputs_3)[0]

    print(outputs_1.shape)
    print(outputs_2.shape)
