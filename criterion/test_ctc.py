import torch

from models.Wav2LetterEmbed import Wav2LetterEmbed

if __name__ == '__main__':
    # Target are to be un-padded
    T = 50  # Input sequence length
    C = 20  # Number of classes (including blank)
    N = 16  # Batch size

    # Initialize random batch of input vectors, for *size = (T,N,C)
    input = torch.randn(T, N, C).log_softmax(2).detach().requires_grad_()
    input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long)

    target_lengths = torch.randint(low=1, high=T, size=(N,), dtype=torch.long)
    target = torch.randint(low=1, high=C, size=(sum(target_lengths),), dtype=torch.long)

    char2index = {
        " ": 0,
        "_": 1,
        **{chr(i + 96): i + 1 for i in range(1, 27)}
    }
    index2char = {
        0: " ",
        1: "_",
        **{i + 1: chr(i + 96) for i in range(1, 27)}
    }

    wav2letter = Wav2LetterEmbed(num_classes=len(char2index), num_features=256)

    wav2letter.train()

    optimizer = torch.optim.Adam(wav2letter.parameters(), lr=1e-4)

    for i in range(10):
        # Initialize random batch of targets (0 = blank, 1:C = classes)

        a = wav2letter(input)
        ctc_loss = torch.nn.CTCLoss()
        loss = ctc_loss(a, target, input_lengths, target_lengths)
        loss.backward()
        optimizer.step()

