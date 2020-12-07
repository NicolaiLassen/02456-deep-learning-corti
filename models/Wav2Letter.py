import math

import torch
import torchaudio.models


class Wav2LetterWithFit(torchaudio.models.Wav2Letter):

    def fit(self, inputs, output, optimizer, ctc_loss, batch_size, epoch, print_every=50):
        """Trains Wav2Letter model.
        Args:
            inputs (torch.Tensor): shape (sample_size, num_features, frame_len)
            output (torch.Tensor): shape (sample_size, seq_len)
            optimizer (nn.optim): pytorch optimizer
            ctc_loss (ctc_loss_fn): ctc loss function
            batch_size (int): size of mini batches
            epoch (int): number of epochs
            print_every (int): every number of steps to print loss
        """

        total_steps = math.ceil(len(inputs) / batch_size)
        seq_length = output.shape[1]

        for t in range(epoch):

            samples_processed = 0
            avg_epoch_loss = 0

            for step in range(total_steps):
                optimizer.zero_grad()
                batch = \
                    inputs[samples_processed:batch_size + samples_processed]

                # log_probs shape (batch_size, num_classes, output_len)
                log_probs = self.forward(batch)

                # CTC_Loss expects input shape
                # (input_length, batch_size, num_classes)
                log_probs = log_probs.transpose(1, 2).transpose(0, 1)

                # CTC arguments
                # https://pytorch.org/docs/master/nn.html#torch.nn.CTCLoss
                # better definitions for ctc arguments
                # https://discuss.pytorch.org/t/ctcloss-with-warp-ctc-help/8788/3
                mini_batch_size = len(batch)
                targets = output[samples_processed: mini_batch_size + samples_processed]

                input_lengths = torch.full((mini_batch_size,), log_probs.shape[0], dtype=torch.long)
                target_lengths = torch.IntTensor([target.shape[0] for target in targets])

                loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)

                avg_epoch_loss += loss.item()

                loss.backward()
                optimizer.step()

                samples_processed += mini_batch_size

                if step % print_every == 0:
                    print("epoch", t + 1, ":", "step", step + 1, "/", total_steps, ", loss ", loss.item())

            print("epoch", t + 1, "average epoch loss", avg_epoch_loss / total_steps)
