# Linux gcc clang
import ctcdecode

blank = "-"

labels = [
    " ",
    *[chr(i + 96) for i in range(1, 27)],
    blank
]


class CTCBeamDecoder:
    def __init__(self, kenlm_path, beam_size=2000):
        self.decoder = ctcdecode.CTCBeamDecoder(
            labels,
            log_probs_input=True,
            alpha=0.922729216841,
            beta=0.66506699808,
            beam_width=beam_size,
            blank_id=labels.index(blank),
            model_path=kenlm_path)

    def __call__(self, output):
        beam_result, beam_scores, timesteps, out_seq_len = self.decoder.decode(output)
        return self.convert_to_string(beam_result[0][0], labels, out_seq_len[0][0])

    def convert_to_string(self, tokens, vocab, seq_len):
        return ''.join([vocab[x] for x in tokens[0:seq_len]])


if __name__ == "__main__":
    "WER"
