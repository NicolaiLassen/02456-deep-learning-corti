import ctcdecode

blank = "-"

labels = [
    " ",
    *[chr(i + 96) for i in range(1, 27)],
    blank
]


class CTCBeamDecoder:

    def __init__(self, beam_size=100, blank_id=labels.index(blank),
                 kenlm_path="./lm/lm_librispeech_kenlm_word_4g_200kvocab.bin"):
        self.decoder = ctcdecode.CTCBeamDecoder(
            labels, alpha=0.522729216841, beta=0.96506699808,
            beam_width=beam_size, blank_id=blank_id,
            model_path=kenlm_path)

    def __call__(self, output):
        beam_result, beam_scores, timesteps, out_seq_len = self.decoder.decode(output)
        return self.convert_to_string(beam_result[0][0], labels, out_seq_len[0][0])

    def convert_to_string(self, tokens, vocab, seq_len):
        return ''.join([vocab[x] for x in tokens[0:seq_len]])
