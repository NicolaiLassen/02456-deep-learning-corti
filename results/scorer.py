import jiwer
import numpy as np
import torchaudio


def get_asr_metric(ground_truths, hypotheses):
    batch_test = {
        "wer": [],
        "mer": [],
        "wil": []
    }

    for ground_truth, hypothesis in zip(ground_truths, hypotheses):
        measures = jiwer.compute_measures(ground_truth, hypothesis)
        batch_test["wer"].append(measures['wer'])
        batch_test["mer"].append(measures['mer'])
        batch_test["wil"].append(measures['wil'])

    batch_test["wer"] = np.array(batch_test["wer"]).mean()
    batch_test["mer"] = np.array(batch_test["mer"]).mean()
    batch_test["wil"] = np.array(batch_test["wil"]).mean()

    return batch_test


if __name__ == '__main__':
    print(get_asr_metric("test", "test"))