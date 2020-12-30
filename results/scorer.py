import jiwer
import numpy as np


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
    ground_truth = "chapter one missus rachel lynde is surprised missus rachel lynde lived just where the avonlea main road dipped down into a little hollow fringed with alders and ladies eardrops and traversed by a brook"
    hypothesis = "chapter one missus rachel lynde russell e s here the avonlea a road led to all old red oh alders and ladies arosa rared a broo"
    print(get_asr_metric([ground_truth, "test test test", "test"], [hypothesis, "test test test", "test"]))
