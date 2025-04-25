import numpy as np

def eval_all(output_all, label_all):
    def hit_1(predictions, label):
        return int(predictions[0] == label)

    def mrr(predictions, label):
        for i, p in enumerate(predictions):
            if p == label:
                return 1 / (i + 1)
        return 0

    def hit_10(predictions, label):
        return int(label in predictions[:10])

    def hit_100(predictions, label):
        return int(label in predictions[:100])

    hit_1s = [hit_1(predictions, label) for predictions, label in zip(output_all, label_all)]
    mrrs = [mrr(predictions, label) for predictions, label in zip(output_all, label_all)]
    hit_100s = [hit_100(predictions, label) for predictions, label in zip(output_all, label_all)]
    hit_10s = [hit_10(predictions, label) for predictions, label in zip(output_all, label_all)]
    return {
        'hit@1': np.mean(hit_1s),
        'mrr': np.mean(mrrs),
        'hit@100': np.mean(hit_100s),
        'hit@10': np.mean(hit_10s),
    }
