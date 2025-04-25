


import json
import os

def read_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

answers = read_json('data/NQ/dataset/biencoder-nq-dev.json')
data = read_json(f'data/NQ/LTRGR/output.json')

r1 = 0
r10 = 0
r100 = 0

for i, d in enumerate(data):
    answer = answers[i]['positive_ctxs'][0]['passage_id']
    predicts = [p['passage_id'] for p in d['ctxs']]
    r1 += answer in predicts[:1]
    r10 += answer in predicts[:10]
    r100 += answer in predicts[:100]

r1 /= len(data)
r10 /= len(data)
r100 /= len(data)

print(f"Recall@1: {r1}")
print(f"Recall@10: {r10}")
print(f"Recall@100: {r100}")


for i in range(1, 6):

    data = read_json(f'data/NQ/output/test_{i}.json')
    answers = read_json(f'data/NQ/dataset/biencoder-nq-dev.json')
    data = read_json(f'data/NQ/LTRGR/new-output-{i}.json')
    answers = read_json(f'data/NQ/dataset/biencoder-nq-dev_{i}.json')


    r1 = 0
    r10 = 0
    r100 = 0

    for i, d in enumerate(data):
        answer = answers[i]['positive_ctxs'][0]['passage_id']
        predicts = [p['passage_id'] for p in d['ctxs']]
        r1 += answer in predicts[:1]
        r10 += answer in predicts[:10]
        r100 += answer in predicts[:100]


    r1 /= len(data)
    r10 /= len(data)
    r100 /= len(data)

    print(f"Recall@1: {r1}")
    print(f"Recall@10: {r10}")
    print(f"Recall@100: {r100}")