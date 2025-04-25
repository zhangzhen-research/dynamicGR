import json


def read_json(file):
    with open(file) as f:
        return json.load(f)

def read_jsonl(file):
    data = []
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))

def write_jsonl(path, data):
    with open (path, 'w', encoding='utf-8') as f:
        for d in data:
            f.write(json.dumps(d) + '\n')


output = []
import copy
for i in range(5):
    path = F'LTRGR/data/results/MINDER_NQ_train_top100_{i}.json'
    data = read_json(path)
    for d in data:
        d['positive_ctxs'] = copy.deepcopy(d['ctxs'])
        d['negative_ctxs'] = []
        del d['ctxs']
        output.append(d)



write_jsonl('LTRGR/data/results/MINDER_NQ_train_top100.json', output)