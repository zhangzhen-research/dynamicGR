import json
import pandas as pd

def write_file(path, data):
    if path.endswith('.json'):
        write_json(path, data)
    elif path.endswith('.csv'):
        write_csv(path, data)
    elif path.endswith('.jsonl'):
        write_jsonl(path, data)
    elif path.endswith('.txt'):
        write_txt(path, data)

def read_file(path):
    if path.endswith('.json'):
        return read_json(path)
    elif path.endswith('.csv'):
        return read_csv(path)
    elif path.endswith('.jsonl'):
        return read_jsonl(path)
    elif path.endswith('.txt'):
        return read_txt(path)

def read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def write_json(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

def read_csv(path):
    res = pd.read_csv(path)
    res_list = []
    for index, row in res.iterrows():
        res_list.append(list(row))
    return res_list

def write_csv(path, data):
    data.to_csv(path, index=False)


def read_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def write_jsonl(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        for d in data:
            f.write(json.dumps(d) + '\n')

def read_txt(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.readlines()

def write_txt(path, data):
    if isinstance(data, list):
        data = [str(x) + '\n' for x in data]
    with open(path, 'w', encoding='utf-8') as f:
        f.writelines(data)



