import csv
import json


def write_csv(file_path, data):
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(data)

def read_json(file_path):
    with open(file_path) as f:
        return json.load(f)


data = read_json('data/NQ/dataset/biencoder-nq-dev.json')
output = []
for d in data:
    output.append([d['question'], ['', '']])

write_csv('data/NQ/test/nq-test.qa.csv', output)


for i in range(1, 6):
    data = read_json(f'data/NQ/dataset/biencoder-nq-dev_{i}.json')
    output = []
    for d in data:
        output.append([d['question'], ['', '']])
    write_csv(f'data/NQ/test/nq-test-{i}.qa.csv', output)


