import json
import pickle

def read_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def write_pkl(file_path, data):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
outdict = {}
data = read_json('data_process/corpus_pseudo_query.json')
for d in data:
    index, doc = d
    cur = []
    for query in doc[1]:
        cur.append(query)
    outdict[str(index)] = cur

write_pkl('data/pseudo_queries/pid2query_nq.pkl', outdict)





