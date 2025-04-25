from util.io import write_file, read_file
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import os

train_data = []

doc2docid = read_file('data/dataset/D0/corpus_docids.json')
tk0 = tqdm(doc2docid)
for doc, index, docid in tk0:
    length = len(word_tokenize(doc))
    head = word_tokenize(doc)[:length//3]
    middle = word_tokenize(doc)[length//3:2*length//3]
    tail = word_tokenize(doc)[2*length//3:]
    train_data.append(['task1', ' '.join(head), index])
    train_data.append(['task1', ' '.join(middle), index])
    train_data.append(['task1', ' '.join(tail), index])


pesudo2docid = read_file('data/dataset/D0/corpus_pseudo_query.json')
tk0 = tqdm(pesudo2docid)
for index, q in tk0:
    queries = q[1]
    for query in queries:
        train_data.append(['task2', query, index])


query2docid = read_file('nq-D0.jsonl')[:-1000]

for i in range(5):
    tk0 = tqdm(query2docid)
    for d in tk0:
        query = d['question_text']
        index = d['document_index']
        train_data.append(['task2', query, index])


test_data = []
query2docid = read_file('nq-D0.jsonl')[-1000:]
for d in query2docid:
    query = d['question_text']
    index = d['document_index']
    test_data.append(['task2', query, index])

write_file('data/dataset/D0/train.json', train_data)
write_file('data/dataset/D0/test.json', test_data)
corpus = read_file('nq-D0-corpus.jsonl')


for i in range(1, 6):
    mode = f'D{i}'
    test_data = []
    query2docid = read_file(f'nq-{mode}.jsonl')[-1000:]
    for d in query2docid:
        query = d['question_text']
        index = d['document_index']
        test_data.append(['task2', query, index])
    write_file(f'data/dataset/{mode}/test.json', test_data)




