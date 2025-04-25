import json

def write_tsv(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        for d in data:
            f.write('\t'.join(d) + '\n')

def write_json(path, data):
    with open (path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

def read_jsonl(path):
    with open (path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def read_tsv(path):
    with open (path, 'r', encoding='utf-8') as f:
        return [line.strip().split('\t') for line in f]



input_path = 'msmarco-D0.jsonl'
data = read_jsonl(input_path)
curpus_path = 'msmarco-D0-corpus.jsonl'
corpus = read_jsonl(curpus_path)
train_data = []
for d in data:
    cur = {}
    cur['dataset'] = 'msmarco_dataset'
    cur['question'] = d['question_text']
    cur['positive_ctxs'] = [{'title': str(corpus[d['document_index']][3]), 'text': corpus[d['document_index']][0], 'passage_id': str(corpus[d['document_index']][1])}]
    cur['negative_ctxs'] = []
    train_data.append(cur)

input_path = 'msmarco-D0-test.jsonl'
data = read_jsonl(input_path)
test_data = []
for d in data:
    cur = {}
    cur['dataset'] = 'msmarco_dataset'
    cur['question'] = d['question_text']
    cur['positive_ctxs'] = [{'title': str(corpus[d['document_index']][3]), 'text': corpus[d['document_index']][0], 'passage_id': str(corpus[d['document_index']][1])}]
    cur['negative_ctxs'] = []
    test_data.append(cur)
new_corpus = []
for c in corpus:
    new_corpus.append([str(c[1]), str(c[3]), str(c[0])])
write_json('dataset/msmarco/stage1/train_data.json', train_data)
write_json('dataset/msmarco/stage1/test_data.json', test_data)
write_tsv('dataset/msmarco/stage1/corpus.tsv', new_corpus)


corpus_len = len(corpus)
for i in range(1, 6):
    input_path = f'msmarco-D{i}-test.jsonl'
    corpus_path = f'msmarco-D{i}-corpus.jsonl'
    corpus += read_jsonl(corpus_path)
    data = read_jsonl(input_path)
    test_data = []
    cnt = 0
    for d in data:
        cur = {}

        cur['dataset'] = 'msmarco_dataset'
        cur['question'] = d['question_text']
        cur['positive_ctxs'] = [{'title': str(corpus[d['document_index']][3]), 'text': corpus[d['document_index']][0],
                                 'passage_id': str(corpus[d['document_index']][1])}]
        passage_id = str(corpus[d['document_index']][1])
        if int(passage_id) >= corpus_len:
            cnt += 1
        else:
            continue
        cur['negative_ctxs'] = []
        test_data.append(cur)
    corpus_len = len(corpus)
    new_corpus = []
    for c in corpus:
        new_corpus.append([str(c[1]), str(c[3]), str(c[0])])

    write_tsv(f'dataset/msmarco/stage1/corpus_{i}.tsv', new_corpus)
    write_json(f'dataset/msmarco/stage1/test_data_{i}.json', test_data)





