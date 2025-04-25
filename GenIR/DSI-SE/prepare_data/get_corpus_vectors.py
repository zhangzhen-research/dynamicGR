from transformers import BertTokenizer, BertModel
import torch

import json
from tqdm import tqdm

def read_json(file):
    with open(file, 'r') as f:
        return json.load(f)
def write_json(file, data):
    with open(file, 'w') as f:
        json.dump(data, f)



model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)


device = 'cuda'
model.to(device)


url2doc_list = read_json('corpus_lite.json')
url2doc_list = [[url2doc_list[index], index] for index in range(len(url2doc_list))]

batch_size = 64
corpus = [doc for doc, index in url2doc_list]
with tqdm(total=len(corpus)) as pbar:
    for i in range(0, len(corpus), batch_size):
        batch_docs = corpus[i:i + batch_size]
        inputs = tokenizer(batch_docs, return_tensors='pt', truncation=True, padding='max_length', max_length=500,
                           return_attention_mask=True)

        
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        last_hidden_states = outputs.last_hidden_state
        batch_embeddings = last_hidden_states[:, 0, :].squeeze().cpu().tolist()  
        for j, doc in enumerate(batch_docs):
            url2doc_list[i + j] = url2doc_list[i + j] + [batch_embeddings[j]]

        pbar.update(batch_size)

sorted(url2doc_list, key=lambda x: x[1])

write_json('data/dataset/corpus_vectors.json', url2doc_list)