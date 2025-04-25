from transformers import BertTokenizer, BertModel
import torch
from util.io import read_file, write_file
from tqdm import tqdm
import os
import joblib
import random
import numpy as np


def load_all_kmeans_models(base_path='models'):
    models = {}
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.endswith('.joblib'):
                depth, node_id = map(int, file.replace('.joblib', '').split('_')[2::2])
                model_path = os.path.join(root, file)
                model = joblib.load(model_path)
                if depth not in models:
                    models[depth] = {}
                models[depth][node_id] = model
    return models


def assign_document_to_cluster(embedding, models, cluster_size_threshold, depth=0, node_id=0, prefix=[]):
    if depth in models and node_id in models[depth]:
        model = models[depth][node_id]
        cluster_id = model.predict([embedding])[0]
        new_prefix = prefix + [cluster_id]

        return assign_document_to_cluster(embedding, models, cluster_size_threshold, depth + 1, cluster_id, new_prefix)
    else:
        return prefix + random.choices(range(100))


model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)


device = 'cuda'
model.to(device)

for i in range(1, 6):
    mode = f'D{i}'


    os.mkdir(f'data/dataset/{mode}')

    
    url2doc_list = read_file(f'nq-{mode}-corpus.jsonl')

    
    batch_size = 64
    corpus = [doc for doc, index in url2doc_list]
    embeddings_list = []

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
            embeddings_list.extend(batch_embeddings)
            pbar.update(batch_size)



    
    all_models = load_all_kmeans_models(base_path='data/dataset/D0/models/hkmeans')
    id_num = 50

    for i, (doc, index) in enumerate(url2doc_list):
        embedding = embeddings_list[i]
        encoded_path = assign_document_to_cluster(embedding, all_models, cluster_size_threshold=id_num)
        k_means_id = ["$" + str(x) + "$" for x in encoded_path]
        url2doc_list[i] = [doc, index, k_means_id]

    
    save_path = f'data/dataset/{mode}/corpus_docids.json'
    write_file(save_path, url2doc_list)

    
    temp_save_path = f'data/dataset/{mode}/corpus_vectors.json'
    write_file(temp_save_path, embeddings_list)