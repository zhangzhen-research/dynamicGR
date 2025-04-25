import numpy as np
from sklearn.cluster import KMeans
import json
import time
import os
import sys
sys.setrecursionlimit(10000)

def read_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def write_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f)



import json
import joblib


def save_hierarchical_kmeans(embeddings, original_indices, cluster_size_threshold, prefix=[], depth=0, base_path='models', node_id=0):
    tree_structure = []

    start_time = time.time()  
    num_docs = len(embeddings)
    if num_docs <= cluster_size_threshold:

        end_time = time.time()  

        def process_num(n):  
            res = []
            while n:
                res.append(n % cluster_size_threshold)
                n = n // cluster_size_threshold
            res.reverse()
            return res

        print(f"{'  ' * depth}Leaf cluster reached: Time taken: {end_time - start_time:.2f}s")
        return [[prefix, original_indices[i]] for i in range(num_docs)], []

    else:

        model = KMeans(n_clusters=cluster_size_threshold, random_state=0).fit(embeddings)
        
        labels = model.labels_
        model_path = os.path.join(base_path, f'kmeans_depth_{depth}_node_{node_id}.joblib')
        

        ids = []
        for cluster_id in range(cluster_size_threshold):
            cluster_indices = np.where(labels == cluster_id)[0]
            cluster_embeddings = embeddings[cluster_indices]
            cluster_prefix = prefix + [cluster_id]
            cluster_original_indices = [original_indices[i] for i in cluster_indices]
            print(f"{'  ' * depth}Clustering at depth {depth}: Cluster {cluster_id} with {len(cluster_indices)} items")
            cluster_ids, sub_tree = save_hierarchical_kmeans(cluster_embeddings, cluster_original_indices, cluster_size_threshold,
                                                cluster_prefix, depth + 1, base_path, cluster_id)
            ids.extend(cluster_ids)
            tree_structure.extend(sub_tree)
        end_time = time.time()  
        print(f"{'  ' * depth}Depth {depth} clustering completed: Time taken: {end_time - start_time:.2f}s")
        return ids, tree_structure


id_num = 32
vectors = read_json('dataset/corpus_vectors.json')
new_url2doc_list = []
the_vectors = [vector[2] for vector in vectors]
embeddings = np.array(the_vectors)
base_model_save_path = 'dataset/hkmeans'
os.makedirs(base_model_save_path, exist_ok=True)
ids, tree_structure = save_hierarchical_kmeans(embeddings, list(range(len(embeddings))), cluster_size_threshold=id_num, base_path=base_model_save_path)

document_ids_sorted = sorted(ids, key=lambda x: x[-1])
for i, vector in enumerate(vectors):
    k_means_id = document_ids_sorted[i][0]
    
    vector[2] = k_means_id

save_path = 'dataset/corpus_docids_fortest.json'
vectors = [[vector[1], vector[2]] for vector in vectors]

write_json(save_path, vectors)

