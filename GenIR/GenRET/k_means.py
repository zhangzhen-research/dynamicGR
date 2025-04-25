import time
import torch
import json
import os
from utils.io import write_pkl
from collections import defaultdict

DATA_READY_SIGNAL = 'data_ready.txt'
TASK_COMPLETE_SIGNAL = 'task_complete.txt'
DATA_FILE = 'data.pkl'
PARAM_FILE = 'params.json'


def constrained_km(data, n_clusters=512):
    from k_means_constrained import KMeansConstrained
    print(f'Running constrained k-means with {n_clusters} clusters')
    size_min = min(len(data) // (n_clusters * 2), n_clusters // 4)
    clf = KMeansConstrained(n_clusters=n_clusters, size_min=size_min, size_max=n_clusters * 2, max_iter=10, n_init=10,
                            n_jobs=30, verbose=True)
    clf.fit(data)
    return clf.cluster_centers_, clf.labels_.tolist()


def wait_for_signal():
    while not os.path.exists(DATA_READY_SIGNAL):
        time.sleep(5)
    os.remove(DATA_READY_SIGNAL)

def main():
    while True:
        wait_for_signal()

        
        data = torch.load(DATA_FILE)  
        with open(PARAM_FILE, 'r') as f:
            params = json.load(f)

        centroids, code = constrained_km(data, params['n_clusters'])

        
        save_path = params['save_path']
        epoch = params['epoch']
        write_pkl(centroids, f'{save_path}/{epoch}.pt.kmeans.{params["n_clusters"]}')
        with open(f'{save_path}/{epoch}.pt.kmeans_code.{params["n_clusters"]}', 'w') as f:
            json.dump(code, f)

        
        with open(TASK_COMPLETE_SIGNAL, 'w') as f:
            f.write('done')

        print('Task complete')


if __name__ == "__main__":
    main()