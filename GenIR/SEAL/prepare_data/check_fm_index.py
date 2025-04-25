from seal.index import FMIndex
import logging
import psutil
import os

def _get_process_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss

def load_fm_index(fm_index_path: str):

    index = FMIndex.load(fm_index_path)

    return index


fm_index_path = "dataset/nq/fm_index/corpus.fm_index"
fm_index = load_fm_index(fm_index_path)
for doc_index in range(fm_index.n_docs):
    doc = fm_index.get_token_index_from_row(doc_index)
    print(doc)
    exit()


