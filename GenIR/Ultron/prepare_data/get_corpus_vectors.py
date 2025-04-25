from sentence_transformers import SentenceTransformer
import torch

from util.io import read_file, write_file
from tqdm import tqdm

# 加载预训练的 Sentence-Transformers 模型
model_name = '../../../huggingface/gtr-t5-base/models--sentence-transformers--gtr-t5-base/snapshots/7027e9594267928589816394bdd295273ddc0739'
model = SentenceTransformer(model_name,
                            cache_folder=model_name)
# 将模型移动到GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)



# url2doc_list = read_file('../../../dataset/version3/msmarco/simplified-msmarco-D0-corpus.jsonl')
#
# batch_size = 64
# corpus = [doc for doc, index, _, _ in url2doc_list]
#
# with tqdm(total=len(corpus)) as pbar:
#     for i in range(0, len(corpus), batch_size):
#         batch_docs = corpus[i:i + batch_size]
#
#         # 获取文档的嵌入
#         batch_embeddings = model.encode(batch_docs, convert_to_tensor=True, device=device)
#
#         # 将结果移回CPU并转换为列表
#         batch_embeddings = batch_embeddings.cpu().tolist()
#
#         for j, doc in enumerate(batch_docs):
#             url2doc_list[i + j] = url2doc_list[i + j] + [batch_embeddings[j]]
#
#         pbar.update(batch_size)
#
# sorted(url2doc_list, key=lambda x: x[1])
#
# with open('dataset/doc_embed/t5_512_doc_all.txt', 'w') as f:
#     for item in url2doc_list:
#         f.write(f'[{item[1]}]\t{str(item[-1])[1:-1]}\n')

batch_size = 64
for k in range(1, 6):
    url2doc_list = read_file(f'../../../dataset/version3/msmarco/simplified-msmarco-D{k}-corpus.jsonl')
    corpus = [doc for doc, index, _, _ in url2doc_list]
    with tqdm(total=len(corpus)) as pbar:
        for i in range(0, len(corpus), batch_size):
            batch_docs = corpus[i:i + batch_size]

            # 获取文档的嵌入
            batch_embeddings = model.encode(batch_docs, convert_to_tensor=True, device=device)

            # 将结果移回CPU并转换为列表
            batch_embeddings = batch_embeddings.cpu().tolist()

            for j, doc in enumerate(batch_docs):
                url2doc_list[i + j] = url2doc_list[i + j] + [batch_embeddings[j]]

            pbar.update(batch_size)
    sorted(url2doc_list, key=lambda x: x[1])
    with open(f'dataset/doc_embed/t5_512_doc_all_{k}.txt', 'w') as f:
        for item in url2doc_list:
            f.write(f'[{item[1]}]\t{str(item[-1])[1:-1]}\n')