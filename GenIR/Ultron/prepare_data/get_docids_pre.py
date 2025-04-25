import nanopq
import numpy as np
import pickle
from tqdm import tqdm
from transformers import T5Tokenizer, T5Model, T5ForConditionalGeneration

def load_pq_model(pq_model_path):
    """
    Load the pre-trained PQ model from a file.
    """
    with open(pq_model_path, "rb") as fr:
        pq = pickle.load(fr)
    return pq

def load_doc_vec(input_path):
    docid_2_idx, idx_2_docid = {}, {}
    idx_2_docid = {}
    doc_embeddings = []

    with open(input_path, "r") as fr:
        for line in tqdm(fr, desc="loading doc vectors..."):
            did, demb = line.strip().split('\t')
            d_embedding = [float(x) for x in demb.split(',')]

            docid_2_idx[did] = len(docid_2_idx)
            idx_2_docid[docid_2_idx[did]] = did

            doc_embeddings.append(d_embedding)

    print("successfully load doc embeddings.")
    return docid_2_idx, idx_2_docid, np.array(doc_embeddings, dtype=np.float32)


def product_quantization_docid(docid_2_idx, idx_2_docid, doc_embeddings, output_path, pq):
    print("generating product quantization docids...")
    model = T5ForConditionalGeneration.from_pretrained('../../../huggingface/t5-base')
    vocab_size = model.config.vocab_size


    # Encode to PQ-codes
    print("encoding doc embeddings...")
    X_code = pq.encode(doc_embeddings)  # [#doc, sub_space] with dtype=np.uint8
    return X_code
    with open(output_path, "w") as fw:
        for idx, doc_code in tqdm(enumerate(X_code), desc="writing doc code into the file..."):
            docid = idx_2_docid[idx]
            new_doc_code = [int(x) for x in doc_code]
            for i, x in enumerate(new_doc_code):
                new_doc_code[i] = int(x) + i*256
            code = ','.join(str(x + vocab_size) for x in new_doc_code)
            fw.write(docid + "\t" + code + "\n")




if __name__ == "__main__":
    # Define paths
    pq_model_path = "dataset/encoded_docid/t5_pq_all.pkl"
    model = T5ForConditionalGeneration.from_pretrained('../../../huggingface/t5-base')
    pq = load_pq_model(pq_model_path)
    vocab_size = model.config.vocab_size
    with open('dataset/encoded_docid/t5_pq_all_1.txt', 'a') as fw1, open('dataset/encoded_docid/t5_pq_all_2.txt', 'a') as fw2, open('dataset/encoded_docid/t5_pq_all_3.txt', 'a') as fw3, open('dataset/encoded_docid/t5_pq_all_4.txt', 'a') as fw4, open('dataset/encoded_docid/t5_pq_all_5.txt', 'a') as fw5:
        for k in range(1, 6):
            new_doc_embeddings_path = f"dataset/doc_embed/t5_512_doc_all_{k}.txt"
            output_path = f"dataset/encoded_docid/t5_pq_all_{k}.txt"

            docid_2_idx, idx_2_docid, doc_embeddings = load_doc_vec(new_doc_embeddings_path)
            X_code = product_quantization_docid(docid_2_idx, idx_2_docid, doc_embeddings, output_path, pq)
            for idx, doc_code in tqdm(enumerate(X_code), desc="writing doc code into the file..."):
                docid = idx_2_docid[idx]
                new_doc_code = [int(x) for x in doc_code]
                for i, x in enumerate(new_doc_code):
                    new_doc_code[i] = int(x) + i*256
                code = ','.join(str(x + vocab_size) for x in new_doc_code)
                if k == 1:
                    print(docid + "\t" + code + "\n")
                    fw1.write(docid + "\t" + code + "\n")
                    fw2.write(docid + "\t" + code + "\n")
                    fw3.write(docid + "\t" + code + "\n")
                    fw4.write(docid + "\t" + code + "\n")
                    fw5.write(docid + "\t" + code + "\n")
                elif k == 2:
                    fw2.write(docid + "\t" + code + "\n")
                    fw3.write(docid + "\t" + code + "\n")
                    fw4.write(docid + "\t" + code + "\n")
                    fw5.write(docid + "\t" + code + "\n")
                elif k == 3:
                    fw3.write(docid + "\t" + code + "\n")
                    fw4.write(docid + "\t" + code + "\n")
                    fw5.write(docid + "\t" + code + "\n")
                elif k == 4:
                    fw4.write(docid + "\t" + code + "\n")
                    fw5.write(docid + "\t" + code + "\n")
                else:
                    fw5.write(docid + "\t" + code + "\n")

