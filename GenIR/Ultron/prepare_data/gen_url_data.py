import json
from tqdm import tqdm
import nltk



if __name__ == '__main__':


    corpus_path = '../../../dataset/version3/msmarco/simplified-msmarco-D0-corpus.jsonl'
    corpus_path1 = '../../../dataset/version3/msmarco/simplified-msmarco-D1-corpus.jsonl'
    corpus_path2 = '../../../dataset/version3/msmarco/simplified-msmarco-D2-corpus.jsonl'
    corpus_path3 = '../../../dataset/version3/msmarco/simplified-msmarco-D3-corpus.jsonl'
    corpus_path4 = '../../../dataset/version3/msmarco/simplified-msmarco-D4-corpus.jsonl'
    corpus_path5 = '../../../dataset/version3/msmarco/simplified-msmarco-D5-corpus.jsonl'

    output_path = "dataset/msmarco-data/msmarco-docs-url.json"
    output_path1 = "dataset/msmarco-data/msmarco-docs-url_1.json"
    output_path2 = "dataset/msmarco-data/msmarco-docs-url_2.json"
    output_path3 = "dataset/msmarco-data/msmarco-docs-url_3.json"
    output_path4 = "dataset/msmarco-data/msmarco-docs-url_4.json"
    output_path5 = "dataset/msmarco-data/msmarco-docs-url_5.json"


    with open(corpus_path, "r") as fr, open(output_path, "w") as fw, open(corpus_path1, "r") as fr1, open(output_path1, "w") as fw1, open(corpus_path2, "r") as fr2, open(output_path2, "w") as fw2, open(corpus_path3, "r") as fr3, open(output_path3, "w") as fw3, open(corpus_path4, "r") as fr4, open(output_path4, "w") as fw4, open(corpus_path5, "r") as fr5, open(output_path5, "w") as fw5:

        for line in tqdm(fr, desc="Processing documents"):
            data = json.loads(line)
            body, docid, url, _ = data
            # Reformatting to include 'docid', 'body', and 'sents'
            doc_data = {
                "docid": f'{str(docid)}',
                "url": url.replace('\n', '')
            }
            fw.write(json.dumps(doc_data) + "\n")
            fw1.write(json.dumps(doc_data) + "\n")
            fw2.write(json.dumps(doc_data) + "\n")
            fw3.write(json.dumps(doc_data) + "\n")
            fw4.write(json.dumps(doc_data) + "\n")
            fw5.write(json.dumps(doc_data) + "\n")
        for line in tqdm(fr1, desc="Processing documents"):
            data = json.loads(line)
            body, docid, url, _ = data
            # Reformatting to include 'docid', 'body', and 'sents'
            doc_data = {
                "docid": f'{str(docid)}',
                "url": url.replace('\n', '')
            }
            fw1.write(json.dumps(doc_data) + "\n")
            fw2.write(json.dumps(doc_data) + "\n")
            fw3.write(json.dumps(doc_data) + "\n")
            fw4.write(json.dumps(doc_data) + "\n")
            fw5.write(json.dumps(doc_data) + "\n")
        for line in tqdm(fr2, desc="Processing documents"):
            data = json.loads(line)
            body, docid, url, _ = data
            # Reformatting to include 'docid', 'body', and 'sents'
            doc_data = {
                "docid": f'{str(docid)}',
                "url": url.replace('\n', '')
            }
            fw2.write(json.dumps(doc_data) + "\n")
            fw3.write(json.dumps(doc_data) + "\n")
            fw4.write(json.dumps(doc_data) + "\n")
            fw5.write(json.dumps(doc_data) + "\n")
        for line in tqdm(fr3, desc="Processing documents"):
            data = json.loads(line)
            body, docid, url, _ = data
            # Reformatting to include 'docid', 'body', and 'sents'
            doc_data = {
                "docid": f'{str(docid)}',
                "url": url.replace('\n', '')
            }
            fw3.write(json.dumps(doc_data) + "\n")
            fw4.write(json.dumps(doc_data) + "\n")
            fw5.write(json.dumps(doc_data) + "\n")
        for line in tqdm(fr4, desc="Processing documents"):
            data = json.loads(line)
            body, docid, url, _ = data
            # Reformatting to include 'docid', 'body', and 'sents'
            doc_data = {
                "docid": f'{str(docid)}',
                "url": url.replace('\n', '')
            }
            fw4.write(json.dumps(doc_data) + "\n")
            fw5.write(json.dumps(doc_data) + "\n")
        for line in tqdm(fr5, desc="Processing documents"):
            data = json.loads(line)
            body, docid, url, _ = data
            # Reformatting to include 'docid', 'body', and 'sents'
            doc_data = {
                "docid": f'{str(docid)}',
                "url": url.replace('\n', '')
            }
            fw5.write(json.dumps(doc_data) + "\n")