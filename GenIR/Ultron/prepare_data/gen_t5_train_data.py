import os
import json
import random
import pickle
import argparse
import collections
import numpy as np
from tqdm import tqdm
from collections import Counter
from collections import defaultdict
from transformers import T5Tokenizer



def my_convert_tokens_to_ids(tokens:list, token_to_id:dict): # token_to_id is dict of word:id
    res = []
    for i, t in enumerate(tokens):
        if t in token_to_id:
            res += [token_to_id[t]]
        else:
            res += [token_to_id['<unk>']]
    return res

def my_convert_ids_to_tokens(input_ids:list, id_to_token:dict): # id_to_token is dict of id:word
    res = []
    for i, iid in enumerate(input_ids):
        if iid in id_to_token:
            res += [id_to_token[iid]]
        else:
            print("error!")
    return res

def add_padding(training_instance, tokenizer, id_to_token, token_to_id):
    input_ids = my_convert_tokens_to_ids(training_instance['tokens'], token_to_id)

    new_instance = {
        "input_ids": input_ids,
        "query_id": training_instance["doc_index"],
        "doc_id": training_instance["encoded_docid"],
    }
    return new_instance

def add_docid_to_vocab(doc_file_path):
    tokenizer = T5Tokenizer.from_pretrained(args.pretrain_model_path)
    vocab = tokenizer.get_vocab()
    new_tokens = []
    with open(doc_file_path) as fin:
        for i, line in tqdm(enumerate(fin), desc='constructing all_documents list'):
            data = json.loads(line)
            docid = data['docid'].lower()
            new_tokens.append("[{}]".format(docid))
    id_to_token = {vocab[k]:k for k in vocab}
    token_to_id = {id_to_token[k]:k for k in id_to_token}
    maxvid = max([k for k in id_to_token])
    start_doc_id = maxvid + 1
    for i, doc_id in enumerate(new_tokens):
        id_to_token[start_doc_id+i] = doc_id
        token_to_id[doc_id] = start_doc_id+i

    return id_to_token, token_to_id, new_tokens, list(vocab.values())

def get_encoded_docid(docid_path, all_docid=None, token_to_id=None):
    encoded_docid = {}
    if docid_path is None:
        for i, doc_id in enumerate(all_docid):
            encoded_docid[doc_id] = str(token_to_id[doc_id])  # atomic
    else:
        with open(docid_path, "r") as fr:
            for line in fr:
                docid, encode = line.strip().split("\t")
                docid = "[{}]".format(docid.lower().strip('[').strip(']'))
                encoded_docid[docid] = encode
    return encoded_docid

def build_idf(doc_file_path):
    tokenizer = T5Tokenizer.from_pretrained(args.pretrain_model_path)
    vocab = tokenizer.get_vocab()

    doc_count = 0
    idf_dict = {key: 0 for key in vocab}
    with open(doc_file_path) as fin:
        for doc_index, line in tqdm(enumerate(fin), desc='building term idf dict'):
            doc_count += 1
            doc_item = json.loads(line)
            docid = doc_item['docid'].lower()
            docid = "[{}]".format(docid)
            body =doc_item["body"]
            all_terms = set(tokenizer.tokenize((body).lstrip().lower()))

            for term in all_terms:
                if term not in idf_dict:
                    continue
                idf_dict[term] += 1
    
    for key in tqdm(idf_dict):
        idf_dict[key] = np.log(doc_count / (idf_dict[key]+1))

    return idf_dict

# 生成各种预训练任务的训练样本
# 任务1.1：passage --> docid
def gen_passage_instance(id_to_token, token_to_id, all_docid, encoded_docid):
    tokenizer = T5Tokenizer.from_pretrained(args.pretrain_model_path)
    
    sample_count = 0
    fw = open(args.output_path + 'pqssage2docid.json', "w")
    with open(args.data_path) as fin:
        for doc_index, line in tqdm(enumerate(fin), desc='generating samples'):
            max_num_tokens = args.max_seq_length - 1

            doc_item = json.loads(line)
            docid = doc_item['docid'].lower()
            docid = "[{}]".format(docid)
            sents_list = doc_item['sents']
            title = doc_item['sents'][0].lower().strip()
            head_terms = tokenizer.tokenize(title)
            current_chunk = head_terms[:]
            current_length = len(head_terms)
            
            sent_id = 0
            sample_for_one_doc = 0
            while sent_id < len(sents_list):
                sent = sents_list[sent_id].lower()
                sent_terms = tokenizer.tokenize(sent)
                current_chunk += sent_terms
                current_length += len(sent_terms)

                if sent_id == len(sents_list) - 1 or current_length >= max_num_tokens: 
                    tokens = current_chunk[:max_num_tokens] + ["</s>"] # truncate the sequence

                    training_instance = {
                        "doc_index":docid,
                        "encoded_docid":encoded_docid[docid],
                        "tokens": tokens,
                    }
                    training_instance = add_padding(training_instance, tokenizer, id_to_token, token_to_id)
                    fw.write(json.dumps(training_instance, ensure_ascii=False)+"\n")
                    sample_count += 1

                    sample_for_one_doc += 1
                    if sample_for_one_doc >= args.sample_for_one_doc:
                        break

                    current_chunk = head_terms[:]
                    current_length = len(head_terms)
                
                sent_id += 1
    fw.close()
    print("total count of samples: ", sample_count)

# 任务1.2：sampled terms --> docid
def gen_sample_terms_instance(id_to_token, token_to_id, all_docid, encoded_docid):
    print("gen_sample_terms_instance")
    tokenizer = T5Tokenizer.from_pretrained(args.pretrain_model_path)
    
    sample_count = 0
    fw = open(args.output_path + 'terms2docid.json', "w")


    idf_dict = build_idf(args.data_path)
    with open(args.data_path) as fin:
        for doc_index, line in tqdm(enumerate(fin), desc='generating samples'):
            max_num_tokens = args.max_seq_length - 1

            doc_item = json.loads(line)
            docid = doc_item['docid'].lower()
            docid = "[{}]".format(docid)
            body = doc_item['body'].lower().strip()
            all_terms = tokenizer.tokenize(body)[:1024]

            temp_tfidf = []
            all_valid_terms = []
            all_term_tfidf = []
            for term in all_terms:
                if term not in idf_dict:
                    continue
                tf_idf = all_terms.count(term) / len(all_terms) * idf_dict[term]
                temp_tfidf.append((term, tf_idf))
                all_term_tfidf.append(tf_idf)
            if len(all_term_tfidf) < 10:
                continue
            tfidf_threshold = sorted(all_term_tfidf, reverse=True)[min(max_num_tokens, len(all_term_tfidf))-1]
            for idx, (term, tf_idf) in enumerate(temp_tfidf):
                if tf_idf >= tfidf_threshold:
                    all_valid_terms.append(term)

            if len(set(all_valid_terms)) < 2:
                continue

            tokens = all_valid_terms[:max_num_tokens] + ["</s>"]
            training_instance = {
                "query_id":docid,
                "doc_id":encoded_docid[docid],
                "input_ids": my_convert_tokens_to_ids(tokens, token_to_id),
            }

            fw.write(json.dumps(training_instance, ensure_ascii=False)+"\n")
            sample_count += 1

    fw.close()

    print("total count of samples: ", sample_count)

# 任务1.3：source docid --> target docid
def gen_enhanced_docid_instance(label_filename, train_filename):
    fw = open(args.output_path, "w")
    label_dict = get_encoded_docid(label_filename)
    train_dict = get_encoded_docid(train_filename)
    for docid, encoded in train_dict.items():
        input_ids = [int(item) for item in encoded.split(',')]
        training_instance = {"input_ids": input_ids, "query_id": docid, "doc_id": label_dict[docid]}
        fw.write(json.dumps(training_instance, ensure_ascii=False)+"\n")
    fw.close()
 
# 任务2：pseudo query --> docid
def gen_fake_query_instance(id_to_token, token_to_id, all_docid, encoded_docid):
    tokenizer = T5Tokenizer.from_pretrained(args.pretrain_model_path)
         
    # 从ms-marco数据集中检索出点击了docid的对应query 
    fw = open(args.output_path + 'pseudo2docid.json', "w")
    max_num_tokens = args.max_seq_length - 1

    
    with open(args.fake_query_path, "r") as fr:
        for line in tqdm(fr, desc="load all fake queries"):
            docid, query = line.strip("\n").split("\t")
            if docid not in token_to_id:
                continue

            query_terms = tokenizer.tokenize(query.lower())
            tokens = query_terms[:max_num_tokens] + ["</s>"]

            training_instance = {
                "doc_index":docid,
                "encoded_docid":encoded_docid[docid],
                "tokens": tokens,
            }
            training_instance = add_padding(training_instance, tokenizer, id_to_token, token_to_id)
            fw.write(json.dumps(training_instance, ensure_ascii=False)+"\n")

    fw.close()

# 任务3：query --> docid,  finetune

def gen_query_instance(id_to_token, token_to_id, all_docid, encoded_docid, input_path, output_path,
                                  pretrain_model_path, max_seq_length):
    tokenizer = T5Tokenizer.from_pretrained(pretrain_model_path)

    # 打开输出文件
    with open(output_path + 'query2docid.json', "w") as fw:
        # 读取输入的JSONL文件
        lines = [line for line in open(input_path)][:-1000]

        for line in tqdm(lines, desc="reading input JSONL"):
            # 解析每一行的JSON
            data = json.loads(line.strip())
            query = data['question_text'].lower()
            docid = "[{}]".format(str(data['document_index']).lower())

            # 检查docid是否存在于token_to_id中
            if docid not in token_to_id:
                continue

            # 对查询进行标记
            query_terms = tokenizer.tokenize(query)
            max_num_tokens = max_seq_length - 1
            tokens = query_terms[:max_num_tokens] + ["</s>"]

            # 创建训练实例
            training_instance = {
                "doc_index": docid,
                "encoded_docid": encoded_docid[docid],
                "tokens": tokens,
            }

            # 添加填充
            training_instance = add_padding(training_instance, tokenizer, id_to_token, token_to_id)

            # 写入训练实例到文件
            fw.write(json.dumps(training_instance, ensure_ascii=False) + "\n")


def gen_eval_query_instance(id_to_token, token_to_id, all_docid, encoded_docid, input_path, output_path,
                                    pretrain_model_path, max_seq_length):
        tokenizer = T5Tokenizer.from_pretrained(pretrain_model_path)

        # 打开输出文件
        with open(output_path + 'eval_query2docid.json', "w") as fw:
            # 读取输入的JSONL文件

            lines = [line for line in open(input_path)][-1000:]


            for line in tqdm(lines, desc="reading input JSONL"):
                # 解析每一行的JSON
                data = json.loads(line.strip())
                query = data['question_text'].lower()
                docid = "[{}]".format(str(data['document_index']).lower())

                # 检查docid是否存在于token_to_id中
                if docid not in token_to_id:
                    continue

                # 对查询进行标记
                query_terms = tokenizer.tokenize(query)
                max_num_tokens = max_seq_length - 1
                tokens = query_terms[:max_num_tokens] + ["</s>"]

                # 创建训练实例
                training_instance = {
                    "doc_index": docid,
                    "encoded_docid": encoded_docid[docid],
                    "tokens": tokens,
                }

                # 添加填充
                training_instance = add_padding(training_instance, tokenizer, id_to_token, token_to_id)

                # 写入训练实例到文件
                fw.write(json.dumps(training_instance, ensure_ascii=False) + "\n")
        nums = [126739, 21518, 20631, 19639, 19105, 18548]
        nums = [126739, 126739 + 21518, 126739 + 21518 + 20631, 126739 + 21518 + 20631 + 19639, 126739 + 21518 + 20631 + 19639 + 19105, 126739 + 21518 + 20631 + 19639 + 19105 + 18548]

        for i in range(1, 6):

            id_to_token, token_to_id, all_docid, all_term = add_docid_to_vocab(args.data_path.replace("all", f"all_{i}"))
            encoded_docid = get_encoded_docid(args.docid_path.replace("all", f"all_{i}"), all_docid, token_to_id)

            with open(output_path + f'eval_query2docid_{i}.json', "w") as fw:
                # 读取输入的JSONL文件
                lines = [line for line in open(input_path.replace("D0", f"D{i}"))]
                cnt = 0
                for line in tqdm(lines, desc="reading input JSONL"):


                    # 解析每一行的JSON
                    data = json.loads(line.strip())
                    index = data['document_index']
                    if index < nums[i-1] or index >= nums[i]:
                        continue


                    query = data['question_text'].lower()
                    docid = "[{}]".format(str(data['document_index']).lower())

                    # 检查docid是否存在于token_to_id中
                    if docid not in token_to_id:
                        continue

                    # 对查询进行标记
                    query_terms = tokenizer.tokenize(query)
                    max_num_tokens = max_seq_length - 1
                    tokens = query_terms[:max_num_tokens] + ["</s>"]

                    # 创建训练实例
                    training_instance = {
                        "doc_index": docid,
                        "encoded_docid": encoded_docid[docid],
                        "tokens": tokens,
                    }

                    # 添加填充
                    training_instance = add_padding(training_instance, tokenizer, id_to_token, token_to_id)

                    # 写入训练实例到文件
                    cnt += 1
                    fw.write(json.dumps(training_instance, ensure_ascii=False) + "\n")
                    if cnt == 1000:
                        break
                print(cnt)




if __name__ == "__main__":
    MaskedLmInstance = collections.namedtuple("MaskedLmInstance", ["index", "label"])

    parser = argparse.ArgumentParser()
    parser.add_argument("--max_seq_length", default=512, type=int, help="max sequence length of model. default to 512.")
    parser.add_argument("--pretrain_model_path", default="../../../huggingface/t5-base", type=str,
                        help='bert model path')
    parser.add_argument("--data_path", default="dataset/msmarco-data/msmarco-docs-sents-all.json", type=str,
                        help='data path')
    parser.add_argument("--docid_path", default='dataset/encoded_docid/t5_pq_all.txt', type=str, help='docid path')
    parser.add_argument("--query_path", default="../../../dataset/version3/msmarco/simplified-msmarco-D0.jsonl", type=str,
                        help='data path')
    parser.add_argument("--output_path", default="dataset/msmarco-data/train_data_pq/", type=str,
                        help='output path')
    parser.add_argument("--fake_query_path", default="dataset/msmarco-data/corpus_pseudo_query.txt", type=str, help='fake query path')
    parser.add_argument("--sample_for_one_doc", default=10, type=int,
                        help="max number of passages sampled for one document.")
    parser.add_argument("--current_data", default=None, type=str, help="current generating data.")
    parser.add_argument("--msmarco_msmarco", default='msmarco', type=str, help="")

    args = parser.parse_args()


    if not os.path.exists(args.output_path):
        os.system(f"mkdir -p {args.output_path}")

    id_to_token, token_to_id, all_docid, all_term = add_docid_to_vocab(args.data_path)
    encoded_docid = get_encoded_docid(args.docid_path, all_docid, token_to_id)
    # gen_passage_instance(id_to_token, token_to_id, all_docid, encoded_docid)
    #
    # gen_sample_terms_instance(id_to_token, token_to_id, all_docid, encoded_docid)

    gen_fake_query_instance(id_to_token, token_to_id, all_docid, encoded_docid)


    gen_query_instance(id_to_token, token_to_id, all_docid, encoded_docid, args.query_path, args.output_path,
                       args.pretrain_model_path, args.max_seq_length)

    merged_file = os.path.join(args.output_path, f"merged_train_data.json")
    with open(merged_file, "w") as fw:
        with open(os.path.join(args.output_path, 'pqssage2docid.json'), "r") as fr:
            for line in fr:
                fw.write(line)
        with open(os.path.join(args.output_path, 'terms2docid.json'), "r") as fr:
            for line in fr:
                fw.write(line)
        with open(os.path.join(args.output_path, 'pseudo2docid.json'), "r") as fr:
            for line in fr:
                fw.write(line)
        with open(os.path.join(args.output_path, 'query2docid.json'), "r") as fr:
            for line in fr:
                fw.write(line)
    pretrain_file = os.path.join(args.output_path, f"pretrain_data.json")
    with open(pretrain_file, "w") as fw:
        with open(os.path.join(args.output_path, 'pqssage2docid.json'), "r") as fr:
            for line in fr:
                fw.write(line)
        with open(os.path.join(args.output_path, 'terms2docid.json'), "r") as fr:
            for line in fr:
                fw.write(line)

    gen_eval_query_instance(id_to_token, token_to_id, all_docid, encoded_docid, args.query_path, args.output_path,
                       args.pretrain_model_path, args.max_seq_length)
