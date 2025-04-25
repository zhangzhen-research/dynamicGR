import argparse
import json
import os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from accelerate import Accelerator

from util.io import read_file
from Dataset.NewNQDataset import NewNQDataset
from genre.trie import Trie, MarisaTrie
from eval.eval import eval_all  


def eval_model(model, data_loader, tokenizer, trie):
    model.eval()
    tk0 = tqdm(data_loader, total=len(data_loader))
    acc = []
    output_all = []
    label_all = []
    top_k = 100
    with torch.no_grad():
        for batch in tk0:
            batch = {k: v.cuda() for k, v in batch.items()}
            output = model.generate(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                max_length=128,
                num_beams=top_k,
                num_return_sequences=top_k,
                length_penalty=1,
                no_repeat_ngram_size=None,
                early_stopping=False,
                prefix_allowed_tokens_fn=lambda batch_id, sent: trie.get(sent.tolist()),
            )
            output = tokenizer.batch_decode(output, skip_special_tokens=True)
            output = [str(x).replace('$', '').strip() for x in output]
            beam = []
            new_output = []
            for line in output:
                if len(beam) >= top_k:
                    new_output.append(beam)
                    beam = []
                beam.append(line)
            new_output.append(beam)
            batch['labels'][batch['labels'] == -100] = 0
            labels = tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)
            labels = [str(x).replace('$', '').replace(', ', '').strip() for x in labels]

            acc.extend([int(l in o) for o, l in zip(new_output, labels)])
            tk0.set_postfix(acc=sum(acc) / len(acc))

            output_all.extend(new_output)
            label_all.extend(labels)

        eval_output = eval_all(output_all, label_all)
        return eval_output


def main(args):
    accelerator = Accelerator()
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.add_tokens(['<TASK1>', '<TASK2>'])
    for i in range(args.doc_id_num):
        tokenizer.add_tokens([f'${i}$'])

    
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    model.resize_token_embeddings(len(tokenizer))
    model.load_state_dict(torch.load(args.checkpoint_path))
    model.to(accelerator.device)

    
    for i in range(6):
        mode = f'D{i}'
        test_data = read_file(f'your_test.json_path')
        corpus = []
        for k in range(i + 1):
            corpus.extend(read_file(f'data/dataset/D{k}/corpus_docids.json'))

        corpus_ids = [[0] + tokenizer.encode(''.join(line[2])) for line in corpus]
        print(corpus_ids[0])
        corpus_ids = list(set([tuple(x) for x in corpus_ids]))
        corpus_ids = [list(x) for x in corpus_ids]
        trie = MarisaTrie(corpus_ids)

        test_dataset = NewNQDataset(test_data, corpus, tokenizer, max_len=args.max_len)
        test_data_loader = DataLoader(test_dataset, collate_fn=test_dataset.collate_fn, batch_size=1, shuffle=False,
                                      num_workers=4)
        model, test_data_loader = accelerator.prepare(model, test_data_loader)
        eval_output = eval_model(model, test_data_loader, tokenizer, trie)
        with open(f'data/output/results/eval_output_{mode}.json', 'w') as f:
            json.dump(eval_output, f, indent=2)

        if i != 0:
            test_data = read_file(f'data/dataset/{mode}/test.json')
            corpus = []
            for k in range(i + 1):
                corpus += read_file(f'data/dataset/D{k}/corpus_docids.json')

            corpus_ids = [[0] + tokenizer.encode(''.join(line[2])) for line in corpus]
            trie = MarisaTrie(corpus_ids)

            test_dataset = NewNQDataset(test_data, corpus, tokenizer, max_len=args.max_len)
            test_data_loader = DataLoader(test_dataset, collate_fn=test_dataset.collate_fn, batch_size=1, shuffle=False,
                                          num_workers=4)
            model, test_data_loader = accelerator.prepare(model, test_data_loader)
            eval_output = eval_model(model, test_data_loader, tokenizer, trie)
            with open(f'data/output/results/eval_output_new_{mode}.json', 'w') as f:
                json.dump(eval_output, f, indent=2)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluation Script")

    parser.add_argument('--model_name', type=str, default='t5-base', help='Model name (e.g., T5-small, T5-base, etc.)')
    parser.add_argument('--doc_id_num', type=int, default=50, help='Num of the document ID')
    parser.add_argument('--max_len', type=int, default=32, help='Maximum token length for the input sequences')
    parser.add_argument('--checkpoint_path', type=str, default='data/output/ckpts/', help='Path to the model checkpoint to evaluate')
    args = parser.parse_args()
    main(args)