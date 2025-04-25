from abc import ABC
import argparse

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
from transformers import AdamW, Adafactor, get_linear_schedule_with_warmup, get_constant_schedule
from accelerate import Accelerator
from torch.utils.data import Dataset
from tqdm import tqdm
import torch
import json
from util.io import write_file, read_file
import os
from Dataset.NewNQDataset import NewNQDataset

from torch.optim.lr_scheduler import LambdaLR
import fcntl




def custom_lr_lambda(warmup_steps, total_steps, min_lr=0.0005):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        else:
            current_lr = max(min_lr, 1.0 - (float(step) - warmup_steps) / float(max(1, total_steps - warmup_steps)))
            return current_lr if current_lr > min_lr else min_lr
    return lr_lambda



def eval_model(model, data_loader, tokenizer, trie):
    model.eval()
    tk0 = tqdm(data_loader, total=len(data_loader))
    acc = []
    output_all = []
    label_all = []
    top_k = 1
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
                min_length=None,
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
        from eval.eval import eval_all
        eval_output = eval_all(output_all, label_all)
        return eval_output

def train(args):


    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    accelerator = Accelerator()

    id_num = args.doc_id_num
    save_path = f'data/output/ckpts/'



    if not os.path.exists(save_path):
        os.makedirs(save_path)

    
    with open(f'{save_path}/args.json', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    model = AutoModelForSeq2SeqLM.from_pretrained(f't5-base')
    tokenizer = AutoTokenizer.from_pretrained(f't5-base')

    tokenizer.add_tokens(['<TASK1>', '<TASK2>'])
    for i in range(id_num):
        tokenizer.add_tokens([f'${i}$'])
    model.resize_token_embeddings(len(tokenizer))

    data = read_file(f'your_train.json_path')

    test_data = read_file(f'your_test.json_path')

    corpus = read_file(
        f'your_corpus_docids.json_path')

    from genre.trie import Trie, MarisaTrie

    corpus_ids = [[0] + tokenizer.encode(''.join(line[2])) for line in corpus]
    trie = MarisaTrie(corpus_ids)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    dataset = NewNQDataset(data=data, corpus=corpus, tokenizer=tokenizer, max_len=args.max_len)
    accelerator.print(f'data size={len(dataset)}')
    data_loader = torch.utils.data.DataLoader(dataset, collate_fn=dataset.collate_fn, batch_size=args.batch_size,
                                              shuffle=True, num_workers=4)
    test_dataset = NewNQDataset(data=test_data, corpus=corpus, tokenizer=tokenizer, max_len=args.max_len)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, collate_fn=test_dataset.collate_fn,
                                                   batch_size=1,
                                                   shuffle=False, num_workers=4)
    model, optimizer, data_loader = accelerator.prepare(model, optimizer, data_loader)

    
    total_training_steps = len(data_loader) * args.epochs
    warmup_steps = int(0.2 * total_training_steps)

    
    if warmup_steps > 0:
        scheduler = LambdaLR(optimizer, custom_lr_lambda(warmup_steps=warmup_steps, total_steps=total_training_steps))
    else:
        scheduler = get_constant_schedule(optimizer)

    os.makedirs(save_path, exist_ok=True)
    accelerator.print(tokenizer.decode(dataset[128][0]))
    accelerator.print('==>')
    accelerator.print(tokenizer.decode(dataset[128][1]), dataset[128][1])
    last_eval_output = {'hit@1': -1, 'mrr': -1}
    flag = 0

    epoch = 0
    while True:
        accelerator.print(f'Training epoch {epoch}')
        accelerator.wait_for_everyone()
        model.train()
        tk0 = tqdm(data_loader, total=len(data_loader))
        loss_report = []
        for batch in tk0:
            out = model(**batch)
            loss = out.loss
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            loss_report.append(loss.item())
            tk0.set_postfix(loss=sum(loss_report) / len(loss_report))
        accelerator.wait_for_everyone()
        if accelerator.is_local_main_process and (epoch + 1) % args.eval_interval == 0:
            accelerator.save(accelerator.unwrap_model(model).state_dict(), f'{save_path}/{epoch}.pt')
            eval_output = eval_model(model, test_data_loader, tokenizer, trie)
            model.train()
            with open(f'{save_path}/eval_epoch_{epoch}.json', 'w') as eval_file:
                json.dump(eval_output, eval_file)
            if eval_output['hit@1'] > last_eval_output['hit@1']:
                last_eval_output = eval_output
            elif eval_output['hit@1'] == 0:
                last_eval_output = eval_output
            elif epoch > args.epochs:
                flag = 1

        log_file = f'{save_path}/log.txt'
        with open(log_file, 'a') as f:
            avg_loss = sum(loss_report) / len(loss_report)
            f.write(f'Epoch {epoch}: Average Loss = {avg_loss}\n')
        if flag == 1:
            break
        epoch += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training Script")

    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--model_name', type=str, default='t5-base', help='Model size (e.g., T5-small, T5-base, etc.)')
    parser.add_argument('--doc_id_num', type=int, default=50, help='Num of the document ID')
    parser.add_argument('--max_len', type=int, default=32, help='Maximum token length for the input sequences')
    parser.add_argument('--eval_interval', type=int, default=5)

    
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--warmup_steps', type=int, default=10000, help='Number of warmup steps for learning rate schedule')

    args = parser.parse_args()

    train(args)


