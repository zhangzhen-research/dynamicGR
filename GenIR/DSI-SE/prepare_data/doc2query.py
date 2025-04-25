import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM
from util.io import read_file, write_file
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = AutoTokenizer.from_pretrained('doc2query-t5-base-msmarco')
model = T5ForConditionalGeneration.from_pretrained('doc2query-t5-base-msmarco')
model.to(device)



BATCH_SIZE = 32  


def generate_queries_batch(batch):
    docs, urls = zip(*batch)
    input_ids = tokenizer(list(docs), return_tensors='pt', truncation=True, padding=True, max_length=512).input_ids.to(
        device)

    outputs = model.generate(
        input_ids=input_ids,
        max_length=64,
        do_sample=True,
        top_k=20,
        num_return_sequences=15
    )

    results = []
    for i, url in enumerate(urls):
        query_list = [tokenizer.decode(outputs[i * 15 + j], skip_special_tokens=True) for j in range(15)]
        results.append((url, [docs[i], query_list]))

    return results



if __name__ == "__main__":
    url2doc = read_file('nq-D0-corpus.jsonl')
    url2doc_items = url2doc

    
    batched_data = [url2doc_items[i:i + BATCH_SIZE] for i in range(0, len(url2doc_items), BATCH_SIZE)]

    results = []
    for batch in tqdm(batched_data):
        results.extend(generate_queries_batch(batch))

    
    sorted(results, key=lambda x: x[0])

    write_file('data/dataset/D0/corpus_pseudo_query.json', results)