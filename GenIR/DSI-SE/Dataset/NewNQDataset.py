from torch.utils.data import Dataset
from abc import ABC
import torch
from torch.nn.utils.rnn import pad_sequence

class NewNQDataset(Dataset, ABC):
    def __init__(self, data, corpus, tokenizer, max_len=128):
        self.data = data
        self.corpus = corpus
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, item):
        task, query, doc_id = self.data[item]
        if task == 'task1':
            query = '<TASK1>' + ' ' + query
        else:
            query = '<TASK2>' + ' ' + query
        doc = self.corpus[doc_id][2]

        doc = ''.join(doc)

        # doc = doc_id
        return (torch.tensor(self.tokenizer.encode(str(query), truncation=True, max_length=self.max_len)),
                torch.tensor(self.tokenizer.encode(str(doc), truncation=True, max_length=self.max_len)))


    def __len__(self):
        return len(self.data)

    @staticmethod
    def collate_fn(data):
        inputs, outputs = zip(*data)
        inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
        return {
            'input_ids': inputs,
            'attention_mask': inputs.ne(0),
            'labels': pad_sequence(outputs, batch_first=True, padding_value=-100),
        }

