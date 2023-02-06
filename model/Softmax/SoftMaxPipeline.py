import sys
sys.path.append('D:\Study\Python_project\essay-review-NER')

from utils.utils import *

import os
# os.environ['GPU_VISIBLE_DEVICE'] = "0"
import sys
from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch
from torch import nn

from typing import List

import torch
from torch.nn import Module, functional
from tqdm import tqdm
from transformers import BertTokenizerFast
from tqdm import tqdm


class Pipeline:
    def __init__(self, model: Module, tokenizer: BertTokenizerFast, device: str=-1):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    def ensure_tensor_on_device(self, device, **inputs):
        if device == -1:
            return {
                name: torch.tensor(tensor)
                for name, tensor in inputs.items()
            }
        else:
            return {
                name: torch.tensor(tensor).cuda()
                for name, tensor in inputs.items()
            }
    
    def __call__(self, queries: List[str], batch_size: int = 16):
        L = len(queries)

        res = []

        for l in tqdm(range(0, L, batch_size)):
            r = min(l + batch_size, L)
            batched_queries = queries[l: r]

            with torch.inference_mode():
                tokenized_queries = self.tokenizer(batched_queries, truncation=True, padding=True, return_offsets_mapping=True)
                offsets_mapping = tokenized_queries.pop('offset_mapping')
                # print(offsets_mapping)
                tokenized_queries = self.ensure_tensor_on_device(self.device, **tokenized_queries)
                outputs = self.model(**tokenized_queries)[0]
                outputs = torch.argmax(outputs, dim=-1)
                # print(outputs.shape)
                for i in range(len(batched_queries)):
                    cur = []
                    for j in range(len(offsets_mapping[i])):
                        start, end = offsets_mapping[i][j][0], offsets_mapping[i][j][1]
                        if start != end:
                            cur.append({
                                'entity': 'LABEL_' + str(int(outputs[i][j])),
                                'word': batched_queries[i][start:end],
                                'start': start,
                                'end': end
                            })
                    res.append(cur)
        return res


class NER_TRF:
    def __init__(self,
                 model_checkpoint=r'D:\Study\Python_project\essay-review-NER-big-data\ckpt\bert-base-chinese_fp32_ckpt\checkpoint-470',
                 device=0):
        print('device', device)
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

        model = AutoModelForTokenClassification.from_pretrained(
            model_checkpoint)
        if device < 0:
            model = model.to('cpu')
        else:
            model = model.cuda()

        self.ner_pipline = Pipeline(model=model, tokenizer=tokenizer, device=device)

        self.label2tag = {
            'LABEL_0': 'O',
            'LABEL_1': 'B-1',
            'LABEL_2': 'I-1',
            'LABEL_3': 'B-2',
            'LABEL_4': 'I-2',
            'LABEL_5': 'B-3',
            'LABEL_6': 'I-3'
        }

        self.colors = {
            'I-1': '\033[92m',  # GREEN
            'B-1': '\033[92m',  # GREEN
            'I-2': '\033[93m',  # YELLOW
            'B-2': '\033[93m',  # YELLOW
            'I-3': '\033[91m',  # RED
            'B-3': '\033[91m',  # RED
            'O': '\033[0m'
        }

    def __call__(self, query):
        return self.ner_pipline(query)

    def get_tagged_entity(self, query: list) -> list:
        # print(query)
        query_outs = self.ner_pipline(query)
        # print(query_outs)
 
        res0 = []
        for query_out in query_outs:
            res = []
            string = tag = ''
            start = end = 0
            # print(query_out)
            for elm in query_out:
                # print(elm)
                _token = elm['word']
                _tag = self.label2tag[elm['entity']]
                _start = elm['start']
                _end = elm['end']

                if _tag == 'O' and string:
                    res.append((string, tag, start, end))
                    string = ''
                elif _tag.startswith('I'):
                    string += _token
                    end = _end
                elif _tag.startswith('B'):
                    if string:
                        res.append((string, tag, start, end))
                    string = _token
                    tag = _tag[2:]
                    start = _start
                    end = _end

            if string:
                res.append((string, tag, start, end))
            res = [e for e in res if e[1]]
            res0.append(res)

        return res0

    def show(self, query):
        # print(query)
        outs = self.ner_pipline(query)
        # print(outs)
        res = []
        for out in outs:
            # print(out)
            cur = ''
            for elm in out:
                token = elm['word']
                tag = self.label2tag[elm['entity']]
                cur += self.colors[tag] + token + self.colors['O']
            res.append(cur)
        # print(res)
        for k in res:
            print(k)
            # pass
            


if __name__ == '__main__':
    ner = NER_TRF()
    data = read_txt(os.path.join(DATA_PATH, 'raw', 'review_9.txt'))
    data = data[:300]
    # entity_res = ner.get_tagged_entity(data)
    # print(entity_res)
    # print(data)
    ner.show(data)

