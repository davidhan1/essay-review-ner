import sys
sys.path.append('D:\Study\Python_project\essay-review-NER')

from utils.utils import *

data = []
raw = read_jsonl(os.path.join(DATA_PATH, 'test', 'train.jsonl'))
for d in raw:
    label = ['O'] * len(d['data'])
    for i, j, tag in d['label']:
        label[i] = 'B-' + tag[0][0]           
        label[i + 1:j] = ['I-' + tag[0][0]] * (j - i - 1)
    data.append({
        'tokens': [c for c in d['data']],
        'ner_tags': label
    })

p_split = int(len(data) * 0.8)
train_data = data[:p_split]
val_data = data[p_split:]

write_json(os.path.join(DATA_PATH, 'test', 'train.json'), train_data)
write_json(os.path.join(DATA_PATH, 'test', 'val.json'), val_data)