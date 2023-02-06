import sys
sys.path.append('D:\Study\Python_project\essay-review-NER')

from utils.utils import *

data = read_jsonl(r'D:\Study\Python_project\essay-review-NER\doccano\0-2000\review1.jsonl')
res = []
for d in data:
    res.append({'text': d['data'], 'label': d['label']})
write_jsonl(r'D:\Study\Python_project\essay-review-NER\doccano\0-2000\review1_new.jsonl', res)