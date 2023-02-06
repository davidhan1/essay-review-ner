import json
import json_lines
import os
import jsonlines

DATA_PATH = 'D:\Study\Python_project\essay-review-NER\data'
LOG_PATH = 'D:\Study\Python_project\essay-review-NER\log'
CKPT_PATH = 'D:\Study\Python_project\essay-review-NER-big-data\ckpt'

def read_json(path, encoding='UTF-8'):
    raw = []
    with open(path, 'r', encoding=encoding) as f:
        raw = json.load(f)
    f.close()
    return raw

def write_json(path, data, encoding='UTF-8'):
    with open(path, 'w', encoding=encoding) as f:
        json.dump(data, f, ensure_ascii=False)
    f.close()

def read_jsonl(path):
    data = []
    with open(path, 'rb') as f: 
        for d in json_lines.reader(f):
            data.append(d)
    f.close()
    return data

def read_txt(path, encoding='UTF-8'):
    data = []
    with open(path, 'r', encoding=encoding) as f:
        for line in f:
            if line and line[-1] == '\n':
                line = line[:-1]
            data.append(line)
    f.close()
    return data

def write_txt(path, data: list, encoding='UTF-8'):
    f = open(path, 'w', encoding=encoding)
    for s in data:
        f.writelines(s)
    f.close()

def write_jsonl(path, data, encoding='UTF-8'):
    with open(path, "w", encoding=encoding) as w:
        for i in data:
            w.write(str(i).replace('\'', '\"') + '\n')

