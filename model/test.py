import sys
sys.path.append('D:\Study\Python_project\essay-review-NER')

from utils.utils import *

# 查看已标注标签
# data = []
# raw = read_jsonl(os.path.join(DATA_PATH, 'test', 'train.jsonl'))
# for d in raw:
#     token = d['data']
#     label = d['label']
#     for start, end, _ in label:
#         data.append(token[start:end])
# print(len(data))

# 切分原始数据
# data = []
# raw = read_txt(os.path.join(DATA_PATH, 'raw', 'review.txt'))
# print(len(raw))
# cnt = 1
# for s in raw:
#     if s in data:
#         continue
#     data.append(s + '\n')
#     if len(data) == 2000:
#         write_txt(os.path.join(DATA_PATH, 'raw', 'review_' + str(cnt) + '.txt'), data)
#         data = []
#         cnt += 1
# if data:
#     write_txt(os.path.join(DATA_PATH, 'raw', 'review_' + str(cnt) + '.txt'), data)

print((-1,)+(2))