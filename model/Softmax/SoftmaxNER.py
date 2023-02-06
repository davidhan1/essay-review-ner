import sys
sys.path.append('D:\Study\Python_project\essay-review-NER')

from utils.utils import *
from loss.focal_loss import FocalLoss

import numpy as np
import transformers
import torch
from torch.utils.tensorboard import SummaryWriter
import json
from collections import defaultdict
from sklearn.model_selection import train_test_split
from transformers import AutoModelForTokenClassification, Trainer, TrainingArguments, AutoTokenizer
from datasets import load_metric
import os
from tqdm import tqdm

def try_all(models, fp):
    class MyDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx])
                    for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    
    for model_typ in models:

        # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        max_len = 60
        train_data = read_json(os.path.join(DATA_PATH, 'test', 'train.json'))
        val_data = read_json(os.path.join(DATA_PATH, 'test', 'val.json'))
        x_train = [d['tokens'][:max_len] for d in train_data]
        y_train = [d['ner_tags'][:max_len] for d in train_data]
        x_val = [d['tokens'][:max_len] for d in val_data]
        y_val = [d['ner_tags'][:max_len] for d in val_data]

        tag2id = {
            'O': 0,
            'B-1': 1,
            'I-1': 2,
            'B-2': 3,
            'I-2': 4,
            'B-3': 5,
            'I-3': 6
        }
        id2tag = defaultdict(str)
        unique_tags = []
        for k in tag2id:
            id2tag[tag2id[k]] = k
            unique_tags.append(k)
        tokenizer = AutoTokenizer.from_pretrained(model_typ)
        train_encodings = tokenizer(x_train, is_split_into_words=True,
                                    return_offsets_mapping=True, padding=True, truncation=True)
        val_encodings = tokenizer(x_val, is_split_into_words=True,
                                return_offsets_mapping=True, padding=True, truncation=True)

        def encode_tags(tags, encodings):
            labels = [[tag2id[tag] for tag in doc] for doc in tags]
            print(encodings.offset_mapping)
            encoded_labels = []
            for doc_labels, doc_offset in tqdm(zip(labels, encodings.offset_mapping)):
                # create an empty array of -100
                doc_enc_labels = np.ones(len(doc_offset), dtype=int) * -100
                arr_offset = np.array(doc_offset)

                # set labels whose first offset position is 0 and the second is not 0
                doc_enc_labels[(arr_offset[:, 0] == 0) & (
                    arr_offset[:, 1] != 0)] = doc_labels
                encoded_labels.append(doc_enc_labels.tolist())

            return encoded_labels

        train_labels = encode_tags(y_train, train_encodings)
        val_labels = encode_tags(y_val, val_encodings)

        # we don't want to pass this to the model
        train_encodings.pop("offset_mapping")
        val_encodings.pop("offset_mapping")
        train_dataset = MyDataset(train_encodings, train_labels)
        val_dataset = MyDataset(val_encodings, val_labels)

        model = AutoModelForTokenClassification.from_pretrained(
            model_typ, num_labels=len(unique_tags))

        def compute_metrics(p):
            metric = load_metric("seqeval")
            predictions, labels = p
            # print(predictions.shape) # (32, 62, 7)
            predictions = np.argmax(predictions, axis=2)

            # Remove ignored index (special tokens)
            true_predictions = [
                [unique_tags[p] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]
            true_labels = [
                [unique_tags[l] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]

            results = metric.compute(
                predictions=true_predictions, references=true_labels)
            print(results)
            # return results
            return {
                # 'eval_C_f1': results['C']['f1'],
                # 'eval_E_f1': results['E']['f1'],
                # 'eval_D_f1': results['D']['f1'],
                "eval_overall_f1": results["overall_f1"],
                "eval_overall_accuracy": results["overall_accuracy"],
                "eval_overall_recall": results["overall_recall"],
                "eval_overall_precision": results["overall_precision"]
            }

        writer = SummaryWriter(log_dir=os.path.join(LOG_PATH, f'{model_typ}_fp' + str(fp) + '_logs'))
        callbacks = transformers.integrations.TensorBoardCallback(writer)

        class FocalLossTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False):
                labels = inputs.pop("labels").view(-1)
                outputs = model(**inputs)
                # print(outputs.logits.shape) # (4, 62, 7)
                logits = outputs.logits.view(-1, len(unique_tags))
                loss_fct = FocalLoss()
                loss = loss_fct(logits, labels)
                return (loss, outputs) if return_outputs else loss

        training_args = TrainingArguments(
            output_dir=os.path.join(CKPT_PATH, f'{model_typ}_fp' + str(fp) + '_ckpt'),             # output directory
            num_train_epochs=10,              # total number of training epochs
            per_device_train_batch_size=4,  # batch size per device during training
            per_device_eval_batch_size=4,   # batch size for evaluation
            # warmup_steps=500,                # number of warmup steps for learning rate scheduler
            weight_decay=0.01,               # strength of weight decay
            # directory for storing logs
            logging_dir=os.path.join(LOG_PATH, f'{model_typ}_fp' + str(fp) + '_logs'),
            evaluation_strategy='epoch',
            logging_strategy='epoch',
            save_strategy ='epoch',
            learning_rate=2e-5,
            fp16=True if fp == 16 else False,
        )

        trainer = FocalLossTrainer(
            # the instantiated 🤗 Transformers model to be trained
            model=model,
            args=training_args,                  # training arguments, defined above
            train_dataset=train_dataset,         # training dataset
            eval_dataset=val_dataset,            # evaluation dataset
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=[callbacks]
        )

        trainer.train()
        trainer.evaluate()

def main():
    models = ['bert-base-chinese']
    fps = [32]
    # models = ['hfl/chinese-macbert-base', 'hfl/chinese-roberta-wwm-ext', 'bert-base-chinese']
    # fps = [32, 16]
    for fp in fps:
        try_all(models, fp)

if __name__ == '__main__':
    main()