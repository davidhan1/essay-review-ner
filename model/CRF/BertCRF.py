import torch
import torch.nn as nn
from CRF_layer import CRF
from transformers import BertModel,BertPreTrainedModel

class BertCrfForNer(BertPreTrainedModel):
    def __init__(self, config):
        super(BertCrfForNer, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout()
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,labels=None):
        outputs =self.bert(input_ids = input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        prediction = self.crf.decode(emissions = logits, mask=attention_mask, nbest=5)[0]
        if self.training:
        # if labels is not None:
            loss = self.crf(emissions = logits, tags=labels, mask=attention_mask)
            outputs =(-1*loss,)+(prediction,)
        else:
            outputs = (prediction,)
        return outputs 