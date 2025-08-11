# -*- coding: utf-8 -*-
import random
import re

import pandas as pd
from torch import nn
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertModel

pretrained_model = 'trueto/medbert-base-chinese'
tokenizer = BertTokenizer.from_pretrained(pretrained_model)

class BertDataset(Dataset):
    def __init__(self, data_path, balance=False, mode='val'):
        df = pd.read_excel(data_path)
        self.data_dict = {}
        for kk in ['text', 'label', 'path', 'center']:
            self.data_dict[kk] = list(df[kk])
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model)
        self.mode = mode
        for n in range(len(self.data_dict['label'])):
            self.data_dict['label'][n] = min(self.data_dict['label'][n], 1)
        self.balance = balance
        self.label_names = ('low-risk', 'high-risk')
        print(self.statistic())
        if self.balance:
            self.data_dict = self.balance_data()
            print(self.statistic())

    def balance_data(self):
        class_dict = {}
        for n in range(len(self.data_dict['label'])):
            my_label = self.label_names[self.data_dict['label'][n]]
            if my_label in class_dict.keys():
                class_dict[my_label].append([self.data_dict['text'][n],
                                             self.data_dict['label'][n],
                                             self.data_dict['path'][n],
                                             self.data_dict['center'][n]])
            else:
                class_dict[my_label] = [[self.data_dict['text'][n],
                                         self.data_dict['label'][n],
                                         self.data_dict['path'][n],
                                         self.data_dict['center'][n]]]
        num_class = []
        for key in class_dict.keys():
            random.shuffle(class_dict[key])
            num_class.append(len(class_dict[key]))
        num = max(num_class)
        all_list = []
        for key in class_dict.keys():
            all_list += (class_dict[key] * (num // len(class_dict[key]) + 1))[:num]
        data_dict = {}
        for n, kk in enumerate(['text', 'label', 'path', 'center']):
            dd_list = []
            for dd in all_list:
                dd_list.append(dd[n])
            data_dict[kk] = dd_list.copy()
        return data_dict

    def statistic(self):
        allclasses = {}
        for lab in self.data_dict['label']:
            cls = self.label_names[lab]
            cls = cls.lower()
            allclasses[cls] = 1 if cls not in allclasses.keys() else allclasses[cls] + 1
        return allclasses

    @staticmethod
    def transfrom(txt):
        # rand del
        del_p = 0.5
        text_list = re.split(r'[，。,.]', txt)
        temp_txt = ''
        for dd in text_list:
            p = random.uniform(0, 1)
            if p > del_p:
                temp_txt += dd
                temp_txt += '。'
        txt = temp_txt
        del_p = 0.05
        temp_txt = ''
        for dd in txt:
            p = random.uniform(0, 1)
            if p > del_p:
                temp_txt += dd
        txt = temp_txt
        return txt

    def __getitem__(self, idx):
        text = self.data_dict['text'][idx]
        path = self.data_dict['path'][idx]
        if self.mode == 'train':
            text = self.transfrom(text)

        text = self.tokenizer(text,
                              padding='max_length',
                              max_length=512,
                              truncation=True,
                              return_tensors="pt")
        return text, self.data_dict['label'][idx], path

    def __len__(self):
        return len(self.data_dict['label'])


class BertClassifier(nn.Module):
    def __init__(self, dropout_p=0.5):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.dropout = nn.Dropout(dropout_p)
        self.linear = nn.Linear(768, 4, bias=True)

    def forward(self, input_id, mask):
        out = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)[1]
        out = self.dropout(out)
        out = self.linear(out)
        return out