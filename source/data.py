import os
import random
import re

import numpy as np
import pandas as pd
import tifffile
from PIL import Image
from torch.utils.data import dataset
from transformers import BertTokenizer


class ImageDataset(dataset.Dataset):
    def __init__(self, image_root, data_path, transform=None, balance=False,
                 transform_txt=False,
                 patient_mode=False,  text_path=None):
        self.image_root = image_root
        self.data_path = data_path
        self.data_list = open(data_path, encoding='utf-8').readlines()
        pretrained_model = 'trueto/medbert-base-chinese'
        self.transform_txt = transform_txt

        self.text_path = text_path
        if text_path is not None:
            self.text_dict = self.read_txt()
            self.tokenizer = BertTokenizer.from_pretrained(pretrained_model)
        self.balance = balance
        self.transform = transform
        self.patient_mode = patient_mode
        self.label_names = ('low-risk', 'intermediate-risk',
                            'intermediate-high-risk', 'high-risk', 'advanced-metastasis')
        self.label_names = ('low-risk', 'high-risk')
        for i in range(len(self.data_list)):
            img_path, label, cc = self.data_list[i].split(', ')
            if int(label) >= 2:
                self.data_list[i] = img_path + ', ' + str(1) + ', ' + str(cc) + '\n'

        if patient_mode:
            self.data_list = self.make_patient()
        print(self.statistic())
        if balance:
            self.data_list = self.balance_data()
            print(self.statistic())

    def read_txt(self):
        data = pd.read_excel(self.text_path)
        data_dict = {}
        for pp, txt in zip(data['path'], data['text']):
            data_dict[pp] = txt
        return data_dict

    @staticmethod
    def tsf_txt(txt):
        # rand del
        del_p = 0.25
        text_list = re.split(r'[，。,.]', txt)
        # text_list = re.split('。', txt)
        temp_txt = ''
        for dd in text_list:
            p = random.uniform(0, 1)
            if p > del_p:
                temp_txt += dd
                temp_txt += '。'
        txt = temp_txt

        del_p = 0.05
        # text_list = re.split(r'[，。,.]', txt)
        # text_list = re.split('。', txt)
        temp_txt = ''
        for dd in txt:
            p = random.uniform(0, 1)
            if p > del_p:
                temp_txt += dd
        txt = temp_txt

        return txt

    def make_patient(self):
        patient_dict = {}
        for kk in self.data_list:
            pat = kk.split(', ')[0].split('/')[1]
            if pat in patient_dict.keys():
                patient_dict[pat].append(kk)
            else:
                patient_dict[pat] = [kk]
        patient_list = [data for key, data in patient_dict.items()]
        return patient_list

    def balance_data(self):
        class_dict = {}
        for kk in self.data_list:
            if self.patient_mode:
                label = kk[0].split(', ')[1]
            else:
                label = kk.split(', ')[1]
            if label in class_dict.keys():
                class_dict[label].append(kk)
            else:
                class_dict[label] = [kk]
        num_class = []
        for key in class_dict.keys():
            random.shuffle(class_dict[key])
            num_class.append(len(class_dict[key]))
        num = max(num_class)
        all_list = []
        for key in class_dict.keys():
            all_list += (class_dict[key] * (num // len(class_dict[key]) + 1))[:num]
        return all_list

    def statistic(self):
        allclasses = {}
        for dict_line in self.data_list:
            if self.patient_mode:
                img_path, label, cc = dict_line[0].split(', ')
            else:
                img_path, label, cc = dict_line.split(', ')
            label_i = int(label)
            cls = self.label_names[label_i]
            cls = cls.lower()
            allclasses[cls] = 1 if cls not in allclasses.keys() else allclasses[cls] + 1

        return allclasses

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        if self.patient_mode:
            images = []
            labels = []
            img_paths = []
            for dd in self.data_list[item]:
                img_path, label, cc = dd.split(', ')
                if img_path[-3:] == 'tif':
                    image = tifffile.imread(os.path.join(self.image_root, img_path)).astype(np.float32)
                else:
                    image = Image.open(os.path.join(self.image_root, img_path)).convert('RGB')
                # image = (image - image.min()) / (image.max() - image.min())
                if self.transform is not None:
                    image = self.transform(image)
                images.append(image)
                labels.append(int(label))
                img_paths.append(img_path)
            if self.text_path is not None:
                key = os.path.split(img_paths[0])[0]
                if key in self.text_dict.keys():
                    txt = self.text_dict[key]
                else:
                    txt = '内容缺少'
                if self.transform_txt:
                    txt = self.tsf_txt(txt)
                txt = self.tokenizer(txt,
                                     padding='max_length',
                                     max_length=512,
                                     truncation=True,
                                     return_tensors="pt")
                return images, labels, img_paths, txt
            return images, labels, img_paths
        else:
            img_path, label, cc = self.data_list[item].split(', ')
            if img_path[-3:] == 'tif':
                image = tifffile.imread(os.path.join(self.image_root, img_path)).astype(np.float32)
            else:
                image = Image.open(os.path.join(self.image_root, img_path)).convert('RGB')
            # image = (image - image.min()) / (image.max() - image.min())
            if self.transform is not None:
                image = self.transform(image)

            if self.text_path is not None:
                key = os.path.split(img_path)[0]
                if key in self.text_dict.keys():
                    txt = self.text_dict[key]
                else:
                    txt = '内容缺少'
                if self.transform_txt:
                    txt = self.tsf_txt(txt)
                token = self.tokenizer(txt,
                                       padding='max_length',
                                       max_length=512,
                                       truncation=True,
                                       return_tensors="pt")
                return image, int(label), token
            return image, int(label)