# -*- coding: utf-8 -*-
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
os.environ["HTTPS_PROXY"] = '127.0.0.1:9999'
import torch
from torch import nn
from torch.optim import NAdam, lr_scheduler
from tqdm import tqdm
import numpy as np
from optparse import OptionParser
import random
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support, classification_report
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from source.bert_model import BertClassifier, BertDataset


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def save_model(save_path, model):
    os.makedirs(os.path.split(save_path)[0], exist_ok=True)
    torch.save(model.state_dict(), save_path)


def test(model, dev_loader):
    model.eval()
    labels = []
    pred_scores = []
    pred_scores_b = []
    pred_labels = []
    with torch.no_grad():
        for test_input, test_label, all_paths in dev_loader:
            labels += list(test_label.numpy())
            input_id = test_input['input_ids'].squeeze(1).cuda()
            mask = test_input['attention_mask'].squeeze(1).cuda()
            output = model(input_id, mask)
            score = output.softmax(dim=-1)
            predicted = torch.argmax(score, dim=-1)
            pred_scores += list(score.cpu().numpy())
            pred_labels += list(predicted.cpu().numpy())
            pred_scores_b += list(score.cpu().numpy()[:, 1])
    precision, recall, fscore, _ = precision_recall_fscore_support(labels, pred_labels, average='binary')
    roc_auc = roc_auc_score(labels, pred_scores_b)
    label_names = ('low-risk', 'high-risk')
    accuracy = accuracy_score(labels, pred_labels)
    print(confusion_matrix(labels, pred_labels))
    print(classification_report(labels, pred_labels, target_names=label_names, digits=4))
    result = {"Recall": float(recall),
              "Precision": float(precision),
              "Accuracy": float(accuracy),
              "F": float(fscore),
              'AUC': float(roc_auc)
              }
    return result


def train(opt):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    setup_seed(opt.random_seed)
    train_dataset = BertDataset(opt.train_data_path, True,  mode='train')
    val_dataset = BertDataset(opt.val_data_path, False,  mode='val')
    # test_dataset2 = GenerateData(mode='test2', is_binary=is_binary)
    train_loader = DataLoader(train_dataset, batch_size=opt.batchsize, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=opt.batchsize, shuffle=False, num_workers=4)

    model = BertClassifier(dropout_p=opt.dropout).cuda()
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = NAdam(model.parameters(), lr=opt.lr, weight_decay=1e-5)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, opt.epochs, eta_min=1e-8)
    model = model.to(device)
    criterion = criterion.to(device)

    best_auc = 0
    for epoch_num in range(opt.epochs):
        model.train()
        for inputs, labels, all_paths in tqdm(train_loader):
            input_ids = inputs['input_ids'].squeeze(1).to(device)
            masks = inputs['attention_mask'].squeeze(1).to(device)
            labels = labels.to(device)
            output = model(input_ids, masks)
            batch_loss = criterion(output, labels)
            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        scheduler.step()
        val_result = test(model, val_loader)
        print(val_result)
        if best_auc < val_result['AUC']:
            best_auc = val_result['AUC']
            print('save best_AUC model', best_auc)
            torch.save(model.state_dict(), f'{opt.model_name}_best_AUC.pt')


def get_args():
    parser = OptionParser()
    parser.add_option('--model_name', dest='model_name', default='StratifyMEC_BERT', help='model name')
    parser.add_option('--train_data_path', dest='train_data_path',
                      default='/mnt/Dataset/risk_dataset_unet/seg_dataset/images_checked/train_val/所见及结论_train-32_ori_new.xlsx', help='model name')
    parser.add_option('--val_data_path', dest='val_data_path',
                      default='/mnt/Dataset/risk_dataset_unet/seg_dataset/images_checked/train_val/所见及结论_val-32_ori_new.xlsx', help='model name')
    parser.add_option( '--epochs', dest='epochs', default=100, type='int', help='number of epochs')
    parser.add_option( '--dropout', dest='dropout', default=0.85, type='float')
    parser.add_option( '--batchsize', dest='batchsize', default=24, type='int', help='batch size')
    parser.add_option('--lr', dest='lr', default=1e-5, type='float', help='learning rate')
    parser.add_option('--eta_min', dest='eta_min', default=1e-8, type='float', help='eta min')
    parser.add_option('--weight_decay', dest='weight_decay', default=1e-5, type='float', help='weight decay')
    parser.add_option('--random_seed', dest='random_seed', type='int', default=3407)
    (options, args) = parser.parse_args()
    return options

if __name__ == '__main__':
    args = get_args()
    train(args)
