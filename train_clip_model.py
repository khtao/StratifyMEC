import os
from optparse import OptionParser

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from torch.utils.data import DataLoader
import torch
from source.data import ImageDataset
from source.clip_model import ClipImageModel
from torch import nn, optim
from tqdm import tqdm
import torchvision.transforms as tsf
from torch.optim import lr_scheduler
from sklearn.metrics import *
import random
import numpy as np


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def cal_metrics(outputs, labels):
    pred_scores = outputs.softmax(dim=-1)
    pred_scores = 1 - pred_scores[:, 0]
    pred_labels = (pred_scores > 0.5).long()
    pred_scores = pred_scores.cpu().numpy()
    pred_labels = pred_labels.cpu().numpy()
    labels = labels.cpu().numpy()

    accuracy = accuracy_score(labels, pred_labels)
    precision, recall, fscore, _ = precision_recall_fscore_support(labels, pred_labels, average='binary')
    roc_auc = roc_auc_score(labels, pred_scores)
    print(confusion_matrix(labels, pred_labels))
    label_names = ('low-risk', 'high-risk')
    print(classification_report(labels, pred_labels, target_names=label_names, digits=4))
    return {"Recall": round(float(recall), 4),
            "Precision": round(float(precision), 4),
            "Accuracy": round(float(accuracy), 4),
            "F": round(float(fscore), 4),
            'AUC': round(float(roc_auc), 4)
            }


def test(model, dataloader):
    model.eval()
    gt_labels = []
    im_outs = []
    with torch.no_grad():
        for imgs, labels, test_input in dataloader:
            imgs = imgs.cuda()
            labels = labels.long()
            im_cls = model.predict(imgs)
            im_outs.append(im_cls)
            gt_labels.append(labels)
    im_res = cal_metrics(torch.concatenate(im_outs, dim=0), torch.concatenate(gt_labels, dim=0))
    return im_res


def train(opt):
    train_tsf = tsf.Compose([
        tsf.RandomCrop(opt.image_size, pad_if_needed=True),
        tsf.RandomRotation(90),
        tsf.RandomVerticalFlip(),
        tsf.RandomHorizontalFlip(),
        tsf.ToTensor(),
    ])
    val_tsf = tsf.Compose([
        tsf.CenterCrop(opt.image_size),
        tsf.RandomCrop(opt.image_size, pad_if_needed=True),
        tsf.ToTensor(),
    ])
    balance_mode = True
    train_data = ImageDataset(
        image_root=opt.train_image_root,
        data_path=opt.train_data_path,
        text_path=opt.train_text_path,
        balance=balance_mode,
        transform=train_tsf,
        transform_txt=True,
    )

    train_dataloader = DataLoader(train_data,
                                  batch_size=opt.batch_size,
                                  shuffle=True,
                                  num_workers=8,
                                  drop_last=True
                                  )

    val_data = ImageDataset(image_root=opt.val_image_root,
                            data_path=opt.val_data_path,
                            text_path=opt.val_text_path,
                            balance=False,
                            transform=val_tsf,
                            transform_txt=False,
                            )

    val_dataloader = DataLoader(val_data,
                                batch_size=opt.batch_size,
                                shuffle=False,
                                num_workers=8,
                                drop_last=False
                                )

    model = ClipImageModel(opt.pretrained_bert_path).cuda()
    loss_func = nn.CrossEntropyLoss().cuda()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr, weight_decay=1e-5)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, opt.epochs, eta_min=1e-8)
    best_metrics = 0
    for num_epochs in range(opt.epochs):
        
        loop = tqdm(train_dataloader, total=len(train_dataloader))
        for imgs, labels, test_input in loop:
            input_id = test_input['input_ids'].squeeze(1).cuda()
            mask = test_input['attention_mask'].squeeze(1).cuda()
            imgs = imgs.cuda()
            labels = labels.cuda().long()
            outs = model(imgs, input_id, mask)
            ground_truth = torch.arange(opt.batch_size, dtype=torch.long).cuda()
            clip_loss = (loss_func(outs[0], ground_truth) + loss_func(outs[1], ground_truth)) / 2
            loss_cls_img = loss_func(outs[2], labels)
            # 反向传播
            total_loss = clip_loss * 2.0 + loss_cls_img
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            loop.set_description(f'Epoch [{num_epochs}/{opt.epochs}]')
        scheduler.step()
        val_result = test(model, val_dataloader)
        print(val_result)
        if val_result['AUC'] > best_metrics:
            best_metrics = val_result['AUC']
            print('save best_AUC model', best_metrics)
            torch.save(model.state_dict(), f'{opt.model_name}_best_AUC.pt')


def get_args():
    parser = OptionParser()
    parser.add_option('--model_name', dest='model_name', default='StratifyMEC_CLIP', help='model name')
    parser.add_option('--pretrained_bert_path', dest='pretrained_bert_path', default='StratifyMEC_BERT_best_AUC.pt',
                      help='Pre-trained bert model')
    parser.add_option('--train_image_root', dest='train_image_root',
                      default='/mnt/Dataset/risk_dataset_unet/seg_dataset/images_checked/train_val')
    parser.add_option('--train_data_path', dest='train_data_path',
                      default='/mnt/Dataset/risk_dataset_unet/seg_dataset/images_checked/train_val/所见及结论_train-random-32.csv', )
    parser.add_option('--train_text_path', dest='train_text_path',
                      default='/mnt/Dataset/risk_dataset_unet/seg_dataset/images_checked/train_val/所见及结论_train-32_ori_new.xlsx', )

    parser.add_option('--val_image_root', dest='val_image_root',
                      default='/mnt/Dataset/risk_dataset_unet/seg_dataset/images_checked/train_val')
    parser.add_option('--val_data_path', dest='val_data_path',
                      default='/mnt/Dataset/risk_dataset_unet/seg_dataset/images_checked/train_val/所见及结论_val-random-32.csv', )
    parser.add_option('--val_text_path', dest='val_text_path',
                      default='/mnt/Dataset/risk_dataset_unet/seg_dataset/images_checked/train_val/所见及结论_val-32_ori_new.xlsx', )
    parser.add_option('--epochs', dest='epochs', default=100, type='int', help='number of epochs')
    parser.add_option('--lr', dest='lr', default=1e-5, type='float', help='learning rate')
    parser.add_option('--batch_size', dest='batch_size', default=32, type='int', help='batch size')
    parser.add_option('--image_size', dest='image_size', default=384, type='int', help='batch size')
    parser.add_option('--eta_min', dest='eta_min', default=1e-8, type='float', help='eta min')
    parser.add_option('--weight_decay', dest='weight_decay', default=1e-5, type='float', help='weight decay')
    (options, args) = parser.parse_args()
    return options


if __name__ == '__main__':
    args = get_args()
    train(args)
