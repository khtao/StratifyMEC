from optparse import OptionParser

from torch.utils.data import DataLoader
import torch
from source.data import ImageDataset
from source.clip_model import ClipImageModel
from source.fusion_model import FusionModel
from torch import nn, optim
from tqdm import tqdm
import torchvision.transforms as tsf
from sklearn.metrics import *
import numpy as np


def test(model, clip, dataloader):
    model.eval()
    pat_labels = []
    pat_pred_scores = []
    pat_pred_labels = []
    with torch.no_grad():
        for ii, (imgs, label, img_paths, test_input) in enumerate(dataloader):
            input_id = test_input['input_ids'].squeeze(1).cuda()
            mask = test_input['attention_mask'].squeeze(1).cuda()
            imgs = torch.concatenate(imgs).cuda()
            image_feat = clip.encode_image_t(imgs)
            text_feat = clip.encode_text_t(input_id, mask)
            output = model(image_feat, text_feat)[2]
            pat_score = output.softmax(dim=-1)
            pat_score = 1 - pat_score[:, 0]
            pat_predicted = (pat_score > 0.5).long()
            pat_pred_scores += list(pat_score.cpu().numpy())
            pat_pred_labels += list(pat_predicted.cpu().numpy())
            try:
                pat_labels.append(label[0])
            except:
                pat_labels.append(label)

    label_names = ('low-risk', 'high-risk')
    print('patient metric')
    pat_accuracy = accuracy_score(pat_labels, pat_pred_labels)
    pat_precision, pat_recall, pat_fscore, _ = precision_recall_fscore_support(pat_labels, pat_pred_labels,
                                                                               average='binary')
    pat_roc_auc = roc_auc_score(pat_labels, np.stack(pat_pred_scores))
    print(confusion_matrix(pat_labels, pat_pred_labels))
    print(classification_report(pat_labels, pat_pred_labels, target_names=label_names, digits=4))
    return {"Recall": float(pat_recall),
            "Precision": float(pat_precision),
            "Accuracy": float(pat_accuracy),
            "F": float(pat_fscore),
            'AUC': pat_roc_auc}


def train(opt):
    balance_mode = True
    image_size = 350
    train_tsf = tsf.Compose([
        tsf.RandomCrop(image_size, pad_if_needed=True),
        tsf.RandomRotation(90),
        tsf.RandomVerticalFlip(),
        tsf.RandomHorizontalFlip(),
        tsf.ToTensor(),
    ])
    val_tsf = tsf.Compose([
        tsf.CenterCrop(image_size),
        tsf.RandomCrop(image_size, pad_if_needed=True),
        tsf.ToTensor(),
    ])
    train_data = ImageDataset(
        image_root=opt.train_image_root,
        data_path=opt.train_data_path,
        text_path=opt.train_text_path,
        balance=balance_mode,
        transform=train_tsf,
        patient_mode=True,
        transform_txt=True,
    )

    train_dataloader = DataLoader(train_data,
                                  batch_size=1,
                                  shuffle=True,
                                  num_workers=8,
                                  drop_last=True
                                  )

    val_data = ImageDataset(image_root=opt.val_image_root,
                            data_path=opt.val_data_path,
                            text_path=opt.val_text_path,
                            balance=False,
                            transform=val_tsf,
                            patient_mode=True,
                            transform_txt=False,

                            )

    val_dataloader = DataLoader(val_data,
                                batch_size=1,
                                shuffle=False,
                                num_workers=8,
                                drop_last=False
                                )
    clip_model = ClipImageModel().cuda()
    clip_model.load_state_dict(torch.load(opt.clip_model_path, weights_only=True))
    model = FusionModel(clip_model.embed_dim).cuda()
    optimizer = optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    loss_function = nn.CrossEntropyLoss()
    schedulers = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.epochs)
    best_auc = 0
    for epoch in range(opt.epochs):
        print('Starting epoch {}/{}'.format(epoch + 1, opt.epochs))
        model.train()
        batch = 0
        loss = []
        for i, (imgs, labels, img_paths, test_input) in tqdm(enumerate(train_dataloader)):
            input_id = test_input['input_ids'].squeeze(1).cuda()
            mask = test_input['attention_mask'].squeeze(1).cuda()
            imgs = torch.concatenate(imgs).cuda()
            labels = labels[0].cuda().long()
            with torch.no_grad():
                image_feat = clip_model.encode_image_t(imgs)
                text_feat = clip_model.encode_text_t(input_id, mask)
            output = model(image_feat, text_feat)
            loss.append(loss_function(output[0], labels)
                        + loss_function(output[1], labels) * 10
                        + loss_function(output[2], labels) * 100)
            batch += 1
            if batch > 4:
                optimizer.zero_grad()
                pat_loss = sum(loss) / len(loss)
                pat_loss.backward()
                optimizer.step()
                batch = 0
                loss = []
            schedulers.step()
        val_result = test(model, clip_model, val_dataloader)
        print(val_result)
        if best_auc < val_result['AUC']:
            best_auc = val_result['AUC']
            print('save best_AUC model', best_auc)
            torch.save(model.state_dict(), f'{opt.model_name}_best_AUC.pt')


def get_args():
    parser = OptionParser()
    parser.add_option('--model_name', dest='model_name', default='StratifyMEC_Fusion', help='model name')
    parser.add_option('--clip_model_path', dest='clip_model_path', default='StratifyMEC_CLIP_best_AUC.pt',
                      help='Pre-trained clip model')
    parser.add_option('--train_image_root', dest='train_image_root',
                      default='/mnt/Dataset/risk_dataset_unet/seg_dataset/images_checked/train_val')
    parser.add_option('--train_data_path', dest='train_data_path',
                      default='/mnt/Dataset/risk_dataset_unet/seg_dataset/images_checked/train_val/所见及结论_train-random-32.csv',)
    parser.add_option('--train_text_path', dest='train_text_path',
                      default='/mnt/Dataset/risk_dataset_unet/seg_dataset/images_checked/train_val/所见及结论_train-32_ori_new.xlsx',)

    parser.add_option('--val_image_root', dest='val_image_root',
                      default='/mnt/Dataset/risk_dataset_unet/seg_dataset/images_checked/train_val')
    parser.add_option('--val_data_path', dest='val_data_path',
                      default='/mnt/Dataset/risk_dataset_unet/seg_dataset/images_checked/train_val/所见及结论_val-random-32.csv',)
    parser.add_option('--val_text_path', dest='val_text_path',
                      default='/mnt/Dataset/risk_dataset_unet/seg_dataset/images_checked/train_val/所见及结论_val-32_ori_new.xlsx',)
    parser.add_option('--epochs', dest='epochs', default=20, type='int', help='number of epochs')
    parser.add_option('--lr', dest='lr', default=1e-6, type='float', help='learning rate')
    parser.add_option('--eta_min', dest='eta_min', default=1e-8, type='float', help='eta min')
    parser.add_option('--weight_decay', dest='weight_decay', default=1e-5, type='float', help='weight decay')
    (options, args) = parser.parse_args()
    return options


if __name__ == '__main__':
    args = get_args()
    train(args)
