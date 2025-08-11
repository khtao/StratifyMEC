import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet50
from source.bert_model import BertClassifier


class ClipImageModel(nn.Module):
    def __init__(self,
                 pretrained_bert: str = None,
                 embed_dim: int = 768,
                 n_classes: int = 2, ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_classes = n_classes
        self.visual = resnet50(weights=True)
        self.visual.fc = nn.Linear(self.visual.fc.in_features, embed_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        bert = BertClassifier()
        if pretrained_bert is not None:
            bert.load_state_dict(torch.load(pretrained_bert, weights_only=True))
        self.bert = bert.bert
        self.im_cls = nn.Linear(embed_dim, self.n_classes)

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    @property
    def device(self):
        return self.visual.conv1.weight.device

    def encode_image(self, image):
        return self.visual(image)

    def predict(self, image):
        return self.im_cls(self.visual(image))

    def encode_text(self, text, mask):
        x = self.bert(input_ids=text, attention_mask=mask, return_dict=False)[1]
        return x

    def encode_image_t(self, x):
        x = self.visual.conv1(x)
        x = self.visual.bn1(x)
        x = self.visual.relu(x)
        x = self.visual.maxpool(x)

        x = self.visual.layer1(x)
        x = self.visual.layer2(x)
        x = self.visual.layer3(x)
        x = self.visual.layer4(x)
        return x

    def encode_text_t(self, text, mask):
        x = self.bert(input_ids=text, attention_mask=mask, return_dict=False)[0]
        return x

    def forward(self, image, text, mask):
        self.bert.eval()
        im_features = self.encode_image(image)
        with torch.no_grad():
            txt_features = self.encode_text(text, mask)
        image_out = self.im_cls(im_features)
        im_norm_features = im_features / im_features.norm(dim=1, keepdim=True)
        txt_norm_features = txt_features / txt_features.norm(dim=1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * im_norm_features @ txt_norm_features.t()
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text, image_out
