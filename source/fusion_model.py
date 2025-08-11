import torch
import torch.nn as nn
from source.rrt import RRTMILFeat


class BidirectionalCrossAttention(nn.Module):
    def __init__(self,
                 text_dim=768,  # BERT-base特征维度
                 image_dim=2048,  # ResNet50最后一层特征维度
                 embed_dim=512,  # 统一嵌入维度
                 num_heads=8,
                 dropout=0.0):
        super().__init__()

        # 文本特征处理分支
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        # 图像特征处理分支
        self.image_proj = nn.Sequential(
            nn.Linear(image_dim, embed_dim),
            nn.LayerNorm(embed_dim))

        # 双向交叉注意力模块
        self.text_to_image_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True)

        self.image_to_text_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True)

        # 特征融合层
        self.fusion = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.GELU(),
        )

        # 归一化层
        self.norm_t = nn.LayerNorm(embed_dim)
        self.norm_i = nn.LayerNorm(embed_dim)

        # 初始化参数
        self._init_weights()

    def _init_weights(self):
        for module in [self.text_proj, self.image_proj, self.fusion]:
            if hasattr(module, 'weight') and isinstance(module.weight, torch.Tensor):
                nn.init.xavier_normal_(module.weight)
            if hasattr(module, 'bias') and isinstance(module.bias, torch.Tensor):
                nn.init.zeros_(module.bias)

    def forward(self, text_features, image_features):
        """
        输入:
            text_features: BERT输出 [batch_size, seq_len, 768]
            image_features: ResNet50特征 [batch_size, channels, height, width]
                            需要转换为序列形式 [batch_size, h*w, 2048]
        """
        # 处理图像特征：展平空间维度
        batch_size = image_features.size(0)
        image_seq = image_features.flatten(2).permute(0, 2, 1)  # [B, h*w, 2048]

        # 投影到统一嵌入空间
        text_proj = self.text_proj(text_features)  # [B, L_t, D]
        image_proj = self.image_proj(image_seq)  # [B, L_i, D]

        # 文本->图像交叉注意力
        text_enhanced, _ = self.text_to_image_attn(
            query=text_proj,
            key=image_proj,
            value=image_proj
        )
        text_enhanced = self.norm_t(text_proj + text_enhanced)

        # 图像->文本交叉注意力
        image_enhanced, _ = self.image_to_text_attn(
            query=image_proj,
            key=text_proj,
            value=text_proj
        )
        image_enhanced = self.norm_i(image_proj + image_enhanced)

        # 双向特征融合
        combined = torch.cat([text_enhanced[:, 0, :], image_enhanced[:, 0, :]], dim=-1)
        fused_features = self.fusion(combined)

        return fused_features


class FusionModel(nn.Module):
    def __init__(self, embed_dim: int, out_dim: int = 2):
        super().__init__()
        self.embed_dim = embed_dim
        self.out_dim = out_dim
        self.early_fusion = BidirectionalCrossAttention(embed_dim=self.embed_dim, num_heads=8, dropout=0.0)
        self.image_fusion = RRTMILFeat(input_dim=self.embed_dim, mlp_dim=self.embed_dim, epeg_k=15, crmsa_k=3)
        self.im_cls = nn.Linear(self.embed_dim, self.out_dim)
        self.text_fusion = nn.Sequential(
            nn.Linear(self.embed_dim + 768, self.embed_dim, bias=True),
            nn.ReLU(True),
            nn.Linear(self.embed_dim, self.out_dim, bias=True),
        )
        self.mm_fusion = nn.Sequential(
            nn.Linear(self.out_dim * 2, self.embed_dim, bias=True),
            nn.ReLU(True),
            nn.Linear(self.embed_dim, self.out_dim, bias=True),
        )

    def forward(self, image_feat, text_feat):
        early_feat = self.early_fusion(text_feat.repeat(image_feat.size(0), 1, 1), image_feat)
        im_x = self.image_fusion(early_feat)[0]
        im_out = self.im_cls(im_x)
        txt_out = self.text_fusion(torch.concatenate([im_x, text_feat[:, 0, :]], dim=1))
        fusion_out = self.mm_fusion(torch.concatenate([im_out, txt_out], dim=1))
        return im_out, txt_out, fusion_out


if __name__ == '__main__':
    model = FusionModel(768)
    image_feat = torch.rand(32, 2048, 11, 11)
    bert_feat = torch.rand(1, 512, 768)
    kk = model(image_feat, bert_feat)
    print(kk)
