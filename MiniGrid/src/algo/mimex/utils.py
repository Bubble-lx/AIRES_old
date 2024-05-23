
import os
import torch
import torch.nn as nn
from src.algo.mimex.mvp.backbones import vit


_HOI_MODELS = {
    "maevit-s16": "mae_pretrain_hoi_vit_small.pth",
}
_IN_MODELS = {
    "vit-s16": "sup_pretrain_imagenet_vit_small.pth",
    "maevit-s16": "mae_pretrain_imagenet_vit_small.pth",
}
pretrain_dir = "/home/liuxinn/RLCLAL/self/learn_from_md/deir-main/src/algo/intrinsic_rewards/pretrain"
# pretrain_dir = "/deir/src/algo/intrinsic_rewards/pretrain"
pretrain_fname = "mae_pretrain_hoi_vit_small.pth"
pretrain_type = 'hoi'
model_type = 'maevit-s16'
freeze = True
emb_dim = 128 # 没用到
# ----------------------------------------------------------------
# bert_ratio
# input_dim = 147
input_dim = 64
decoder_embed_dim = 32
mask_ratio = 0.7
# ----------------------------------------------------------------
# MIMEx setings
input_dim_mimex = 64
decoder_embed_dim_mimex = 32
anneal_weight = 1.0

#  ---------------------------------------------------------------

# class Encoder(nn.Module):
#     def __init__(self):
#         super(Encoder, self).__init__()
#         assert pretrain_type in ["imagenet", "hoi", "none"]
#         if pretrain_type == "imagenet":
#             assert model_type in _IN_MODELS
#             pretrain_fname = _IN_MODELS[model_type]
#             pretrain_path = os.path.join(pretrain_dir, pretrain_fname)
#         elif pretrain_type == "hoi":
#             assert model_type in _HOI_MODELS
#             pretrain_fname = _HOI_MODELS[model_type]
#             pretrain_path = os.path.join(pretrain_dir, pretrain_fname)
#         else:
#             pretrain_path = "none"
#         assert pretrain_type == "none" or os.path.exists(pretrain_path)
#         pretrain_path = os.path.join(pretrain_dir, pretrain_fname)
#         self.backbone, gap_dim = vit.vit_s16(pretrain_path)
#         if freeze:
#             self.backbone.freeze()
#         self.freeze = freeze
#         self.projector = nn.Linear(gap_dim, emb_dim)

#     @torch.no_grad()
#     def forward(self, x):
#         feat = self.backbone.extract_feat(x)
#         return self.projector(self.backbone.forward_norm(feat)), feat

#     def forward_feat(self, feat):
#         return self.projector(self.backbone.forward_norm(feat))