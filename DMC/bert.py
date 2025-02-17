from functools import partial

import torch
import torch.nn as nn
import math

from timm.models.vision_transformer import Block

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()

        # compute the positional encodings once in log space
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe.require_grad = False
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        return self.pe[:, :x.size(1)]


class BERT(nn.Module):
    """
    MAE-like BERT.
    """

    def __init__(self, seq_len, feature_dim, embed_dim=128, depth=4,
        num_heads=4, decoder_embed_dim=64, decoder_num_heads=2, decoder_depth=1,
        mlp_ratio=4., norm_layer=partial(nn.LayerNorm, eps=1e-6), dropout=0.,
        mask_ratio=0.7, norm_loss=False, use_cls=False):
        super().__init__()

        self.mask_ratio = mask_ratio
        self.norm_loss = norm_loss # 是否在损失计算中使用标准化
        self.use_cls = use_cls # 是否在序列前加入CLS（分类）标记，通常用于分类任务。

        # --------------------------------------------------------------------------
        # BERT encoder specifics

        self.encoder_embed = nn.Linear(feature_dim, embed_dim)

        if self.use_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # sum of features and positional embeddings
        self.pos_embed = PositionalEncoding(
            d_model=embed_dim, max_len=seq_len)

        # mlp_ratio：Transformer块中前馈网络的隐藏层大小与嵌入维度的比率。
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # BERT decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim)) # 掩码标记，用于在解码过程中替换被掩码的元素。

        self.decoder_pos_embed = PositionalEncoding(
            d_model=decoder_embed_dim, max_len=seq_len)

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, feature_dim, bias=True) # 将解码器的输出映射回原始特征空间的线性层。
        # --------------------------------------------------------------------------
        self.norm_loss = norm_loss

        self.initialize_weights()

    def initialize_weights(self):
        torch.nn.init.normal_(self.mask_token, std=.02)

        if self.use_cls:
            torch.nn.init.normal_(self.cls_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def random_masking(self, x):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch 1024, length 5, dim 128
        len_keep = int(L * (1 - self.mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1] ，输出[1024, 5]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x):
        # (B, L, feature_dim) -> (B, L, embed_dim)
        x = self.encoder_embed(x) # 输出[1024, 5, 128]

        # add pos embed
        x = x + self.pos_embed(x) # 输出[1024, 5, 128]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x) # 输出x为[1024, 0, 128], mask为[1024, 5], ids_restore为[1024, 5]

        if self.use_cls:
            # append cls token
            cls_token = self.cls_token  # pos emb can be ignored since it's 0 [1, 1, 128]
            cls_tokens = cls_token.expand(x.shape[0], -1, -1) # [1024, 1, 128]
            x = torch.cat((cls_tokens, x), dim=1) # 输出x为[1024, 2, 128]

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        if self.use_cls:
            # append mask tokens to sequence
            mask_tokens = self.mask_token.repeat(
                x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
            x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
            x_ = torch.gather(
                x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
            x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

            # add pos embed to non-cls-tokens
            x[:, 1:, :] = x[:, 1:, :] + self.decoder_pos_embed(x[:, 1:, :])
        else:
            # append mask tokens to sequence
            mask_tokens = self.mask_token.repeat(
                x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
            x_ = torch.cat([x, mask_tokens], dim=1)
            x = torch.gather(
                x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

            x = x + self.decoder_pos_embed(x)

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        if self.use_cls:
            # remove cls token
            x = x[:, 1:, :]

        return x

    def forward_loss(self, x, pred, mask, keep_batch=False):
        """
        x: [B, L, D]
        pred: [B, L, D]
        mask: [B, L], 0 is keep, 1 is remove
        """
        if self.norm_loss:
            # normalize loss
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, keepdim=True)
            x = (x - mean) / (var + 1.e-6)**.5

        loss = (pred - x) ** 2
        loss = loss.mean(dim=-1)  # [B, L], mean loss per timestep 此处是需要变化的loss值。
        loss_reward = loss
        # mean loss on removed timesteps
        if keep_batch:
            loss = (loss * mask).sum(dim=-1) / mask.sum(dim=-1)  # (N,)
        else:
            loss = (loss * mask).sum() / mask.sum()
        return loss, loss_reward

    def forward(self, x, keep_batch=False):
        '''
        Args:
            x: [B, L, D]
        Returns:
            loss
        '''
        latent, mask, ids_restore = self.forward_encoder(x) # [1024, 5, 39200]
        pred = self.forward_decoder(latent, ids_restore)  # [B, L, D]
        loss, loss_reward = self.forward_loss(x, pred, mask, keep_batch)
        return loss, pred, mask, loss_reward
