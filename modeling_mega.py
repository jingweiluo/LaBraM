# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# By Wei-Bang Jiang
# Based on BEiT-v2, timm, DeiT, and DINO code bases
# https://github.com/microsoft/unilm/tree/master/beitv2
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# ---------------------------------------------------------

import math
import torch
import torch.nn as nn
from functools import partial

from modeling_finetune import Block, _cfg, PatchEmbed
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
from einops import rearrange
import numpy as np
import torch.nn.functional as F


def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


class TemporalConv(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, in_chans=1, out_chans=8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chans, out_chans, kernel_size=(1, 15), stride=(1, 8), padding=(0, 7))
        self.gelu1 = nn.GELU()
        self.norm1 = nn.GroupNorm(4, out_chans)
        self.conv2 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 3), padding=(0, 1))
        self.gelu2 = nn.GELU()
        self.norm2 = nn.GroupNorm(4, out_chans)
        self.conv3 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 3), padding=(0, 1))
        self.norm3 = nn.GroupNorm(4, out_chans)
        self.gelu3 = nn.GELU()

    def forward(self, x, **kwargs):
        x = rearrange(x, 'B N A T -> B (N A) T')
        B, NA, T = x.shape
        x = x.unsqueeze(1)
        x = self.gelu1(self.norm1(self.conv1(x)))
        x = self.gelu2(self.norm2(self.conv2(x)))
        x = self.gelu3(self.norm3(self.conv3(x)))
        x = rearrange(x, 'B C NA T -> B NA (T C)')
        return x


class NeuralTransformerForMaskedEEGModeling(nn.Module):
    def __init__(self, EEG_size=1600, patch_size=200, in_chans=1, out_chans=8, vocab_size=8192, embed_dim=200, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_norm=None, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6), init_values=None, attn_head_dim=None,
                 use_abs_pos_emb=True, use_rel_pos_bias=False, use_shared_rel_pos_bias=False, init_std=0.02):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed = TemporalConv(out_chans=out_chans)
        self.num_heads = num_heads
        self.patch_size = patch_size

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if use_abs_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, 128 + 1, embed_dim))
        else:
            self.pos_embed = None
        self.time_embed = nn.Parameter(torch.zeros(1, 16, embed_dim), requires_grad=True)
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.rel_pos_bias = None

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_norm=qk_norm, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, window_size=self.patch_embed.patch_shape if use_rel_pos_bias else None,
                attn_head_dim=attn_head_dim,
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.init_std = init_std
        self.lm_head = nn.Linear(embed_dim, vocab_size)
    


        # ================================================================================================
        # ===================添加decoder部分===============================================================
        # ================================================================================================

        self.decoder_embed_dim = embed_dim // 2
        self.decoder_depth = 3
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.decoder_embed_dim))


        self.decoder_embed = nn.Linear(embed_dim, self.decoder_embed_dim, bias=True)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, 128 + 1, self.decoder_embed_dim))
        self.decoder_blocks = nn.ModuleList([Block(
            dim=self.decoder_embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_norm=qk_norm, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
            init_values=init_values, window_size=self.patch_embed.patch_shape if use_rel_pos_bias else None,
            attn_head_dim=attn_head_dim,
        ) for i in range(self.decoder_depth)])
        self.decoder_norm = norm_layer(self.decoder_embed_dim)

        self.pred_embed = nn.Linear(self.decoder_embed_dim, embed_dim, bias=True)



        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=self.init_std)
            trunc_normal_(self.decoder_pos_embed, std=self.init_std)
        trunc_normal_(self.time_embed, std=self.init_std)
        trunc_normal_(self.cls_token, std=self.init_std)
        trunc_normal_(self.mask_token, std=self.init_std)
        trunc_normal_(self.lm_head.weight, std=self.init_std)
        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'time_embed'}

    def get_num_layers(self):
        return len(self.blocks)
    
    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        # 按照mask_ratio比例抽出的x_masked
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, input_chans, mask_ratio):
        batch_size, c, time_window, _ = x.size()
        x = self.patch_embed(x) # B 256 200
        batch_size, seq_len, _ = x.size()

        # cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        # mask_token = self.mask_token.expand(batch_size, seq_len, -1)

        # 对比：上面是使用visible + marked；下面是只是用visible作为输入
        # w = bool_masked_pos.unsqueeze(-1).type_as(mask_token)
        # x = x * (1 - w) + mask_token * w

        # 第一步：先加上pos_embed 和 time_embed
        # pos_embed_used = 包括cls_token，以及所有用到的channel的位置嵌入 = （1， 65， D)
        pos_embed_used = self.pos_embed[:, input_chans] if input_chans is not None else self.pos_embed
        print('SHAPE', pos_embed_used.shape)

        if self.pos_embed is not None:
            pos_embed = pos_embed_used[:, 1:, :].unsqueeze(2).expand(batch_size, -1, time_window, -1).flatten(1, 2)
            # pos_embed = torch.cat((pos_embed[:,0:1,:].expand(batch_size, -1, -1), pos_embed), dim=1)
            x = x + pos_embed
        if self.time_embed is not None:
            time_embed = self.time_embed[:, 0:time_window, :].unsqueeze(1).expand(batch_size, c, -1, -1).flatten(1, 2)
            # x[:, 1:, :] += time_embed
            x = x + time_embed
        # x = self.pos_drop(x)


        # 第二步：random_mask，抽出visible的部分
        x, mask, ids_restore = self.random_masking(x, mask_ratio=mask_ratio)

        # 第三步：添加cls_token
        cls_token = self.cls_token + pos_embed_used[:, :1, :]
        cls_tokens = cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        # for blk in self.blocks:
        #     x = blk(x, rel_pos_bias=rel_pos_bias)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore
    
    def forward_decoder(self, latent, ids_restore, input_chans):
        latent = self.decoder_embed(latent) # -> (B, n, decoder_dim)
        mask_tokens = self.mask_token.repeat(latent.shape[0], ids_restore.shape[1] - latent.shape[1] + 1, 1)
        unshuffled_recs = torch.cat([latent[:, 1:, :], mask_tokens], dim=1)
        resc = torch.gather(unshuffled_recs, dim=1, index=ids_restore.unsqueeze(-1).expand(-1,-1,latent.shape[2]))
        resc = torch.cat([latent[:, :1, :], resc], dim=1)

        pos_embed_used = self.decoder_pos_embed[:, input_chans] # 包含了cls_token (1, 65, 100)
        pos_embed = pos_embed_used[:, 1:, :].unsqueeze(2).expand(resc.shape[0], -1, 4, -1).flatten(1,2)
        pos_embed = torch.cat([pos_embed_used[:, :1, :].expand(resc.shape[0], -1, -1), pos_embed], dim=1)

        resc = resc + pos_embed

        for blk in self.decoder_blocks:
            resc = blk(resc)
        resc = self.decoder_norm(resc)

        resc = resc[:, 1:, :] # remove cls token
        return resc
    
    def forward_loss(self, x, pred, mask):
        """
        x: [Batch_size, Chan_nums, Window_size, Trial_length]
        pred: [Batch_size, Chan_nums * Window_size, Trial_length]
        mask: [N, L], 0 is keep, 1 is remove, 
        """

        
        
        batch_size, c, time_window, trial_length = x.size()

        # target = x.reshape(batch_size, c * time_window, trial_length)
        target = rearrange(x, 'B N A T -> B (N A) T')
        # target = self.patchify(x)

        # if self.norm_pix_loss:
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1.e-6)**.5

        pred = self.pred_embed(pred)

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss


    def forward(self, x, input_chans=None, mask_ratio=0.75, return_all_tokens=False, return_patch_tokens=False, return_all_patch_tokens=False):
        """
        x: B N A T
        latent: encoded visible data. (B,N*(1-mask_ratio)+1,D) dim1还包含cls_token
        mask: 随机打乱的0/1 矩阵, 0:visible 1:mask (B, N)
        ids_restore: (B, N) dim=1的索引矩阵, 记录mask在dim=1上原来的index, 最终把聚集起来的mask打散回原本的data中需要用到
        """

        latent, mask, ids_restore = self.forward_encoder(x, input_chans, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore, input_chans)  # [N, L, p*p*3]
        loss = self.forward_loss(x, pred, mask)
        print(loss)
        return loss, pred, mask
        # if return_all_patch_tokens:
        #     return x
        # x = x[:, 1:]
        # if return_patch_tokens:
        #     return x
        # if return_all_tokens:
        #     return self.lm_head(x)
        # else:
        #     # return the masked tokens
        #     return self.lm_head(x[bool_masked_pos])
    
    # def forward_return_qkv(self, x, bool_masked_pos=None, split_out_as_qkv=False):
    #     if bool_masked_pos is None:
    #         bool_masked_pos = torch.zeros((x.shape[0], x.shape[1] * x.shape[2]), dtype=torch.bool).to(x.device)
    #     x = self.patch_embed(x, bool_masked_pos=bool_masked_pos)
    #     batch_size, seq_len, _ = x.size()

    #     cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
    #     mask_token = self.mask_token.expand(batch_size, seq_len, -1)

    #     # replace the masked EEG tokens by mask_token
    #     w = bool_masked_pos.unsqueeze(-1).type_as(mask_token)
    #     x = x * (1 - w) + mask_token * w

    #     x = torch.cat((cls_tokens, x), dim=1)
    #     if self.pos_embed is not None:
    #         x = x + self.pos_embed
    #     x = self.pos_drop(x)

    #     rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
    #     for i, blk in enumerate(self.blocks):
    #         if i < len(self.blocks) - 1:
    #             x = blk(x, rel_pos_bias=rel_pos_bias)
    #         else:
    #             # with torch.cuda.amp.autocast(enabled=False):
    #             x, qkv = blk(x, rel_pos_bias=rel_pos_bias, return_qkv=True)

    #     if split_out_as_qkv:
    #         x = self.norm(x)
    #         x = self.lm_head(x) # [b, n+1, 3*c]
    #         q, k, v = x.chunk(3, dim=-1) # [b, n+1, c]
    #         b, n, c =q.shape
    #         q = q.reshape(b, n, self.num_heads, -1).permute(0, 2, 1, 3)
    #         k = k.reshape(b, n, self.num_heads, -1).permute(0, 2, 1, 3)
    #         v = v.reshape(b, n, self.num_heads, -1).permute(0, 2, 1, 3)
    #         return x, q, k, v
    #     else:
    #         x = self.norm(x)
    #         x = x[:, 1:]
    #         x = self.lm_head(x[bool_masked_pos])

    #         q, k, v = qkv[0], qkv[1], qkv[2]

    #     return x, q, k, v

    # def get_last_selfattention(self, x):
    #     x = self.patch_embed(x)
    #     batch_size, seq_len, _ = x.size()
    #     cls_tokens = self.cls_token.expand(batch_size, -1, -1)
    #     x = torch.cat((cls_tokens, x), dim=1)
    #     if self.pos_embed is not None:
    #         x = x + self.pos_embed
    #     x = self.pos_drop(x)
    #     rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None

    #     for i, blk in enumerate(self.blocks):
    #         if i < len(self.blocks) - 1:
    #             x = blk(x, rel_pos_bias=rel_pos_bias)
    #         else:
    #             # return attention of the last block
    #             return blk(x, rel_pos_bias=rel_pos_bias, return_attention=True)
            

# class NeuralTransformerForMEM(nn.Module):
#     def __init__(self, EEG_size=1600, patch_size=200, in_chans=1, out_chans=8, vocab_size=8192, embed_dim=200, depth=12,
#                  num_heads=10, mlp_ratio=4., qkv_bias=True, qk_norm=None, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
#                  drop_path_rate=0., norm_layer=None, init_values=None, attn_head_dim=None,
#                  use_abs_pos_emb=True, use_rel_pos_bias=False, use_shared_rel_pos_bias=False, init_std=0.02, **kwargs):
#         super().__init__()
#         self.patch_size = patch_size
#         self.student = NeuralTransformerForMaskedEEGModeling(EEG_size, patch_size, in_chans, out_chans, vocab_size, embed_dim, depth,
#                  num_heads, mlp_ratio, qkv_bias, qk_norm, qk_scale, drop_rate, attn_drop_rate, drop_path_rate, norm_layer, init_values, attn_head_dim,
#                  use_abs_pos_emb, use_rel_pos_bias, use_shared_rel_pos_bias, init_std)
        
#         self.lm_head = nn.Linear(embed_dim, vocab_size)
#         self.projection_head = nn.Sequential(
#             nn.Linear(embed_dim, embed_dim),
#             nn.ReLU()
#         )

#         trunc_normal_(self.lm_head.weight, std=init_std)
    
#     @torch.jit.ignore
#     def no_weight_decay(self):
#         return {'student.cls_token', 'student.pos_embed', 'student.time_embed'}
    
#     def forward(self, x, input_chans=None, bool_masked_pos=None):
#         x_masked = self.student(x, input_chans, bool_masked_pos, return_all_patch_tokens=True)
#         x_masked_no_cls = x_masked[:, 1:]

#         # x_masked_no_cls[bool_masked_pos] shape为 N, embed_dim
#         # x_rec (N, vacab_size)
#         x_rec = self.lm_head(x_masked_no_cls[bool_masked_pos]) # linear project到8192个类别上

#         #symetric
#         x_masked_sym = self.student(x, input_chans, ~bool_masked_pos, return_all_patch_tokens=True)
#         x_masked_no_cls_sym = x_masked_sym[:, 1:]
#         x_rec_sym = self.lm_head(x_masked_no_cls_sym[~bool_masked_pos])

#         return x_rec, x_rec_sym


@register_model
# pretrained为True表示使用已经预训练好的模型
def model_mega(pretrained=False, **kwargs):
    model = NeuralTransformerForMaskedEEGModeling(init_values=kwargs['init_values'])
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model


# @register_model
# def labram_base_patch200_1600_8k_vocab(pretrained=False, **kwargs): #5M
#     if "num_classes" in kwargs:
#         _ = kwargs.pop("num_classes")
#     if 'vocab_size' in kwargs:
#         vocab_size = kwargs['vocab_size']
#         _ = kwargs.pop("vocab_size")
#     else:
#         vocab_size = 8192
#     model = NeuralTransformerForMEM(
#         patch_size=200, embed_dim=200, depth=12, num_heads=10, mlp_ratio=4, qkv_bias=False, qk_norm=partial(nn.LayerNorm, eps=1e-6),
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), vocab_size=vocab_size, **kwargs)
#     model.default_cfg = _cfg()
#     if pretrained:
#         checkpoint = torch.load(
#             kwargs["init_ckpt"], map_location="cpu"
#         )
#         model.load_state_dict(checkpoint["model"])
#     return model


# @register_model
# def labram_large_patch200_1600_8k_vocab(pretrained=False, **kwargs): #50M
#     if "num_classes" in kwargs:
#         _ = kwargs.pop("num_classes")
#     if 'vocab_size' in kwargs:
#         vocab_size = kwargs['vocab_size']
#         _ = kwargs.pop("vocab_size")
#     else:
#         vocab_size = 8192
#     model = NeuralTransformerForMEM(
#         patch_size=200, embed_dim=400, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=False, qk_norm=partial(nn.LayerNorm, eps=1e-6), out_chans=16,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), vocab_size=vocab_size, **kwargs)
#     model.default_cfg = _cfg()
#     if pretrained:
#         checkpoint = torch.load(
#             kwargs["init_ckpt"], map_location="cpu"
#         )
#         model.load_state_dict(checkpoint["model"])
#     return model

# @register_model
# def labram_huge_patch200_1600_8k_vocab(pretrained=False, **kwargs): #380M
#     if "num_classes" in kwargs:
#         _ = kwargs.pop("num_classes")
#     if 'vocab_size' in kwargs:
#         vocab_size = kwargs['vocab_size']
#         _ = kwargs.pop("vocab_size")
#     else:
#         vocab_size = 8192
#     model = NeuralTransformerForMEM(
#         patch_size=200, embed_dim=800, depth=48, num_heads=16, mlp_ratio=4, qkv_bias=False, qk_norm=partial(nn.LayerNorm, eps=1e-6), out_chans=32,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), vocab_size=vocab_size, **kwargs)
#     model.default_cfg = _cfg()
#     if pretrained:
#         checkpoint = torch.load(
#             kwargs["init_ckpt"], map_location="cpu"
#         )
#         model.load_state_dict(checkpoint["model"])
#     return model
