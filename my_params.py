# vqnsp_encoder
EEG_size=1600, # encoder是1600， decoder = 1600 // patch_size(200)
patch_size=200, # decoder = 1
in_chans=1,
num_classes=0,
embed_dim=200,
depth=12, # 层数 12 or 24  decoder的depth是3
num_heads=10,
mlp_ratio=4.,
qkv_bias=True,
qk_scale=None,
drop_rate=0.,
attn_drop_rate=0.,
drop_path_rate=0.,
norm_layer=partial(nn.LayerNorm, eps=1e-6),
init_values=0.,
use_abs_pos_emb=True, 
use_rel_pos_bias=False,
use_shared_rel_pos_bias=False,
use_mean_pooling=True,
init_scale=0.001