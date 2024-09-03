# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# By Wei-Bang Jiang
# Based on BEiT-v2, timm, DeiT, and DINO code bases
# https://github.com/microsoft/unilm/tree/master/beitv2
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# ---------------------------------------------------------

from cgitb import enable
import math
import sys
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from einops import rearrange
from contextlib import nullcontext

def random_masking(x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        # x-shape = (B, 256, 200)
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        # noise = torch.tensor([[0.3, 0.1, 0.4, 0.2],
        #               [0.6, 0.7, 0.5, 0.8],
        #               [0.9, 0.2, 0.4, 0.3]])
        
        # ([[1, 3, 0, 2],   # 第 0 行排序后，[0.1, 0.2, 0.3, 0.4] 对应原始索引 [1, 3, 0, 2]
        # [2, 0, 1, 3],   # 第 1 行排序后，[0.5, 0.6, 0.7, 0.8] 对应原始索引 [2, 0, 1, 3]
        # [1, 3, 2, 0]])  # 第 2 行排序后，[0.2, 0.3, 0.4, 0.9] 对应原始索引 [1, 3, 2, 0]

        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore) # 通过 ids_restore 提供的索引顺序重新排列 mask 张量中的元素
        # mask = np.hstack([
        #     np.zeros(len_keep),
        #     np.ones(L - len_keep),
        # ])
        # np.random.shuffle(mask)

        return mask.to(torch.bool) # 生成掩码张量，0为False，非0为True


def train_one_epoch(model: torch.nn.Module, vqnsp: torch.nn.Module,
                    data_loader_list: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    log_writer=None, lr_scheduler=None, start_steps=None,
                    lr_schedule_values=None, wd_schedule_values=None, ch_names_list=None, args=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    loss_fn = nn.CrossEntropyLoss()

    step_loader = 0
    for data_loader, ch_names in zip(data_loader_list, ch_names_list):
        # data_loader对应ShockDataSet，同一个规范下的数据集
        if len(data_loader) == 0:
            continue
        input_chans = utils.get_input_chans(ch_names) # 一个数字序列的数组，表示ch_names对应的编号
        for step, (batch) in enumerate(metric_logger.log_every(data_loader, print_freq * args.gradient_accumulation_steps, header)):
            # assign learning rate & weight decay for each step
            it = start_steps + step + step_loader  # global training iteration
            # 动态更新优化器的lr和weight_decay参数
            if lr_schedule_values is not None or wd_schedule_values is not None:
                for i, param_group in enumerate(optimizer.param_groups):
                    if lr_schedule_values is not None:
                        param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                    if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                        param_group["weight_decay"] = wd_schedule_values[it]

            samples = batch
            samples = samples.float().to(device, non_blocking=True) / 100 # non_blocking 允许异步的数据传输，从而可能提高性能。
            samples = rearrange(samples, 'B N (A T) -> B N A T', T=200) # (B, 64, 4, 200) or (B, 32, 8, 200)
            bool_masked_pos = random_masking(samples.flatten(1, 2), mask_ratio=0.5).to(device, non_blocking=True)
            # samples.flatten(1, 2) 会将第1个到第2个维度合并成一个，所以上述两种最终都会变成：(B, 256, 200)
            # x-shape = (B, 256, 200)
            # bool_masked_pos的shape （B, 256）的bool maxtrix，随机掩码张量

            with torch.no_grad():
                with torch.cuda.amp.autocast(): #  启用混合精度训练
                    input_ids = vqnsp.get_codebook_indices(samples, input_chans)

                # 结果 labels 将是一个一维张量，包含所有被掩盖的元素（即那些在 bool_masked_pos 中为 True 的位置对应的 input_ids 中的元素）
                labels = input_ids[bool_masked_pos] # 被掩盖的元素的codebook id
                labels_sym = input_ids[~bool_masked_pos]

            my_context = model.no_sync if args.distributed and (step + 1) % args.gradient_accumulation_steps != 0 else nullcontext
            # 这是一个 PyTorch 提供的上下文管理器，用于在分布式数据并行（Distributed Data Parallel，DDP）中临时关闭梯度同步。在 no_sync 上下文中，模型不会在每个步骤同步梯度，从而减少通信开销，直到累积到足够的步数再进行同步。
            with my_context():
                with torch.cuda.amp.autocast(): # enabled=False
                    outputs = model(samples, input_chans, bool_masked_pos=bool_masked_pos)

                    x_rec, x_rec_sym = outputs
                    loss_rec = loss_fn(x_rec, labels)
                    loss_rec_sym = loss_fn(x_rec_sym, labels_sym)
                    loss = loss_rec + loss_rec_sym

            loss_value = loss.item()

            if not math.isfinite(loss_value):
                print(f"Loss is {loss_value}, stopping training at rank {utils.get_rank()}", force=True)
                
                sys.exit(1)

            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= args.gradient_accumulation_steps
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order, update_grad=(step + 1) % args.gradient_accumulation_steps == 0)
            loss_scale_value = loss_scaler.state_dict()["scale"]
            # 通过梯度累积，我们可以将多个小批量的梯度累积起来，从而在内存允许的情况下模拟更大的批量。这种方法既节省了内存，又能够使梯度更加稳定，从而提高训练的效果。
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.zero_grad()

            torch.cuda.synchronize()
            
            mlm_acc = (x_rec.max(-1)[1] == labels).float().mean().item()
            mlm_acc_sym = (x_rec_sym.max(-1)[1] == labels_sym).float().mean().item()
            metric_logger.update(mlm_acc=mlm_acc)
            metric_logger.update(mlm_acc_sym=mlm_acc_sym)
            metric_logger.update(loss_rec=loss_rec.item() / 2)

            if log_writer is not None:
                log_writer.update(mlm_acc=mlm_acc, head="loss")
                log_writer.update(mlm_acc_sym=mlm_acc_sym, head="loss")
                log_writer.update(loss_rec=loss_rec.item() / 2, head="loss")

            metric_logger.update(loss=loss_value)
            metric_logger.update(loss_scale=loss_scale_value)
            min_lr = 10.
            max_lr = 0.
            for group in optimizer.param_groups:
                min_lr = min(min_lr, group["lr"])
                max_lr = max(max_lr, group["lr"])

            metric_logger.update(lr=max_lr)
            metric_logger.update(min_lr=min_lr)
            weight_decay_value = None
            for group in optimizer.param_groups:
                if group["weight_decay"] > 0:
                    weight_decay_value = group["weight_decay"]
            metric_logger.update(weight_decay=weight_decay_value)
            metric_logger.update(grad_norm=grad_norm)

            if log_writer is not None:
                log_writer.update(loss=loss_value, head="loss")
                log_writer.update(loss_scale=loss_scale_value, head="opt")
                log_writer.update(lr=max_lr, head="opt")
                log_writer.update(min_lr=min_lr, head="opt")
                log_writer.update(weight_decay=weight_decay_value, head="opt")
                log_writer.update(grad_norm=grad_norm, head="opt")

                log_writer.set_step()

            if lr_scheduler is not None:
                lr_scheduler.step_update(start_steps + step + step_loader)
        step_loader += step
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

