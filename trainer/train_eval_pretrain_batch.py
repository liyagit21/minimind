import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
import torch
import torch.distributed as dist
import math
import numpy as np
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import PretrainDataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler
from torch.utils.data import random_split

warnings.filterwarnings('ignore')


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    start_time = time.time()
    # print("***len(loader)***", len(loader))
    for step, (X, Y, loss_mask) in enumerate(loader, start=start_step + 1):
        # print("***step***", step)
        # # 检查batch是否完整，不完整则跳过
        # if X.size(0) != args.batch_size:
        #     if step % args.log_interval == 0:  # 只在日志间隔时记录跳过信息
        #         Logger(f'Epoch:[{epoch+1}/{args.epochs}]({step}/{iters}) Skip incomplete batch: {X.size(0)} samples')
        #     continue
            
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)
        # lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        lr = args.learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            res = model(X)
            loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),
                Y.view(-1)
            ).view(Y.size())

            loss = (loss * loss_mask).sum() / loss_mask.sum()
            # print("***res.aux_loss:", res.aux_loss)
            loss += res.aux_loss
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)  # 解除梯度缩放
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)  # 梯度裁剪

            scaler.step(optimizer)  # 更新模型参数
            scaler.update()         # 更新缩放器

            optimizer.zero_grad(set_to_none=True) # 清空梯度
            torch.cuda.empty_cache()  # 释放GPU缓存

        if step % args.log_interval == 0 or step == iters - 1:  # 记录日志
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
            # max_vio_t = np.mean(res.aggregated_stats["max_vio_tokens"])
            max_vio = np.mean(res.aggregated_stats["max_vio_selections"])
            load_imbalance = np.mean(res.aggregated_stats["load_imbalance"])
            expert_sparsity = np.mean(res.aggregated_stats["expert_sparsity"])
            Logger(f'Epoch:[{epoch+1}/{args.epochs}]({step}/{iters}) loss:{current_loss:.6f} lr:{current_lr:.12f} epoch_Time:{eta_min}min: max_vio:{max_vio} load_imbalance:{load_imbalance} expert_sparsity:{expert_sparsity}')
            # Logger(f'Epoch:[{epoch+1}/{args.epochs}]({step}/{iters}) loss:{current_loss:.6f} lr:{current_lr:.12f} epoch_Time:{eta_min}min: \
            #        max_vio:{max_vio} max_vio_t:{max_vio_t} load_imbalance:{load_imbalance}')
            
            if wandb: wandb.log({"loss": current_loss, "lr": current_lr, "epoch_Time": eta_min, "max_vio": max_vio, "load_imbalance": load_imbalance, "expert_sparsity": expert_sparsity})
            # if wandb: wandb.log({"loss": current_loss, "lr": current_lr, "epoch_Time": eta_min, \
            #                      "max_vio": max_vio, "max_vio_t": max_vio_t, "load_imbalance": load_imbalance})

        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            state_dict = {k: v.half() for k, v in state_dict.items()}  # 半精度保存
            torch.save(state_dict, ckp)
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
            
            # 在保存模型时也进行验证
            if val_ds is not None and val_loader is not None:
                perplexity = evaluate_model(model, val_loader, args.device, autocast_ctx)
                Logger(f'Validation perplexity after saving: {perplexity:.4f}')
                if wandb: wandb.log({"val_perplexity": perplexity})
            
            model.train()
            del state_dict
        del X, Y, loss_mask, res, loss

def evaluate_model(model, val_loader, device, autocast_ctx):
    """评估模型并返回困惑度，跳过不完整的batch"""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_batches = 0
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    
    with torch.no_grad():
        for X, Y, loss_mask in val_loader:
            # # 检查batch是否完整，不完整则跳过
            # if X.size(0) != args.val_batch_size:
            #     continue
                
            X = X.to(device)
            Y = Y.to(device)
            loss_mask = loss_mask.to(device)
            
            with autocast_ctx:
                res = model(X)
                loss = loss_fct(
                    res.logits.view(-1, res.logits.size(-1)),
                    Y.view(-1)
                ).view(Y.size())
                
                # 只计算有效token的损失
                masked_loss = loss * loss_mask
                total_loss += masked_loss.sum().item()
                total_tokens += loss_mask.sum().item()
                total_batches += 1
    
    if total_tokens == 0:
        Logger(f'Warning: No valid batches in validation!')
        return float('inf')
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    
    Logger(f'Validation: {total_batches} batches, {total_tokens} tokens, avg_loss: {avg_loss:.4f}')
    model.train()
    return perplexity


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='pretrain', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数（建议1轮zero或2-6轮充分训练）")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--val_batch_size", type=int, default=32, help="验证集batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=16, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=2000, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=512, type=int, help="训练的最大截断长度")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_hq.jsonl", help="预训练数据路径")
    parser.add_argument('--from_weight', default='none', type=str, help="基于哪个权重训练，为none则从头开始")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain", help="wandb项目名")
    parser.add_argument("--val_split", type=float, default=0.1, help="验证集划分比例")
    parser.add_argument("--eval_interval", type=int, default=500, help="验证间隔（步数）")
    parser.add_argument("--skip_incomplete", action="store_true", default=True, help="跳过不完整的batch")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
    
    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 配wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 定义模型、数据、优化器 ==========
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    
    # 加载完整数据集
    full_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    
    # 按9:1划分训练集和验证集
    val_size = int(len(full_ds) * args.val_split)
    train_size = len(full_ds) - val_size
    
    # 确保划分在不同进程中一致
    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(full_ds, [train_size, val_size], generator=generator)
    
    Logger(f"Total samples: {len(full_ds)}")
    Logger(f"Training samples: {len(train_ds)}")
    Logger(f"Validation samples: {len(val_ds)}")
    
    # # 使用分布式采样器，每个进程看到的数据量是len(train_ds)/world_size
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    
    # 只在主进程上创建验证集数据加载器
    val_loader = None
    if val_ds is not None and (not dist.is_initialized() or dist.get_rank() == 0):
        val_loader = DataLoader(
            val_ds, 
            batch_size=args.val_batch_size, 
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True  # 我们不在这里drop，而是在验证函数中跳过
        )
        # 计算完整的batch数量
        num_val_batches = len(val_ds) // args.val_batch_size
        Logger(f"Validation will use {num_val_batches} full batches (skipping incomplete last batch)")
    
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7. DDP包模型 ==========
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        # model = DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        if epoch == start_epoch and start_step > 0: # 第一个epoch且存在检查点
            batch_sampler = SkipBatchSampler(train_sampler or range(len(train_ds)), args.batch_size, start_step + 1)
            loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True, drop_last=True)
            # actual_batches = (len(train_ds) - (start_step + 1) * args.batch_size) // args.batch_size
            # Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始，实际可用batch: {actual_batches}')
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + start_step + 1, start_step, wandb)
            # train_epoch(epoch, loader, actual_batches, start_step, wandb)
        else: # 默认从头开始
            loader = DataLoader(
                train_ds, 
                batch_size=args.batch_size, 
                shuffle=(train_sampler is None), 
                sampler=train_sampler, 
                num_workers=args.num_workers, 
                pin_memory=True,
                drop_last=True  
            )
            
            # # 计算完整的batch数量
            # total_full_batches = len(train_ds) // args.batch_size
            # Logger(f'Training will use {total_full_batches} full batches (skipping incomplete last batch)')
            
            # 训练前进行初始验证（只在主进程）
            # if val_loader is not None and epoch == start_epoch and start_step == 0:
            #     initial_perplexity = evaluate_model(model, val_loader, args.device, autocast_ctx)
            #     Logger(f'Initial validation perplexity: {initial_perplexity:.4f}')
            #     if wandb: wandb.log({"val_perplexity": initial_perplexity})
            # print("***len(loader):", len(loader))
            train_epoch(epoch, loader, len(loader), 0, wandb)
            
        # 每个epoch结束后进行验证（只在主进程）
        if val_loader is not None and (not dist.is_initialized() or dist.get_rank() == 0):
            perplexity = evaluate_model(model, val_loader, args.device, autocast_ctx)
            Logger(f'Epoch [{epoch+1}/{args.epochs}] validation perplexity: {perplexity:.4f}')
            if wandb: wandb.log({"val_perplexity": perplexity, "epoch": epoch+1})
