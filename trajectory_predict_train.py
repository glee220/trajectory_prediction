from __future__ import annotations

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["NVIDIA_TF32_OVERRIDE"]  = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from dataclasses import dataclass
import argparse
import json
import random
import time
from pathlib import Path
from typing import Tuple
import numpy as np

import torch
torch.backends.cuda.matmul.allow_tf32  = False
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
torch.backends.cudnn.allow_tf32 = False

from accelerate import (
    Accelerator,
    DeepSpeedPlugin,
    DistributedDataParallelKwargs,
)
from torch import nn, optim
from torch.optim import lr_scheduler

from dataFactory.data_provider import data_provider  
from models import llm_load                            
from models.GatedAttnTrajLLM import Model        

from utils.tools import EarlyStopping
# Clean up process group when done
import torch.distributed as dist
import datetime
import contextlib
from torch import amp
from packaging import version 


def setup_environment(seed: int) -> None:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"  # 减少内存碎片
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def build_parser() -> argparse.ArgumentParser:
    
    parser = argparse.ArgumentParser("Llama trajectory predict trainer")

    # ─────────────────────── General ───────────────────────
    parser.add_argument("--task_name", default="trajectory_forecast")
    parser.add_argument("--load_mode", choices=["debug", "social", "map"], default="social")
    parser.add_argument("--is_training", type=int, default=1)
    parser.add_argument("--model_comment", default="trajectory_forecast",help='prefix when saving test results')

    # ─────────────────────── Data ───────────────────────────
    parser.add_argument("--data", default="Nuscene")
    parser.add_argument("--root_path", default="./precess_data")
    parser.add_argument("--seq_len", type=int, default=4)
    parser.add_argument("--pred_len", type=int, default=12)
    parser.add_argument("--micro_batch", type=int, default=24, help="samples per GPU, per step")
    parser.add_argument("--grad_acc", type=int, default=2, help="gradient-accumulation steps")
    parser.add_argument("--eval_batch_size", type=int, default=24)

    # ─────────────────────── Model ──────────────────────────
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--llm_layers", type=int, default=32) 
    parser.add_argument('--llm_dim', type=int, default='4096', help='LLM model dimension')
    # parser.add_argument("--llm_model", default="LLAMA")
    # parser.add_argument("--llm_model_path" , default='./models/Llama-2-7b-hf/')    
    parser.add_argument("--llm_model", default="LLAMA3")
    parser.add_argument("--llm_model_path" , default='./models/Meta-Llama-3-8B-Instruct/')  
    # parser.add_argument("--llm_model", default="QWEN")
    # parser.add_argument("--llm_model_path" , default='./models/qwen2.5-VL-7B-Instruct/')       
    # parser.add_argument('--llm_dim', type=int, default='3584', help='LLM model dimension')
    # parser.add_argument("--llm_model", default="Mistral")
    # parser.add_argument("--llm_model_path" , default='./models/Mistral-7B-Instruct-v0.2/') 
    # parser.add_argument("--llm_model", default="vicuna")
    # parser.add_argument("--llm_model_path" , default='./models/vicuna-7b-v1.5/')     
    # parser.add_argument("--llm_model", default="WizardLM")
    # parser.add_argument("--llm_model_path" , default='./models/WizardLM-7B-V1.0/')  
    
    # ─────────────────────── Optim ──────────────────────────
    # parser.add_argument('--num_workers', type=int, default=4, help='data loader num workers')
    parser.add_argument('--num_workers', type=int, default=4, help='data loader num workers')
    parser.add_argument("--train_epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5, help='early stopping patience')
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    # parser.add_argument("--lradj", choices=["OneCycle", "CosineRestart", "ReduceOnPlateau", "WarmupCosine"], default="WarmupCosine")                                   
    parser.add_argument("--lradj", choices=["OneCycle", "CosineRestart", "ReduceOnPlateau", "WarmupCosine"], default="CosineRestart")                                   
    parser.add_argument("--pct_start", type=float, default=0.3, help='The percentage of the cycle (in number of steps) spent increasing the learning rate in OneCycle，默认0.3')

    # ─────────────────────── Misc ───────────────────────────
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--checkpoints", default="./checkpoints")
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument("--ds_config", default="./config/ds_config_onslurm.json")
    parser.add_argument("--use_deepspeed",action="store_true",default=True,help="Enable DeepSpeed ZeRO & related config (picked up in get_accelerator)")
    parser.add_argument('--local_rank', type=int, default=0,help='[auto] provided by deepspeed.launch')
    parser.add_argument('--deepspeed', type=str, default='./config/ds_config_onslurm.json',help='[auto] config file path injected by launcher')
    parser.add_argument('--master_port', type=str, default=29555)
    parser.add_argument('--resume', action='store_true', help='如果设置，则从中断的训练中恢复；否则从头开始')


    # ─────────────────────── Ablation ───────────────────────────
    parser.add_argument('--use_map',action='store_true', default=False, help="预测时是否加入地图")
    

    return parser

def get_accelerator(args) -> Accelerator:
    # ddp_kwargs 决定模型在torch.nn.parallel.DistributedDataParallel中的打包方式,
    # find_unused_parameters=True将多线程训练中没有梯度的模型的梯度赋值为0，方便后续训练顺利进行
    # deepspeed_plugin一种模型训练过程中进程管理的手段，确保训练更快的进行  
    ddp = DistributedDataParallelKwargs(find_unused_parameters=True)
    ds_plugin = None
    if args.use_deepspeed:
        ds_plugin = DeepSpeedPlugin(hf_ds_config=args.ds_config) 

    return Accelerator(kwargs_handlers=[ddp], 
                       deepspeed_plugin=ds_plugin,
                       mixed_precision=args.mixed_precision)
# ---------------------------------------------------------------------------
# Core trainer
# ---------------------------------------------------------------------------

@dataclass
class TrainerConfig:
    args: argparse.Namespace
    accelerator: Accelerator
    criterion: nn.Module = nn.MSELoss()
    mae_metric: nn.Module = nn.L1Loss()

class Trainer:

    def __init__(self, cfg: TrainerConfig):
        self.cfg = cfg
        self.device = cfg.accelerator.device

        # —— checkpoint/日志路径 ——         
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.ckpt_dir = self._ckpt_dir(ts)

        self.log_path = self.ckpt_dir / f"loss_{self.cfg.args.model_comment}_{ts}.json"

        # —— 训练模式：resume=True 则从 checkpoint 继续，否则全新开始 —— 
        self.start_epoch = 0
        
        # 每次训练覆盖保存最新状态
        self.ckpt_path = self.ckpt_dir / "latest_checkpoint.pth"    
        # 用于记录至今为止最低的验证 loss
        self.best_val_loss = float("inf")
        # Data
        self.train_dl,self.val_dl = self._build_dataloaders()

        # Model
        llm, tokenizer = llm_load(cfg.args)
        self.model = Model(configs=cfg.args, llm_model=llm, tokenizer=tokenizer)
        # ❹ 训练前打印一下有效 batch，方便确认
        world = self.cfg.accelerator.num_processes
        self.cfg.args.batchsize = self.cfg.args.micro_batch * self.cfg.args.grad_acc * world
            
        # Optim & LR
        self.optimizer = optim.Adam(self._trainable_params(), lr=cfg.args.learning_rate)
        self.scheduler = self._build_scheduler()

        # Early stopping & metrics (verbose=True to print each check)
        self.early_stopping = EarlyStopping(accelerator=cfg.accelerator, patience=cfg.args.patience, verbose=True)
        # Prepare with Accelerator
        (self.train_dl, self.val_dl, self.model, self.optimizer, self.scheduler) = cfg.accelerator.prepare(self.train_dl, self.val_dl, self.model, self.optimizer, self.scheduler)

        if getattr(self.cfg.args, "resume", False):
            unwrapped = self.cfg.accelerator.unwrap_model(self.model)
            model_engine = getattr(unwrapped, "engine", None)
            if model_engine is not None:
                load_path, _ = model_engine.load_checkpoint(str(self.ckpt_dir))
                if load_path is None:
                    print(f"[Resume] No DeepSpeed checkpoint found in {self.ckpt_dir}, training from scratch.")
                else:
                    print(f"[Resume] Successfully restored from {load_path}")
            else:
                print("[Resume] Not using DeepSpeed engine, fallback to training from scratch.")

            self.start_epoch = 0
            state_path = self.ckpt_dir / "trainer_state.json"
            if state_path.exists():
                with open(state_path, "r") as f:
                    self.start_epoch = json.load(f).get("next_epoch", 0)

    def _trainable_params(self):
        return [p for p in self.model.parameters() if p.requires_grad]
    def _build_scheduler(self):
        args = self.cfg.args
        steps_per_epoch = len(self.train_dl)
        
        # 新增带warmup的调度器
        if args.lradj == "WarmupCosine":
            warmup_epochs = max(2, int(0.1 * args.train_epochs))  # 10%的epoch用于warmup
            return lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[
                    lr_scheduler.LinearLR(
                        self.optimizer, 
                        start_factor=0.01,  # 从初始学习率的1%开始
                        total_iters=warmup_epochs
                    ),
                    lr_scheduler.CosineAnnealingLR(
                        self.optimizer,
                        T_max=args.train_epochs - warmup_epochs,
                        eta_min=args.learning_rate * 0.01  # 最小学习率为初始的1%
                    )
                ],
                milestones=[warmup_epochs]
            )
        
        # 改进的OneCycle策略（更安全）
        if args.lradj == "OneCycle":
            # 动态调整最大学习率（避免过大导致发散）
            max_lr = min(args.learning_rate * 1.5, 5e-3)  # 不超过初始学习率1.5倍或5e-3
            return lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=max_lr,
                epochs=args.train_epochs,
                steps_per_epoch=steps_per_epoch,
                pct_start=args.pct_start,
                anneal_strategy='cos',  # 使用余弦退火而非线性
                div_factor=25.0,        # 初始学习率 = max_lr / 25
                final_div_factor=100.0   # 最终学习率 = max_lr / (25*100)
            )
        
        # 改进的StepLR（带warmup）
        if args.lradj == "Step":
            # 添加前3个epoch的warmup
            warmup = lr_scheduler.LinearLR(
                self.optimizer, 
                start_factor=0.1,
                total_iters=3
            )
            main_scheduler = lr_scheduler.StepLR(
                self.optimizer, 
                step_size=5, 
                gamma=0.7
            )
            return lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[warmup, main_scheduler],
                milestones=[3]
            )
        
        # 带重启的余弦退火（适合跳出局部最优）
        if args.lradj == "CosineRestart":
            return lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=10,             # 初始周期长度
                T_mult=2,            # 每次重启周期长度加倍
                eta_min=args.learning_rate * 0.001
            )
        
        # 自适应调度器（基于验证集表现）
        if args.lradj == "ReduceOnPlateau":
            return lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',          # 监控验证损失
                factor=0.5,          # 学习率减半
                patience=3,          # 3个epoch无改善则调整
                verbose=True,        # 打印调整信息
                min_lr=1e-6
            )
        
        # 改进的Cosine（带最小学习率限制）
        if args.lradj == "Cosine":
            return lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=args.train_epochs, 
                eta_min=args.learning_rate * 0.01  # 不低于初始学习率的1%
            )
        
        # 默认使用带warmup的常量（优于纯常量）
        warmup = lr_scheduler.LinearLR(
            self.optimizer, 
            start_factor=0.01, 
            total_iters=5
        )
        constant = lr_scheduler.ConstantLR(
            self.optimizer, 
            factor=1.0
        )
        return lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[warmup, constant],
            milestones=[5]
        )
    def _build_dataloaders(self):
        args, acc = self.cfg.args, self.cfg.accelerator

        # 拆包拿到 dataset 和 dataloader
        train_data_set, train_dl = data_provider(args, "train")
        val_data_set, val_dl   = data_provider(args, "val")

        return train_dl, val_dl

    def _training_step(self, batch):
        (batch_agent_vec, batch_nbr_rlpos_mask, batch_nbr_rlpos, batch_nbr_vec_mask, batch_nbr_vec, batch_agent_pos, batch_vehicle_map) = batch

        args = self.cfg.args
        device = self.device
        precision = next(self.model.parameters()).dtype # 跟权重保持一致
        x_agent = batch_agent_vec[:, :args.seq_len].to(device,dtype=precision)
        x_nbr_vec = batch_nbr_vec[:, :, :args.seq_len].to(device,dtype=precision)
        x_nbr_vec_mask = batch_nbr_vec_mask[:, :, :args.seq_len].to(device)
        x_nbr_rlpos = batch_nbr_rlpos[:, :, :args.seq_len].to(device,dtype=precision)
        x_nbr_rlpos_mask = batch_nbr_rlpos_mask[:, :, :args.seq_len].to(device)
        y_pos = batch_agent_pos[:, -args.pred_len:].to(device,dtype=precision)

        use_mp = precision != torch.float32 
        ctx = amp.autocast if use_mp else contextlib.nullcontext
        with ctx("cuda", dtype=precision):
            if args.use_map:         
                vehicle_map = batch_vehicle_map.to(device,dtype=precision)
                # preds,adjusted_trajectory = self.model(
                preds = self.model(
                    batch_agent_vec=x_agent,
                    batch_nbr_vec=x_nbr_vec,
                    batch_nbr_vec_mask=x_nbr_vec_mask,
                    batch_nbr_rlpos=x_nbr_rlpos,
                    batch_nbr_rlpos_mask=x_nbr_rlpos_mask,
                    batch_vehicle_map=vehicle_map,
                )            
            else:
                # preds,adjusted_trajectory = self.model(
                preds = self.model(
                    batch_agent_vec=x_agent,
                    batch_nbr_vec=x_nbr_vec,
                    batch_nbr_vec_mask=x_nbr_vec_mask,
                    batch_nbr_rlpos=x_nbr_rlpos,
                    batch_nbr_rlpos_mask=x_nbr_rlpos_mask,
                )            
            loss = self.cfg.criterion(preds, y_pos)
            # 增加动力学约束损失
            # loss = self.cfg.criterion(preds, y_pos) + self.cfg.criterion(preds, adjusted_trajectory) 


        self.cfg.accelerator.backward(loss)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # NEW
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        return loss.item()
    
    def _eval_step(self, batch):
        (batch_agent_vec, batch_nbr_rlpos_mask, batch_nbr_rlpos, batch_nbr_vec_mask, batch_nbr_vec, batch_agent_pos, batch_vehicle_map) = batch
        args, device = self.cfg.args, self.device

        precision = next(self.model.parameters()).dtype

        x_agent = batch_agent_vec[:, :args.seq_len].to(device,dtype=precision)
        x_nbr_vec = batch_nbr_vec[:, :, :args.seq_len].to(device,dtype=precision)
        x_nbr_vec_mask = batch_nbr_vec_mask[:, :, :args.seq_len].to(device)
        x_nbr_rlpos = batch_nbr_rlpos[:, :, :args.seq_len].to(device,dtype=precision)
        x_nbr_rlpos_mask = batch_nbr_rlpos_mask[:, :, :args.seq_len].to(device)
        y_pos = batch_agent_pos[:, -args.pred_len:].to(device,dtype=precision)

        use_mp = precision != torch.float32 
        ctx = amp.autocast if use_mp else contextlib.nullcontext
        with ctx("cuda", dtype=precision):
            if args.use_map:          
                vehicle_map = batch_vehicle_map.to(device,dtype=precision)
                # preds,adjusted_trajectory = self.model(
                preds = self.model(
                    batch_agent_vec=x_agent,
                    batch_nbr_vec=x_nbr_vec,
                    batch_nbr_vec_mask=x_nbr_vec_mask,
                    batch_nbr_rlpos=x_nbr_rlpos,
                    batch_nbr_rlpos_mask=x_nbr_rlpos_mask,
                    batch_vehicle_map=vehicle_map,
                )            
            else:                  
                # preds,adjusted_trajectory = self.model(
                preds = self.model(
                    batch_agent_vec=x_agent,
                    batch_nbr_vec=x_nbr_vec,
                    batch_nbr_vec_mask=x_nbr_vec_mask,
                    batch_nbr_rlpos=x_nbr_rlpos,
                    batch_nbr_rlpos_mask=x_nbr_rlpos_mask,
                ) 
            loss = self.cfg.criterion(preds, y_pos)
            mae  = self.cfg.mae_metric(preds, y_pos)            
            # loss = self.cfg.criterion(preds, y_pos) + self.cfg.criterion(preds, adjusted_trajectory) 
            # mae  = self.cfg.mae_metric(preds, y_pos)+ self.cfg.mae_metric(preds, adjusted_trajectory) 
        return loss.item(), mae.item()

    def _evaluate(self, loader) -> Tuple[float, float]:
        self.model.eval()
        loss_acc, mae_acc = 0.0, 0.0
        with torch.no_grad():
            for batch in loader:
                (loss, mae) = self._eval_step(batch)
                loss_acc += loss
                mae_acc += mae
        n = len(loader)
        return loss_acc / n, mae_acc / n
    # utils
    def _ckpt_dir(self, ts) -> Path:
        p = Path(self.cfg.args.checkpoints) / f"Exp_{self.cfg.args.model_comment}_{ts}"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _append_history(self, entry: dict):
        try:
            # 打开文件写入一行 JSON
            with open(self.log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
                # 强制 flush 到 OS 缓存
                f.flush()
                # 强制写入磁盘
                os.fsync(f.fileno())
        except Exception as e:
            # 打印写入失败时的详细信息
            self.cfg.accelerator.print(
                f"[Warning] Failed to append history at epoch {getattr(self, 'current_epoch', '?')}: {e}"
            )
    def train(self) -> None:
        cfg = self.cfg
        t0 = time.time()
        # 用于异常打印
        self.current_epoch = None
        try:
            for epoch in range(self.start_epoch, cfg.args.train_epochs):
                # 记录当前 epoch（1-based）
                self.current_epoch = epoch + 1
                self.model.train()
                epoch_start = time.time()
                epoch_loss = 0.0
                for step, batch in enumerate(self.train_dl):  

                    # 1. 计算当前 batch 的样本数
                    if isinstance(batch, torch.Tensor):
                        batch_sz = batch.size(0)
                    elif isinstance(batch, (list, tuple)):
                        batch_sz = batch[0].size(0)
                    elif isinstance(batch, dict):
                        # 假设 batch 是 dict，比如 {"input_ids": Tensor[B, L], …}
                        batch_sz = next(iter(batch.values())).size(0)    

                    loss = self._training_step(batch)
                    # epoch_loss += loss.item()
                    epoch_loss += loss
                    if (step + 1) % 100 == 0 and cfg.accelerator.is_local_main_process:
                        elapsed = time.time() - t0
                        # seen_samples = 到目前为止，这个进程（GPU）看到的样本数
                        seen_samples = (step + 1) * batch_sz
                        # total_samples = 一个 epoch 这个进程（GPU）需要看到的总样本数
                        total_samples = len(self.train_dl) * batch_sz
                        cfg.accelerator.print(
                            f"epoch {epoch+1} iter {step+1} loss {loss:.4f} | "
                            f"batch_size {batch_sz} | samples {seen_samples}/{total_samples} | "
                            f"elapsed {elapsed:.1f}s"
                        )

                # Validation & early stop
                val_loss, val_mae = self._evaluate(self.val_dl)
                
                # —— 如果出现新的最低验证 loss，立刻保存最优模型 —— 
                if cfg.accelerator.is_local_main_process and val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss

                    unwrapped = cfg.accelerator.unwrap_model(self.model)
                    model_engine = getattr(unwrapped, "engine", None)
                    if model_engine is not None:
                        model_engine.save_checkpoint(str(self.ckpt_dir), tag="best")
                    else:
                        # fallback 保存为普通 torch 权重
                        torch.save(unwrapped.state_dict(), str(self.ckpt_dir / "best_model.pth"))

                    cfg.accelerator.print(
                        f"[Epoch {epoch+1}] New best val_loss={val_loss:.4f}, saved to best_model.pth"
                    )

                epoch_elapsed = time.time() - epoch_start
                avg_train_loss = epoch_loss / len(self.train_dl)
                if cfg.accelerator.is_local_main_process:
                    entry = {
                        "timestamp": datetime.datetime.now().isoformat(),
                        "epoch": epoch + 1,
                        "avg_train_loss": avg_train_loss,
                        "val_loss": val_loss,
                        "val_mae": val_mae,
                    }
                    self._append_history(entry)
                    cfg.accelerator.print(f"[{entry['timestamp']}] epoch {entry['epoch']} "
                        f"train_loss {avg_train_loss:.4f} val_loss {val_loss:.4f} "
                        f"elapsed {epoch_elapsed:.1f}s")
    
                # —— 每个 epoch 末尾保存最新 checkpoint —— 
                # 下次直接从 epoch+1 开始
                if cfg.accelerator.is_local_main_process:
                    unwrapped = cfg.accelerator.unwrap_model(self.model)
                    model_engine = getattr(unwrapped, "engine", None)
                    if model_engine is not None:
                        model_engine.save_checkpoint(str(self.ckpt_dir), tag="best")
                    else:
                        torch.save(unwrapped.state_dict(), str(self.ckpt_dir / "latest_checkpoint.pth"))


                    self._save_checkpoint(epoch + 1) 

                # 如果 EarlyStopping 触发，则跳出
                self.early_stopping(val_loss, self.model, self.ckpt_dir)
                if self.early_stopping.early_stop:
                    cfg.accelerator.print("Early stopping triggered")
                    break
                # 如果是基于 Plateau 的调度（如 ReduceLROnPlateau），用 val_loss，否则无参 step
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
        except Exception as e:
            # 训练崩溃时打印出错 epoch 号和异常
            cfg.accelerator.print(
                f"Training crashed at epoch {self.current_epoch} with error: {e}"
            )
            # 继续抛出，方便外层或日志系统捕获
            raise

    def _save_checkpoint(self, next_epoch: int):
        # 保存模型权重、optimizer、scheduler、RNG（DeepSpeed自动处理
        unwrapped = self.cfg.accelerator.unwrap_model(self.model)
        model_engine = getattr(unwrapped, "engine", None)
        if model_engine is not None:
            model_engine.save_checkpoint(str(self.ckpt_dir))
        else:
            torch.save(unwrapped.state_dict(), str(self.ckpt_dir / "latest_checkpoint.pth"))

        # 手动保存 epoch 信息（用于 resume 时恢复）
        if self.cfg.accelerator.is_local_main_process:
            state_path = self.ckpt_dir / "trainer_state.json"
            with open(state_path, "w") as f:
                json.dump({"next_epoch": next_epoch}, f)

def main() -> None:
    try:
        parser = build_parser()
        args = parser.parse_args()
        setup_environment(args.random_seed)

        # --------- 自动同步混合精度策略 ----------
        import json
        if args.deepspeed:  # 指定了 config
            with open(args.deepspeed, "r") as f:
                ds_cfg = json.load(f)
            if ds_cfg.get("bf16", {}).get("enabled", False):
                args.mixed_precision = "bf16"
            elif ds_cfg.get("fp16", {}).get("enabled", False):
                args.mixed_precision = "fp16"
            else:
                args.mixed_precision = "no"
        # --------- end ----------

        accelerator = get_accelerator(args)
        cfg = TrainerConfig(args=args, accelerator=accelerator)
        trainer = Trainer(cfg)
        trainer.train()
    except Exception as e:
        try:
            cfg.accelerator.print(
                f"Training crashed with error: {e}"
            )
        except Exception:
            print(f"Training crashed with error: {e}")

    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

if __name__ == "__main__":
    main()