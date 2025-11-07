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

from dataFactory.LazyPickleDataset import LazyPickleDataset

def setup_environment(seed: int) -> None:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"  # 减少内存碎片
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)



def generate_map_description(traj, map_api):
    """
    根据轨迹的终点生成地图的自然语言描述，可用于LLM prompt输入。

    输入：
        traj: numpy array, [T, 2]，轨迹点序列
        map_api: NuScenesMap 对象

    输出：
        map_text: str，地图的自然语言描述
    """
    pt = traj[-1]  # 取终点坐标
    radius = 3.0

    road_info = map_api.get_records_in_radius(pt[0], pt[1], radius, ['road_segment'])
    lane_info = map_api.get_records_in_radius(pt[0], pt[1], radius, ['lane'])

    lane_count = len(set(lane_info.get('lane', [])))
    road_tokens = road_info.get('road_segment', [])

    is_intersection = any(
        map_api.get('road_segment', token).get('is_intersection', False)
        for token in road_tokens
    )

    # 可扩展的地图信息
    desc_parts = []
    desc_parts.append(f"The vehicle is on a road with {lane_count} lane{'s' if lane_count > 1 else ''}.")
    desc_parts.append("It is approaching an intersection." if is_intersection else "There is no intersection ahead.")

    return " ".join(desc_parts)


def build_prompt_with_map(observe_traj, map_api):
    """
    构造一个带地图描述的prompt。

    输入：
        observe_traj: numpy array, [T, 2]，观察轨迹
        map_api: NuScenesMap 对象
    输出：
        str: prompt文本
    """
    map_text = generate_map_description(observe_traj, map_api)
    obs_text = str(observe_traj.tolist())

    prompt = f"""
You are an autonomous driving expert. Your task is to predict the future motion trend based on observed trajectory and map.

Observed trajectory: {obs_text}
Map: {map_text}

What is the motion type? Choose one of [straight, turn].
Answer:
"""
    return prompt
class PromptModel(nn.Module):
    def __init__(self, configs, llm_model, tokenizer):
        super(Model, self).__init__()

        ################## 属性读取
        self.seq_len = configs.seq_len  # observed sequence length
        self.pred_len = configs.pred_len  # predicted sequence length

        self.llm_model = llm_model
        self.tokenizer = tokenizer

        self.d_llm = configs.llm_dim  # LLM 输入token的维度，对于llama来说是4096

        self.d_model = configs.d_model  # 在进行embedding的过程中所考虑的维度
        ################## 设置LLM Tokenizer的 pad_token
        # eos_token (end of sequence) 是一个代表序列结束的特殊字符'</s>', pad_token 作为填充字符统一batch内不同序列长度
        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token
        ################## 初始化prompt
        self.description = 'Given the historical trajectory and local-map of vehicles, predict their future trajectory.'
        ################## 初始化 prototype learning
        # word_embedding, size of (vocabulary size, embedding_dim), a vocabulary of token embeddings
        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]

    def forward(self, batch_agent_vec, batch_nbr_vec, batch_nbr_vec_mask, batch_nbr_rlpos, batch_nbr_rlpos_mask, batch_vehicle_map=None):
        """
        :param batch_agent_vec:
        :param batch_nbr_vec:
        :param batch_nbr_vec_mask:
        :param batch_nbr_rlpos:
        :param batch_nbr_rlpos_mask:
        :return:
        """
        ####################################################### 构造prompt
        prompt = []
        for b in range(batch_agent_vec.shape[0]):
            prompt_ = (f"<|start_prompt|>Dataset description: {self.description}"
                       f"Task description: forecast the next {str(self.pred_len)} steps given the previous {str(self.seq_len)} steps information; "
                       f"<|<end_prompt>|>")  
            prompt.append(prompt_)
        # prompt 转为 token_id (text => token => token_id => embedding vector)
        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        # get_input_embeddings是个(30000,4096)的矩阵，prompt embedding：(BZ, token length, 4090), 代表每个token的embedding
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(batch_agent_vec.device))  # (batch, prompt_token_num, dim)
            
        with amp.autocast("cuda", dtype=torch.float16):        # ① 暂停混合精度  
            llm_raw = self.llm_model(                    #    推理 (FP32 GEMM)  
                inputs_embeds        = prompt_embeddings,
                output_hidden_states = True,
                return_dict          = True,
            )
        print(llm_raw)

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
    parser.add_argument("--llm_model", default="LLAMA3")
    parser.add_argument("--llm_model_path" , default='./models/Meta-Llama-3-8B-Instruct/')  
    
    # ─────────────────────── Optim ──────────────────────────
    parser.add_argument('--num_workers', type=int, default=4, help='data loader num workers')
    parser.add_argument("--train_epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5, help='early stopping patience')
    parser.add_argument("--learning_rate", type=float, default=5e-5)                                
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
    # ─────────────────────── Prompt ───────────────────────────
    parser.add_argument('--prompt_type',choices=["scene-tokens", "zero_shot", "In-context_Learning", "Chain-of-Thought"], default="zero_shot")

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
        # self.model = Model(configs=cfg.args, llm_model=llm, tokenizer=tokenizer)
        self.model = PromptModel(configs=cfg.args, llm_model=llm, tokenizer=tokenizer)
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
        args  = self.cfg.args 
        train_data= LazyPickleDataset(
            pkl_path=f'{args.root_path}/type_train.pkl',
            index_path=f'{args.root_path}/type_train.index'
        )
        val_data   = LazyPickleDataset(
            pkl_path=f'{args.root_path}/type_val.pkl',
            index_path=f'{args.root_path}/type_val.index'
        )

        return train_data, val_data

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



zero_shot = """
You are an expert in autonomous driving. Given the observed trajectory of a vehicle and its surrounding road environment, predict the next 12 future positions of the vehicle at 0.5-second intervals (2Hz sampling).

Observed trajectory (past 2 seconds, sampled at 2Hz):  
[[x1, y1], [x2, y2], [x3, y3], [x4, y4]]

Map description:  
The vehicle is currently on a 3-lane straight road with lane markings. There is a slight curve ahead to the right. No intersections nearby.

Predict the future trajectory for the next 6 seconds (12 points), sampled at 2Hz.  
Output format: [[x5, y5], [x6, y6], ..., [x16, y16]]

Answer:
"""


In-context_Learning = """
You are an autonomous driving model. Predict the vehicle's future trajectory based on the observed trajectory and the map information.

Example 1:  
Observed trajectory: [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [1.5, 0.0]]  
Map: A straight single-lane road.  
Future trajectory: [[2.0, 0.0], [2.5, 0.0], [3.0, 0.0], [3.5, 0.0], [4.0, 0.0], [4.5, 0.0], [5.0, 0.0], [5.5, 0.0], [6.0, 0.0], [6.5, 0.0], [7.0, 0.0], [7.5, 0.0]]

Example 2:  
Observed trajectory: [[1.0, 1.0], [1.3, 1.3], [1.6, 1.6], [1.9, 1.9]]  
Map: A curved road to the left, with 2 lanes.  
Future trajectory: [[2.2, 2.1], [2.4, 2.3], [2.6, 2.5], [2.7, 2.7], [2.8, 2.9], [2.9, 3.1], [3.0, 3.3], [3.1, 3.5], [3.2, 3.7], [3.3, 3.9], [3.4, 4.1], [3.5, 4.3]]

Now your turn:  
Observed trajectory: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]  
Map: {Insert your generated map description here}

Future trajectory (12 points at 2Hz):  
Answer:
"""

Chain-of-Thought ="""
You are a reasoning agent for autonomous vehicle motion forecasting.

Task: Given a 2-second observed trajectory (sampled at 2Hz, 4 points), and a map description, reason step-by-step and then predict the next 6 seconds (12 future points) of the vehicle trajectory at 2Hz.

Observed trajectory: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]

Map: The road is a 3-lane highway with a gentle right turn. No intersections nearby.

Let's think step by step:
1. Estimate the current direction and speed of the vehicle.
2. Use the road curvature and direction to project the motion.
3. Generate the 12 future positions sampled every 0.5 seconds.

Answer:
"""