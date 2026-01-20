import os
import argparse
import random
import numpy as np
import torch
from torch import amp
import time

from accelerate import Accelerator, DistributedDataParallelKwargs, DeepSpeedPlugin

from dataFactory.data_provider import data_provider
from models import llm_load
from models.GatedAttnTrajLLM import Model

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Trajectory prediction tester (ADE/FED + 多指标)")

    # ─────────────────────── General ───────────────────────
    parser.add_argument("--task_name", default="trajectory_forecast")
    parser.add_argument(
        "--model_comment",
        default="trajectory_forecast",
        help="训练时使用的 model_comment，用于定位 checkpoint 目录"
    )
    parser.add_argument(
        "--checkpoint_path",
        default="checkpoints/Exp_trajectory_forecast_20250614_172847/best_model.pth",
        help="待加载的模型权重路径，如 ./checkpoints/Exp_xxx/best_model.pth"
    )

    # ─────────────────────── Data ───────────────────────────
    parser.add_argument("--root_path", default="./precess_data", help="数据预处理后文件所在根路径")
    parser.add_argument("--seq_len", type=int, default=4, help="历史轨迹长度")
    parser.add_argument("--pred_len", type=int, default=12, help="预测轨迹长度（总时长 pred_len*0.5s）")
    parser.add_argument("--eval_batch_size", type=int, default=24, help="测试时的 batch size")
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--micro_batch", type=int, default=24, help="samples per GPU, per step")
    # ─────────────────────── Model ──────────────────────────
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--llm_layers", type=int, default=32)
    parser.add_argument("--llm_model", default="LLAMA")
    parser.add_argument("--llm_model_path", default="./models/Llama-2-7b-hf/")
    parser.add_argument("--llm_dim", type=int, default=4096)

    # ─────────────────────── Misc ───────────────────────────
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--mixed_precision",choices=["no", "fp16", "fp32", "bf16"],default="bf16",help="是否启用混合精度")
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument("--use_deepspeed", action="store_true", default=False)
    parser.add_argument("--ds_config", default="./config/ds_config_onslurm.json")
    parser.add_argument('--local_rank', type=int, default=0,help='[auto] provided by deepspeed.launch')

    # ─────────────────────── Ablation ───────────────────────────
    parser.add_argument('--use_map',action='store_true', default=False, help="预测时是否加入地图")

    return parser

def get_accelerator(args) -> Accelerator:
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    ds_plugin = None
    if args.use_deepspeed:
        ds_plugin = DeepSpeedPlugin(hf_ds_config=args.ds_config)
    return Accelerator(
        kwargs_handlers=[ddp_kwargs],
        deepspeed_plugin=ds_plugin,
        mixed_precision=args.mixed_precision
    )

def setup_environment(seed: int) -> None:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def compute_displacements(preds: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    # 计算欧氏距离 L2 误差 [B, pred_len]
    disp = torch.norm(preds - gt, dim=-1)
    return disp

def compute_acceleration(preds: torch.Tensor) -> torch.Tensor:
    # 计算加速度 [B, pred_len-2]
    v = preds[:, 1:, :] - preds[:, :-1, :]
    a = v[:, 1:, :] - v[:, :-1, :]
    acc = torch.norm(a, dim=-1)
    return acc

def compute_curvature(preds: torch.Tensor) -> torch.Tensor:
    # 用三帧位置计算曲率 [B, pred_len-2]
    A = preds[:, :-2, :]
    B = preds[:, 1:-1, :]
    C = preds[:, 2:, :]
    AB = B - A
    BC = C - B
    AC = C - A
    cross = AB[..., 0]*BC[..., 1] - AB[..., 1]*BC[..., 0]
    num = 2 * torch.abs(cross)
    denom = (torch.norm(AB, dim=-1) * torch.norm(BC, dim=-1) * torch.norm(AC, dim=-1) + 1e-8)
    curvature = num / denom
    return curvature

def compute_miss_rate(disp: torch.Tensor, threshold=2.0) -> float:
    # disp: [N, T]，统计最后一步误差超过2m的样本比例
    miss = (disp[:, -1] > threshold).float()
    return miss.mean().item()

def main():
    parser = build_parser()
    args = parser.parse_args()

    # 1. 环境 & 加速器
    setup_environment(args.random_seed)
    accelerator = get_accelerator(args)

    # 2. 构造测试 DataLoader
    test_dataset, test_dl = data_provider(args, "test", accelerator)

    # 3. 构造模型并加载 checkpoint
    llm_model, tokenizer = llm_load(args)
    model = Model(configs=args, llm_model=llm_model, tokenizer=tokenizer)

    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"指定的 checkpoint 不存在: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint)

    # 4. 用加速器包装：模型 + DataLoader
    model, test_dl = accelerator.prepare(model, test_dl)

    # 5. 切换到 eval 模式
    model.eval()

    # 6. 在测试集上逐 batch 计算所有指标
    all_disp_list = []
    all_pred_list = []
    start_time = time.time()  # 开始计时
    with torch.no_grad():
        for batch in test_dl:
            (batch_agent_vec,
             batch_nbr_rlpos_mask,
             batch_nbr_rlpos,
             batch_nbr_vec_mask,
             batch_nbr_vec,
             batch_agent_pos,
             batch_vehicle_map) = batch

            device = accelerator.device
            precision = next(model.parameters()).dtype

            x_agent = batch_agent_vec[:, : args.seq_len].to(device, dtype=precision)
            x_nbr_vec = batch_nbr_vec[:, :, : args.seq_len].to(device, dtype=precision)
            x_nbr_vec_mask = batch_nbr_vec_mask[:, :, : args.seq_len].to(device)
            x_nbr_rlpos = batch_nbr_rlpos[:, :, : args.seq_len].to(device, dtype=precision)
            x_nbr_rlpos_mask = batch_nbr_rlpos_mask[:, :, : args.seq_len].to(device)
            y_pos = batch_agent_pos[:, - args.pred_len :].to(device, dtype=precision)
            vehicle_map = batch_vehicle_map.to(device, dtype=precision)

            use_mp = precision != torch.float32
            ctx = amp.autocast if use_mp else torch.no_grad
            if use_mp:
                caster = ctx("cuda", dtype=precision)
            else:
                caster = ctx()
            with caster:
                if args.use_map:          
                    vehicle_map = batch_vehicle_map.to(device,dtype=precision)
                    # preds,adjusted_trajectory = self.model(
                    preds = model(
                        batch_agent_vec=x_agent,
                        batch_nbr_vec=x_nbr_vec,
                        batch_nbr_vec_mask=x_nbr_vec_mask,
                        batch_nbr_rlpos=x_nbr_rlpos,
                        batch_nbr_rlpos_mask=x_nbr_rlpos_mask,
                        batch_vehicle_map=vehicle_map,
                    )            
                else:                  
                    # preds,adjusted_trajectory = self.model(
                    preds = model(
                        batch_agent_vec=x_agent,
                        batch_nbr_vec=x_nbr_vec,
                        batch_nbr_vec_mask=x_nbr_vec_mask,
                        batch_nbr_rlpos=x_nbr_rlpos,
                        batch_nbr_rlpos_mask=x_nbr_rlpos_mask,
                    ) 

                # import dill
                # data_list = [x_agent,y_pos, preds,vehicle_map,x_nbr_vec,x_nbr_rlpos]
                # data_dict_path = os.path.join('./precess_data/', 'result.pkl')
                # with open(data_dict_path, 'wb') as f:
                #     dill.dump(data_list, f, protocol=dill.HIGHEST_PROTOCOL)
                # print("result.pkl saved!")
                # return 
            disp = compute_displacements(preds, y_pos)
            all_disp_list.append(disp.cpu())
            all_pred_list.append(preds.cpu())
    elapsed_time = time.time() - start_time  # 统计推理总耗时

    # 7. 拼接所有样本的指标
    all_disp = torch.cat(all_disp_list, dim=0)
    all_pred = torch.cat(all_pred_list, dim=0)
    all_disp = all_disp.to(accelerator.device)
    all_disp = accelerator.gather(all_disp)
    all_pred = all_pred.to(accelerator.device)
    all_pred = accelerator.gather(all_pred)

    if accelerator.is_local_main_process:
        all_disp = all_disp.cpu()
        all_pred = all_pred.cpu()
        N, T = all_disp.shape

        # ADE, FDE等传统指标
        mae = all_disp.mean().item()
        last_disp = all_disp[:, -1]
        fde_mean = last_disp.mean().item()
        fde_std = last_disp.std().item()
        disp_2s = all_disp[:, :4]
        mean_2s = disp_2s.mean().item()
        std_2s = disp_2s.std().item()
        disp_4s = all_disp[:, :8]
        mean_4s = disp_4s.mean().item()
        std_4s = disp_4s.std().item()
        disp_6s = all_disp[:, :12]
        mean_6s = disp_6s.mean().item()
        std_6s = disp_6s.std().item()

        # 新增指标
        avg_time_per_sample = elapsed_time / N
        acc = compute_acceleration(all_pred)
        max_acc = acc.max().item()
        mean_acc = acc.mean().item()
        curv = compute_curvature(all_pred)
        max_curv = curv.max().item()
        mean_curv = curv.mean().item()
        miss_rate = compute_miss_rate(all_disp, threshold=2.0)



        # 打印所有指标
        print("===========================================")
        print(f"Total samples evaluated: {N}")
        print(f"MAE (all steps mean): {mae:.6f}")
        print(f"FDE   mean: {fde_mean:.6f}, std: {fde_std:.6f}")
        print(f"2s    mean: {mean_2s:.6f}, std: {std_2s:.6f}")
        print(f"4s    mean: {mean_4s:.6f}, std: {std_4s:.6f}")
        print(f"6s    mean: {mean_6s:.6f}, std: {std_6s:.6f}")
        print("-------------------------------------------")
        print(f"Computational Efficiency: Total {elapsed_time:.3f}s, Per-sample {avg_time_per_sample:.6f}s")
        print(f"Acceleration (L2): mean {mean_acc:.6f}, max {max_acc:.6f}")
        print(f"Curvature continuity: mean {mean_curv:.6f}, max {max_curv:.6f}")
        print(f"Miss Rate (last step >2m): {miss_rate:.6f}")
        print("===========================================")

if __name__ == "__main__":
    main()


