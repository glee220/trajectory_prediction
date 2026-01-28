import torch
import numpy as np
import argparse
import time
from torch.utils.data import DataLoader
from dataFactory.data_provider import data_provider  # 假设已实现的数据加载
from utils.tools import EarlyStopping  # 假设已实现的早停机制
from trajectory_predict_train_cvae import CVAEModel  # 根据你的文件结构导入CVAE模型

# 计算 ADE 和 FDE
def compute_ade_fde(predictions, ground_truth, time_horizons=[4, 8, 12]):
    """
    计算 Average Displacement Error (ADE) 和 Final Displacement Error (FDE)
    predictions: 模型预测的轨迹，形状 [batch_size, seq_len, 2] (x, y坐标)
    ground_truth: 真实的轨迹，形状 [batch_size, seq_len, 2] (x, y坐标)
    time_horizons: 计算 ADE 的时刻（单位：秒）
    """
    batch_size, seq_len, _ = predictions.shape

    # 计算 ADE 和 FDE
    ade = []  # 用来存储每个时间点的 ADE
    fde = []  # 用来存储 FDE（最后一步的误差）
    
    for t in time_horizons:
        # 计算每个时间点的 ADE
        ade_t = np.linalg.norm(predictions[:, t-1, :] - ground_truth[:, t-1, :], axis=-1)  # [batch_size,]
        ade.append(ade_t)

    # 计算 FDE（最后一步的误差）
    fde = np.linalg.norm(predictions[:, -1, :] - ground_truth[:, -1, :], axis=-1)  # [batch_size,]

    # 计算均值和标准差
    ade_mean = [np.mean(ae) for ae in ade]  # 计算每个时刻的均值
    ade_std = [np.std(ae) for ae in ade]    # 计算每个时刻的标准差
    fde_mean = np.mean(fde)  # 计算 FDE 的均值
    fde_std = np.std(fde)    # 计算 FDE 的标准差

    # 返回 ADE（均值 ± 标准差）和 FDE（均值 ± 标准差）
    return (ade_mean[0], ade_std[0]), (ade_mean[1], ade_std[1]), (ade_mean[2], ade_std[2]), (fde_mean, fde_std)

# 计算 Miss Rate：最后一步预测距离大于 2 米的比例
def compute_miss_rate(predictions, ground_truth, threshold=2.0):
    """
    计算 Miss Rate：最后一步预测距离大于 threshold 的比例
    predictions: 模型预测的轨迹，形状 [batch_size, seq_len, 2] (x, y坐标)
    ground_truth: 真实的轨迹，形状 [batch_size, seq_len, 2] (x, y坐标)
    threshold: 判断最后一步预测与真实轨迹的最大允许距离（单位：米）
    """
    final_pred = predictions[:, -1, :]  # 预测轨迹的最后一步
    final_gt = ground_truth[:, -1, :]  # 真实轨迹的最后一步

    # 计算最后一步的距离
    distances = np.linalg.norm(final_pred - final_gt, axis=-1)

    # 计算超过阈值的预测比例
    miss_rate = np.mean(distances > threshold)

    return miss_rate

# 计算推理效率：每个样本的平均推理时间（秒）
def compute_inference_efficiency(inference_times):
    """
    计算推理效率：每个样本的平均推理时间（秒）
    inference_times: 每个样本的推理时间（秒）的列表
    """
    return np.mean(inference_times)


class TestModel:
    def __init__(self, model_path, args, batch_size=32, device=None):
        self.model_path = model_path
        self.args = args
        self.use_map = args.use_map
        self.batch_size = batch_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # 加载训练好的模型
        self.cvae_model = self.load_model(args)
        self.cvae_model.to(self.device)
        self.cvae_model.eval()

    def load_model(self, args):
        model = CVAEModel(configs=args, input_dim=2, condition_dim=128, latent_dim=64)  # 根据你的模型配置调整

        # 加载权重
        state_dict = torch.load(self.model_path)

        # 如果不使用 map，则从 state_dict 中删除与 map 相关的权重
        if not self.use_map:
            # 删除与 map_encoder 相关的键
            keys_to_remove = [key for key in state_dict.keys() if "map_encoder" in key]
            for key in keys_to_remove:
                del state_dict[key]

        model.load_state_dict(state_dict)
        return model

    def run_test(self, test_dl):
        predictions = []
        ground_truth = []
        inference_times = []

        with torch.no_grad():
            for batch in test_dl:
                start_time = time.time()

                # 处理输入数据
                batch_agent_vec, batch_nbr_rlpos_mask, batch_nbr_rlpos, batch_nbr_vec_mask, batch_nbr_vec, batch_agent_pos, batch_vehicle_map = batch
                batch_agent_vec_t = batch_agent_vec[:, :self.args.seq_len].to(self.device)

                batch_nbr_rlpos_t = batch_nbr_rlpos[:, :, :self.args.seq_len].to(self.device)
                batch_nbr_rlpos_mask_t = batch_nbr_rlpos_mask[:, :, :self.args.seq_len].to(self.device)        

                batch_nbr_valid = batch_nbr_rlpos_t * batch_nbr_rlpos_mask_t.unsqueeze(-1) 
                
                # 自车未来轨迹
                y_pos = batch_agent_pos[:, -self.args.pred_len:].to(self.device)  
                # 地图
                vehicle_map = batch_vehicle_map.to(self.device)              
                

                # 前向传播，得到预测结果
                predictions_batch, _, _ = self.cvae_model(batch_agent_vec_t, batch_nbr_valid, vehicle_map)

                # 记录推理时间
                inference_times.append(time.time() - start_time)

                # 将预测结果和真实轨迹加入列表
                predictions.append(predictions_batch.cpu().numpy())
                ground_truth.append(y_pos.cpu().numpy())

        # 将所有结果拼接成一个大数组
        predictions = np.concatenate(predictions, axis=0)
        ground_truth = np.concatenate(ground_truth, axis=0)

        # 计算评估指标
        ade_2s, ade_4s, ade_6s, fde = compute_ade_fde(predictions, ground_truth, time_horizons=[4, 8, 12])
        miss_rate = compute_miss_rate(predictions, ground_truth, threshold=2.0)  # Miss Rate for last step > 2 meters
        inference_efficiency = compute_inference_efficiency(inference_times)

        print(f"ADE at 2s: {ade_2s[0]:.4f} ± {ade_2s[1]:.4f}")
        print(f"ADE at 4s: {ade_4s[0]:.4f} ± {ade_4s[1]:.4f}")
        print(f"ADE at 6s: {ade_6s[0]:.4f} ± {ade_6s[1]:.4f}")
        print(f"FDE: {fde[0]:.4f} ± {fde[1]:.4f}")
        print(f"Miss Rate: {miss_rate:.4f}")
        print(f"Inference Efficiency (avg. time per sample): {inference_efficiency:.4f} sec")



# 主函数
def main():
    parser = argparse.ArgumentParser(description="CVAE Trajectory Prediction Testing")
    
    # parser.add_argument("--use_map", type=bool, default=True, help="Whether to use map in the model")
    # parser.add_argument("--model_path", type=str, default="cvae_checkpoints/True_cvae_best_model_20260122_225726.pth", help="Path to the trained model")
    parser.add_argument("--use_map", type=bool, default=False, help="Whether to use map in the model")
    parser.add_argument("--model_path", type=str, default="cvae_checkpoints/False_cvae_best_model_20260123_000220.pth", help="Path to the trained model")
    # parser.add_argument("--model_path", type=str, default="False_cvae_best_model.pth", help="Path to the trained model")
    
    parser.add_argument("--seq_len", type=int, default=4, help="Length of historical sequence")
    parser.add_argument("--pred_len", type=int, default=12, help="Length of prediction sequence")
    parser.add_argument("--root_path", default="./precess_data")
    parser.add_argument("--micro_batch", type=int, default=32, help="samples per GPU, per step")
    parser.add_argument('--num_workers', type=int, default=1, help='data loader num workers')
    args = parser.parse_args()

    # 加载数据
    test_data_set, test_dl = data_provider(args, "test")

    # 创建并运行测试
    tester = TestModel(model_path=args.model_path, args=args, batch_size=args.micro_batch)
    tester.run_test(test_dl)

if __name__ == "__main__":
    main()
