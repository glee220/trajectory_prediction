import os
import argparse
import random
import time
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from pathlib import Path
from utils.tools import EarlyStopping  # 假设已经有的早停机制
from dataFactory.data_provider import data_provider  # 假设已实现的数据加载
from utils.init_weight import init_weights
import datetime

# 设置训练环境
def setup_environment(seed: int):
    # 设置 CUDA 可见设备，限制为使用 CUDA 设备 2 或 3
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # 或者 "3"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"  # 减少内存碎片
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# 数据加载器
def build_dataloaders(args):
    train_data_set, train_dl = data_provider(args, "train")
    val_data_set, val_dl = data_provider(args, "val")
    return train_dl, val_dl

def _ckpt_dir(self, ts) -> Path:
    p = Path("./cvae_checkpoints") / f"Exp_{self.args.use_map}_{ts}"
    p.mkdir(parents=True, exist_ok=True)
    return p

# ------------------


MIN_Feat_DIM = 8
class AgentEmbedding(nn.Module):
    def __init__(self, in_channel: int, out_channel: int) -> None:
        super(AgentEmbedding, self).__init__()

        # 先把 in_channel 映射到 MIN_Feat_DIM（例如：8）
        self.preproj = nn.Linear(in_channel, MIN_Feat_DIM, bias=False)

        # 然后通过一系列线性层处理数据
        self.embed = nn.Sequential(
            nn.Linear(MIN_Feat_DIM, out_channel),  # 映射到目标维度
            nn.LayerNorm(out_channel),
            nn.ReLU(inplace=True),
            nn.Linear(out_channel, out_channel),
            nn.LayerNorm(out_channel),
            nn.ReLU(inplace=True),
            nn.Linear(out_channel, out_channel),
            nn.LayerNorm(out_channel)
        )
        
        # 防止出现nan值，应用初始化方法
        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.preproj(x)  # 通过线性层进行转换
        return self.embed(x)  # 通过后续的 embed 层处理


class SurroundingAgentEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_agents=100, seq_len=4, dropout=0.2):
        """
        :param input_dim: 每个时间步的输入特征数，例如 [x, y] 的维度
        :param hidden_dim: LSTM隐藏层的维度，用于处理时序特征
        :param output_dim: 输出的维度，即最终编码的潜在空间维度
        :param num_agents: 周车数量
        :param seq_len: 输入的时间步长，轨迹长度
        :param dropout: Dropout比率，用于防止过拟合
        """
        super(SurroundingAgentEncoder, self).__init__()

        self.seq_len = seq_len
        self.input_dim = input_dim
        self.num_agents = num_agents

        # 使用一个LSTM来处理轨迹数据的时序特征
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=2, batch_first=True, dropout=dropout)

        # 输出层，映射到潜在空间
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),  # 映射到隐空间
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)  # 输出潜在特征空间
        )

        # 通过加权求和或注意力机制对周车的轨迹进行聚合
        self.attention = nn.Sequential(
            nn.Linear(output_dim, 1),
            nn.Softmax(dim=1)  # 对每辆车的特征进行注意力加权
        )

    def forward(self, x):
        """
        :param x: 输入的周车轨迹，形状为 [batch_size, num_agents, seq_len, input_dim]
        :return: 编码后的特征表示，形状为 [batch_size, seq_len, output_dim]
        """
        batch_size, num_agents, seq_len, input_dim = x.shape

        # 初始化一个空的张量来存储每个时间步的特征
        time_step_embeddings = []

        # 对每个时间步进行处理
        for t in range(seq_len):
            # 选择当前时间步的周车轨迹，形状 [batch_size, num_agents, input_dim]
            time_step_trajectory = x[:, :, t, :]  # 形状 [batch_size, num_agents, input_dim]

            # 处理每个周车的轨迹
            agent_embeddings = []
            for i in range(num_agents):
                agent_trajectory = time_step_trajectory[:, i, :]  # 形状 [batch_size, input_dim]
                lstm_out, _ = self.lstm(agent_trajectory.unsqueeze(1))  # lstm_out 形状 [batch_size, 1, hidden_dim]
                final_hidden_state = lstm_out[:, -1, :]  # 取最后一个时间步的输出 [batch_size, hidden_dim]
                agent_embeddings.append(self.fc(final_hidden_state))  # 映射到潜在空间

            # 将所有周车的编码合并成一个张量，形状 [batch_size, num_agents, output_dim]
            agent_embeddings = torch.stack(agent_embeddings, dim=1)

            # 对周车的编码进行注意力加权，获得一个 [batch_size, output_dim] 形状的张量
            attention_weights = self.attention(agent_embeddings)  # 形状 [batch_size, num_agents, 1]
            weighted_embeddings = agent_embeddings * attention_weights  # 加权合并
            aggregated_embedding = weighted_embeddings.sum(dim=1)  # 聚合周车特征，形状 [batch_size, output_dim]

            time_step_embeddings.append(aggregated_embedding.unsqueeze(1))  # 将当前时间步的特征加入列表

        # 将每个时间步的特征拼接成最终的形状 [batch_size, seq_len, output_dim]
        time_step_embeddings = torch.cat(time_step_embeddings, dim=1)

        return time_step_embeddings
    
class CVAEEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim=256):
        """
        :param input_dim: 输入的维度，通常为自车轨迹和周车轨迹拼接后的维度
        :param latent_dim: 潜在空间的维度
        :param hidden_dim: 编码器的隐藏层维度
        """
        super(CVAEEncoder, self).__init__()

        # 编码器的全连接层
        self.fc1 = nn.Linear(input_dim*2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)  # 输出均值
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)  # 输出对数方差

        # 激活函数
        self.relu = nn.ReLU()


    def forward(self, h):
        """
        :param h: 输入的拼接特征，形状 [batch_size, input_dim + condition_dim]
        :return: 潜在空间的均值mu和对数方差logvar
        """
        # 通过全连接层并应用激活函数
        x = self.relu(self.fc1(h))
        x = self.relu(self.fc2(x))

        # 输出均值和对数方差
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar


class EnhancedMapEncoder(nn.Module):
    def __init__(self, map_channels=3, d_model=128, cvae_dim=256):
        super().__init__()
        # CNN部分保持输出d_model=128
        self.conv_layers = nn.Sequential(
            nn.Conv2d(map_channels, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        self.fc = nn.Sequential(
            nn.Linear(512*4*4, d_model),
            nn.ReLU(),
            nn.Linear(d_model, cvae_dim)  # 最终输出匹配的维度
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
    
class LSTMDecoder(nn.Module):
    def __init__(self, latent_dim, condition_dim, hidden_dim=128, output_dim=2, seq_len=12, dropout_rate=0.1):
        super(LSTMDecoder, self).__init__()

        # 拼接后的输入通道数
        self.input_dim = latent_dim + condition_dim  # latent_dim + condition_dim
        self.seq_len = seq_len  # 预测的时间步长
        
        # LSTM 解码器
        self.lstm = nn.LSTM(input_size=self.input_dim, hidden_size=hidden_dim, batch_first=True, dropout=dropout_rate)
        
        # 输出层：每个时间步输出 2 个坐标（x, y）
        self.fc = nn.Linear(hidden_dim, output_dim)  # output_dim = 2 (x, y)

    def forward(self, z, condition):
        """
        :param z: 潜在变量，形状为 [batch_size, 4, latent_dim]
        :param condition: 条件信息，形状为 [batch_size, 4, condition_dim]
        :return: 预测的轨迹，形状为 [batch_size, 12, 2]（x, y坐标）
        """
        # 将潜在变量 z 和条件信息 condition 拼接
        z_cond = torch.cat([z, condition], dim=-1)  # 拼接后的形状 [batch_size, 4, latent_dim + condition_dim]

        # LSTM 处理
        lstm_out, _ = self.lstm(z_cond)  # 输出形状 [batch_size, 4, hidden_dim]

        # 使用全连接层生成每个时间步的 (x, y) 坐标
        # 我们希望生成 12 个时间步的轨迹
        lstm_out = lstm_out[:, -1, :]  # 取 LSTM 输出的最后一个时间步的隐藏状态 [batch_size, hidden_dim]

        # 通过全连接层生成每个时间步的 (x, y) 坐标
        recon_x = self.fc(lstm_out)  # 输出形状 [batch_size, 2]

        # 通过 LSTM 和全连接层扩展到 12 个时间步
        recon_x = recon_x.unsqueeze(1).repeat(1, self.seq_len, 1)  # 形状变为 [batch_size, 12, 2]

        return recon_x

class FeatureFusion(nn.Module):
    def __init__(self):
        super(FeatureFusion, self).__init__()

    def forward(self, map_embeddings, nbr_embeddings):
        """
        :param map_embeddings: 形状为 [batch_size, 128]，地图特征
        :param nbr_embeddings: 形状为 [batch_size, 4, 128]，周车特征
        :return: 融合后的特征，形状为 [batch_size, 4, 128]
        """
        # 扩展 map_embeddings 为 [batch_size, 4, 128]
        map_embeddings_expanded = map_embeddings.unsqueeze(1).expand(-1, 4, -1)  # 形状变为 [batch_size, 4, 128]

        # 进行融合，这里使用加法（您也可以使用拼接）
        fused_embeddings = map_embeddings_expanded + nbr_embeddings  # 形状为 [batch_size, 4, 128]

        return fused_embeddings

# ------------------
# CVAE模型实现
class CVAEModel(nn.Module):
    def __init__(self, configs, input_dim, condition_dim, latent_dim):
        super(CVAEModel, self).__init__()
        self.use_map = configs.use_map
        self.d_CVAE = input_dim

        if self.use_map:
                    
            ################## 对map_encoder 进行初始化
            # 修改地图编码器初始化
            self.map_encoder = EnhancedMapEncoder(d_model=input_dim, cvae_dim=condition_dim) 

        # 编码器（Encoder）
        self.agent_encoder = AgentEmbedding(in_channel=input_dim, out_channel=condition_dim)
        self.nbr_encoder = SurroundingAgentEncoder(input_dim=input_dim, hidden_dim=condition_dim, output_dim=condition_dim, seq_len=configs.seq_len)

        self.cvae_encoder = CVAEEncoder(input_dim=condition_dim, latent_dim=latent_dim)
        # 解码器（Decoder）
        self.decoder = LSTMDecoder(latent_dim=latent_dim, condition_dim=condition_dim)
        # 统一特征融合层
        self.feature_fusion = FeatureFusion()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, nbr, map):
        x = self.agent_encoder(x)
        if self.use_map :
            map_embeddings = self.map_encoder(map)   
            nbr_embeddings = self.nbr_encoder(nbr)     
            condition = self.feature_fusion(map_embeddings, nbr_embeddings)
        else:            
            nbr_embeddings = self.nbr_encoder(nbr)
            condition = nbr_embeddings

        h = torch.cat([x, condition], dim=-1)  # 拼接自车轨迹和条件（如周车轨迹、地图）

        mu, logvar = self.cvae_encoder(h)  # 均值和对数方差 直接接收返回的 mu 和 logvar

        z = self.reparameterize(mu, logvar)  # 使用reparameterization trick
        recon_x = self.decoder(z, condition)  # 生成预测轨迹
        return recon_x, mu, logvar

# Trainer 类
class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化CVAE模型
        self.cvae_model = CVAEModel(configs=self.args, input_dim=2, condition_dim=128, latent_dim=64)  # 输入维度，条件维度，潜在维度
        self.cvae_model.to(self.device)
        
        # 优化器
        self.optimizer = optim.Adam(self.cvae_model.parameters(), lr=self.args.learning_rate)
        self.early_stopping = EarlyStopping(patience=self.args.patience)
        
        # 加载数据
        self.train_dl, self.val_dl = build_dataloaders(self.args)

    def train_step(self, batch):
        """
        :param batch: 输入的 batch 数据
        :return: 计算损失
        """
        (batch_agent_vec, batch_nbr_rlpos_mask, batch_nbr_rlpos, batch_nbr_vec_mask, batch_nbr_vec, batch_agent_pos, batch_vehicle_map) = batch

        args = self.args
        device = self.device

        batch_agent_vec_t = batch_agent_vec[:, :args.seq_len].to(device)
        # batch_nbr_vec_t = batch_nbr_vec[:, :, :args.seq_len].to(device)
        # batch_nbr_vec_mask_t = batch_nbr_vec_mask[:, :, :args.seq_len].to(device)

        batch_nbr_rlpos_t = batch_nbr_rlpos[:, :, :args.seq_len].to(device)
        batch_nbr_rlpos_mask_t = batch_nbr_rlpos_mask[:, :, :args.seq_len].to(device)        

        batch_nbr_valid = batch_nbr_rlpos_t * batch_nbr_rlpos_mask_t.unsqueeze(-1) 
        
        # 自车未来轨迹
        y_pos = batch_agent_pos[:, -args.pred_len:].to(device)  
        # 地图
        vehicle_map = batch_vehicle_map.to(device)

        # 前向传播：将自车数据（x_agent）和条件（condition）传递给CVAE模型
        recon_x, mu, logvar = self.cvae_model(batch_agent_vec_t, batch_nbr_valid, vehicle_map)

        # 计算L2损失（重构误差）和KL散度（潜在空间正则化）
        loss = F.mse_loss(recon_x, y_pos)  # L2损失
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())  # KL散度
        total_loss = loss + kl_divergence

        return total_loss.item()

    def evaluate(self, dataloader):
        self.cvae_model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in dataloader:
                loss = self.train_step(batch)
                total_loss += loss
        return total_loss / len(dataloader)

    def train(self):
        best_val_loss = float('inf')
    
        # 在训练开始时生成一个固定的时间戳
        training_start_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_path = Path("./cvae_checkpoints/")
        base_path.mkdir(parents=True, exist_ok=True)
    

        for epoch in range(self.args.train_epochs):
            self.cvae_model.train()
            epoch_loss = 0
            for step, batch in enumerate(self.train_dl):
                loss = self.train_step(batch)
                epoch_loss += loss
                if (step + 1) % 100 == 0:
                    print(f"Epoch {epoch+1}, Step {step+1}, Loss: {loss:.4f}")
        # ————————————————————————————
                avg_train_loss = epoch_loss / len(self.train_dl)
                val_loss = self.evaluate(self.val_dl)

                print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    print(f'{self.args.use_map}_cvae_best_model.pth')
                    torch.save(self.cvae_model.state_dict(), f'{self.args.use_map}_cvae_best_model.pth')

                    # 使用训练开始的时间戳保存带时间戳的模型
                    checkpoint_path = base_path / f"{self.args.use_map}_cvae_best_model_{training_start_time}.pth"
                    torch.save(self.cvae_model.state_dict(), checkpoint_path)
                    
                    print(f"Best model updated and saved: {checkpoint_path}")

                # 调用 early_stopping
                self.early_stopping(val_loss, self.cvae_model, checkpoint_path)

                if self.early_stopping.early_stop:
                    print("Early stopping triggered")
                    break
        # ————————————————————————————
        
        # 训练结束后，可以输出最终保存的模型路径
        print(f"Training completed. Best model saved with timestamp: {training_start_time}")

# 主函数
def main():
    parser = argparse.ArgumentParser(description="CVAE Trajectory Prediction")
    parser.add_argument("--train_epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience") 
    parser.add_argument("--seq_len", type=int, default=4, help="Length of historical sequence")
    parser.add_argument("--pred_len", type=int, default=12, help="Length of prediction sequence")
    parser.add_argument("--use_map", default=True, help="Use map as condition input")

    parser.add_argument("--root_path", default="./precess_data")
    parser.add_argument("--micro_batch", type=int, default=32, help="samples per GPU, per step")
    parser.add_argument('--num_workers', type=int, default=1, help='data loader num workers')
    args = parser.parse_args()

    setup_environment(seed=42)

    # 初始化训练器并开始训练
    trainer = Trainer(args)
    trainer.train()

if __name__ == "__main__":
    main()
