from math import sqrt
import transformers
transformers.logging.set_verbosity_error()
from utils.init_weight import init_weights
import torch
from torch import nn
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast
from torch import amp
from transformers import ViTFeatureExtractor, ViTModel

from torchvision import transforms


class GateAttnSceneEmbedder(nn.Module):
    def __init__(self, dropout_rate=0.1, num_head=8,
                 agent_input_dim=2, nbr_input_dim=4, embed_dim=128,
                 obs_len=4, pred_len=12):
        """

        :param dropout_rate:
        :param num_head:
        :param agent_input_dim:
        :param nbr_input_dim:
        """
        super(GateAttnSceneEmbedder, self).__init__()

        ########## 属性赋值
        self.dropout_rate = dropout_rate
        self.num_head = num_head
        self.agent_input_dim = agent_input_dim
        self.nbr_input_dim = nbr_input_dim
        self.embed_dim = embed_dim
        self.obs_len = obs_len
        self.pred_len = pred_len
        if self.embed_dim % self.num_head != 0:
            raise ValueError('embed_dim must be divisible by num_head')

        ########## 创建用于占位的 learnable embedding NA_token_seq
        self.NA_token_seq = nn.Parameter(torch.Tensor(self.obs_len, self.embed_dim))
        # normal_用于对 NA_token_seq 的值进行初始化，没有此操作的话，NA_token_seq 中会出现nan
        nn.init.normal_(self.NA_token_seq, mean=0., std=.02)

        ########## 初始化各种网络模块组件
        self.agent_embedder = AgentEmbedding(self.agent_input_dim, self.embed_dim)
        self.nbr_embedder = AgentEmbedding(self.nbr_input_dim, self.embed_dim)
        self.layer_norm_1 = nn.LayerNorm(self.embed_dim)
        self.layer_norm_2 = nn.LayerNorm(self.embed_dim)
        self.lin_q = nn.Linear(self.embed_dim, self.embed_dim)
        self.lin_k = nn.Linear(self.embed_dim, self.embed_dim)
        self.lin_v = nn.Linear(self.embed_dim, self.embed_dim)
        self.attn_drop = nn.Dropout(self.dropout_rate)
        self.mlp = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.embed_dim * 4, self.embed_dim),
            nn.Dropout(self.dropout_rate))

        self.lin_ih = nn.Linear(self.embed_dim, self.embed_dim)
        self.lin_hh = nn.Linear(self.embed_dim, self.embed_dim)
        self.lin_self = nn.Linear(self.embed_dim, self.embed_dim)
        ########## 对网络模块组件进行初始化
        self.apply(init_weights)

    def forward(self, batch_agent_vec, batch_nbr_vec, batch_nbr_vec_mask, batch_nbr_rlpos, batch_nbr_rlpos_mask):

        """
        :param batch_agent_vec: tensor of shape [batch_size, obs_len, 2], 此处第3维的2代表central agent沿着x,y两个方向的位移向量
        :param batch_nbr_vec: tensor of shape [batch_size, MaxNbrNum, obs_len, 2], 此处MaxNbrNum=100, 代表能考虑周边最多的agent，第四维度的2代表nbr agent沿着x,y两个方向的位移向量
        :param batch_nbr_vec_mask: tensor of shape [batch_size, MaxNbrNum, obs_len], 确定batch_nbr_vec中无效值位置
        :param batch_nbr_rlpos: tensor of shape [batch_size, MaxNbrNum, obs_len, 2], 第四维度的2代表nbr agent和central agent的沿着xy轴相对位置
        :param batch_nbr_rlpos_mask: tensor of shape [batch_size, MaxNbrNum, obs_len], 确定batch_nbr_rlpos中无效值位置
        :return:
        """

        self.batch_size, self.MaxNbrNum, obs_len, _ = batch_nbr_vec.shape
        if obs_len != self.obs_len:
            raise ValueError('observe length of input sequence must be equal to pre-defined observe length')

        ########## central agent的embedding
        # [batch_size, obs_len, 2] -> [batch_size, obs_len, embed_dim]
        batch_agent_embeddings = self.agent_embedder(batch_agent_vec)
        # 用learnable token填充agent embedding的首个时刻
        batch_agent_embeddings[:, 0, :] = self.NA_token_seq[0]  # 它里面有NaN

        ########## neighboring agent的embedding
        # [batch_size, MaxNbrNum, obs_len, nbr_input_dim (2+2)] -> [batch_size, MaxNbrNum, obs_len, embed_dim]
        batch_nbr_embeddings = self.nbr_embedder(torch.cat([batch_nbr_vec, batch_nbr_rlpos], dim=-1))
        # 扩展复制，[obs_Len, embed_dim] -> [MaxNbrNum, obs_len, embed_dim]
        batch_NA_token_seq = torch.repeat_interleave(self.NA_token_seq.unsqueeze(0), repeats=self.MaxNbrNum, dim=0)
        # 扩展复制，[MaxNbrNum, obs_Len, embed_dim] -> [MaxNbrNum, obs_len, embed_dim]
        batch_NA_token_seq = torch.repeat_interleave(batch_NA_token_seq.unsqueeze(0), repeats=self.batch_size, dim=0)
        # 建立nbr embeddings的mask [BZ, MaxAgentNum, obs_len]，必须同时满足有rlpos和vec
        batch_nbr_embed_mask = batch_nbr_vec_mask * batch_nbr_rlpos_mask
        # 扩展复制为 [BZ, MaxAgentNum, obs_len, 128]
        batch_nbr_embed_mask = torch.repeat_interleave(input=batch_nbr_embed_mask.unsqueeze(-1), repeats=128,
                                                       dim=-1)  # [1][31][1]
        # -> [batch_size, MaxNbrNum, obs_len, embed_dim], 将batch_nbr_embeddings中无效的区域填充为NA_token
        batch_nbr_embeddings = (batch_nbr_embeddings * batch_nbr_embed_mask) + (
                (1 - batch_nbr_embed_mask) * batch_NA_token_seq)

        ########## 进行Attention操作

        # x先residual connection, 再norm，再MHA
        batch_agent_embeddings = torch.add(batch_agent_embeddings , self._mha_(self.layer_norm_1(batch_agent_embeddings), batch_nbr_embeddings) )
        batch_agent_embeddings = torch.add(batch_agent_embeddings , batch_agent_embeddings + self._ff_block(self.layer_norm_2(batch_agent_embeddings)))

        return batch_agent_embeddings

    def _mha_(self, batch_agent_embeddings, batch_nbr_embeddings):

        ########## 计算query
        # [BZ, TimeHorizon, embed_dim] -> [BZ, TimeHorizon, num_head, head_dim]
        query = self.lin_q(batch_agent_embeddings).view(self.batch_size, -1, self.num_head,
                                                        self.embed_dim // self.num_head)
        # [BZ, TimeHorizon, num_head, head_dim] -> [batch_size, obs_len, MaxNbrNum, num_head, head_dim]
        query = torch.repeat_interleave(input=query.unsqueeze(1), repeats=self.MaxNbrNum, dim=1).transpose(1, 2)

        ########## 计算key
        # [batch_size, MaxNbrNum, obs_len, embed_dim] -> [batch_size, obs_len, MaxNbrNum, num_head, head_dim]
        key = self.lin_k(batch_nbr_embeddings).view(self.batch_size, self.MaxNbrNum, -1, self.num_head,
                                                    self.embed_dim // self.num_head).transpose(1, 2)

        ########## 计算value
        # [batch_size, MaxNbrNum, obs_len, embed_dim] -> [batch_size, obs_len, MaxNbrNum, num_head, head_dim]
        value = self.lin_v(batch_nbr_embeddings).view(self.batch_size, self.MaxNbrNum, -1, self.num_head,
                                                      self.embed_dim // self.num_head).transpose(1, 2)

        ########## 计算scale
        scale = (self.embed_dim // self.num_head) ** 0.5 + 1e-6 

        ########## 计算scale，计算attention map
        # -> [batch_size, obs_len, MaxNbrNum, num_head], 先做element-wise product，再去做sum，就相当于转置相乘
        alpha = (query * key).sum(dim=-1) / scale
        # [batch_size, obs_len, MaxNbrNum, num_head] -> [batch_size, obs_len, MaxNbrNum, num_head]
        alpha = nn.functional.softmax(input=alpha, dim=-2)
        # alpha of shape [BZ, TimeHorizon, MaxNodeNum, AttnHeadNum]
        alpha = self.attn_drop(alpha)

        ######## dot product
        # [batch_size, obs_len, MaxNbrNum, num_head, head_dim] -> [batch_size, obs_len, num_head, head_dim], 对NbrAgent的信息进行聚合
        # agr denotes aggregated
        batch_nbr_agg_embeddings = (alpha.unsqueeze(-1) * value).sum(dim=-3)
        # [batch_size, obs_len, num_head, head_dim] ->  [batch_size, obs_len, embed_dim]
        batch_nbr_agg_embeddings = batch_nbr_agg_embeddings.view(self.batch_size, self.obs_len, self.embed_dim)

        # -> [batch_size, obs_len, embed_dim]
        gate = torch.sigmoid(self.lin_ih(batch_nbr_agg_embeddings) + self.lin_hh(batch_agent_embeddings))

        return gate * self.lin_self(batch_agent_embeddings) + (1 - gate) * batch_nbr_agg_embeddings

    def _ff_block(self, x):
        return self.mlp(x)
    

class EnhancedMapEncoder(nn.Module):
    def __init__(self, map_channels=3, d_model=128, llm_dim=4096):
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
        
        # 维度转换：512*4*4 → d_model → llm_dim
        self.fc = nn.Sequential(
            nn.Linear(512*4*4, d_model),
            nn.ReLU(),
            nn.Linear(d_model, llm_dim)  # 最终输出匹配LLM的4096维
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

# 交叉注意力模块，让轨迹特征主动查询地图特征
class EnhancedCrossModalAttention(nn.Module):
    def __init__(self, d_model=4096, n_heads=8, llm_dim=4096):
        super().__init__()
        self.d_model = d_model
        self.llm_dim = llm_dim
        
        # Since both dimensions are now 4096, we can simplify the projections
        self.traj_projection = nn.Sequential(
            nn.Linear(llm_dim, llm_dim),
            nn.LayerNorm(llm_dim),
            nn.GELU()
        )
        
        self.map_proj = nn.Sequential(
            nn.Linear(llm_dim, llm_dim),
            nn.LayerNorm(llm_dim)
        )
        
        # Both attention modules now use llm_dim
        self.traj_to_map_attn = nn.MultiheadAttention(
            embed_dim=llm_dim,
            num_heads=n_heads,
            batch_first=True
        )
        
        self.map_to_traj_attn = nn.MultiheadAttention(
            embed_dim=llm_dim,
            num_heads=n_heads,
            batch_first=True
        )
        
        self.fusion_gate = nn.Sequential(
            nn.Linear(llm_dim * 2, llm_dim),
            nn.Sigmoid()
        )

    def forward(self, traj_feat, map_feat):
        B, T, _ = traj_feat.shape
        
        # 1. Project trajectory features
        traj_proj = self.traj_projection(traj_feat)  # [B, T, llm_dim]
        
        # 2. Trajectory-to-Map attention
        map_aware_traj, _ = self.traj_to_map_attn(
            query=map_feat,
            key=traj_proj,
            value=traj_proj
        )  # [B, 1, llm_dim]
        
        # 3. Map-to-Trajectory attention
        map_proj = self.map_proj(map_feat)  # [B, 1, llm_dim]
        traj_aware_map, _ = self.map_to_traj_attn(
            query=traj_feat,  # [B, T, llm_dim]
            key=map_proj.expand(-1, T, -1),
            value=map_proj.expand(-1, T, -1)
        )  # [B, T, llm_dim]
        
        # 4. Gated fusion
        gate_input = torch.cat([
            traj_feat,
            map_aware_traj.expand(-1, T, -1)
        ], dim=-1)
        gate = self.fusion_gate(gate_input)
        fused_feat = gate * traj_aware_map + (1-gate) * traj_feat
        
        return fused_feat  # [B, T, llm_dim]
    

class CVAEModel(nn.Module):
    def __init__(self, input_dim, condition_dim, latent_dim):
        """
        :param input_dim: 输入维度，通常是自车和周车历史轨迹的特征维度
        :param condition_dim: 条件维度，地图信息的维度
        :param latent_dim: 潜在空间的维度
        """
        super(CVAEModel, self).__init__()

        # 编码器（Encoder）部分
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + condition_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim * 2)  # 输出均值和对数方差
        )

        # 解码器（Decoder）部分
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)  # 生成预测的轨迹
        )

    def reparameterize(self, mu, logvar):
        """
        使用 reparameterization trick 生成潜在变量
        :param mu: 均值
        :param logvar: 对数方差
        :return: 潜在变量
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, condition):
        """
        :param x: 输入数据（自车和周车历史轨迹）
        :param condition: 条件数据（地图）
        :return: 生成的轨迹、均值、对数方差
        """
        # 编码输入和条件
        h = torch.cat([x, condition], dim=-1)  # 将输入和条件拼接
        mu, logvar = self.encoder(h).chunk(2, dim=-1)  # 均值和对数方差
        z = self.reparameterize(mu, logvar)  # 使用reparameterization trick

        # 解码
        z_cond = torch.cat([z, condition], dim=-1)  # 将潜在变量和条件拼接
        recon_x = self.decoder(z_cond)  # 生成预测轨迹

        return recon_x, mu, logvar




class Model(nn.Module):
    def __init__(self, configs, cvae_model: nn.Module):
        super(Model, self).__init__()

        ################## 属性读取
        self.seq_len = configs.seq_len  # observed sequence length
        self.pred_len = configs.pred_len  # predicted sequence length

        self.cvae_model = cvae_model  # 使用CVAE模型

        self.d_model = configs.d_model  # 在进行embedding的过程中所考虑的维度
        self.dropout_rate = configs.dropout
        self.n_heads = configs.n_heads

        ################## 初始化encoder
        self.encoder = GateAttnSceneEmbedder(dropout_rate=self.dropout_rate,
                                             num_head=8, agent_input_dim=2, nbr_input_dim=4,
                                             embed_dim=self.d_model, obs_len=self.seq_len, pred_len=self.pred_len)

        self.use_map = configs.use_map
        if self.use_map:
            ################## 对map_encoder 进行初始化
            self.map_encoder = EnhancedMapEncoder(
                d_model=configs.d_model,
                llm_dim=configs.llm_dim
            )

            # 交叉注意力部分保留
            self.cross_attn = EnhancedCrossModalAttention(
                d_model=configs.llm_dim,
                n_heads=configs.n_heads,
                llm_dim=configs.llm_dim
            )

            # 特征融合层
            self.feature_fusion = nn.Sequential(
                nn.Linear(2 * configs.llm_dim, configs.llm_dim),
                nn.LayerNorm(configs.llm_dim)
            )

    def forward(self, batch_agent_vec, batch_nbr_vec, batch_nbr_vec_mask, batch_nbr_rlpos, batch_nbr_rlpos_mask, batch_vehicle_map=None):
        """
        :param batch_agent_vec: [batch_size, seq_len, 2] 自车历史轨迹
        :param batch_nbr_vec: [batch_size, MaxNbrNum, seq_len, 2] 周车历史轨迹
        :param batch_nbr_vec_mask: [batch_size, MaxNbrNum, seq_len] 周车轨迹的mask
        :param batch_nbr_rlpos: [batch_size, MaxNbrNum, seq_len, 2] 周车相对位置
        :param batch_nbr_rlpos_mask: [batch_size, MaxNbrNum, seq_len] 周车位置mask
        :param batch_vehicle_map: [batch_size, map_channels, map_height, map_width] 地图信息
        :return: self.pred_len 未来轨迹的预测
        """
        # 轨迹和场景编码
        encoded_rep = self.encoder(batch_agent_vec, batch_nbr_vec, batch_nbr_vec_mask, batch_nbr_rlpos, batch_nbr_rlpos_mask)

        if self.use_map:
            # 使用地图条件
            map_encoded = self.map_encoder(batch_vehicle_map).unsqueeze(1)  # [B, 1, 4096]
            
            # 交叉注意力模块
            cross_attn_out = self.cross_attn(encoded_rep, map_encoded)
            
            # 特征融合
            fused_features = self.feature_fusion(
                torch.cat([encoded_rep, cross_attn_out], dim=-1)
            )
        else:
            # 无地图，直接使用编码的自车+周车轨迹
            fused_features = encoded_rep  # 如果没有地图，直接使用编码结果

        # 将编码后的特征送入CVAE模型进行轨迹预测
        recon_x, mu, logvar = self.cvae_model(fused_features, batch_nbr_vec)  # CVAE模型，输入：fused_features，条件：周车轨迹

        return recon_x  # 输出预测的自车未来轨迹
