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

MIN_Feat_DIM = 8          # 必须是 8 的倍数
########################################################################################################################
########################################### Encoder     ################################################################
########################################################################################################################
class AgentEmbedding(nn.Module):
    def __init__(self,
                 in_channel: int,
                 out_channel: int) -> None:
        super(AgentEmbedding, self).__init__()
        # self.embed = nn.Sequential(nn.Linear(in_channel, out_channel),
        # ① 先把 in_channel → MIN_Feat_DIM（8）
        self.preproj = nn.Linear(in_channel, MIN_Feat_DIM, bias=False)
        # ② 再投到真正的 embed_dim
        self.embed = nn.Sequential(
            nn.Linear(MIN_Feat_DIM, out_channel),
            nn.LayerNorm(out_channel),
            nn.ReLU(inplace=True),
            nn.Linear(out_channel, out_channel),
            nn.LayerNorm(out_channel),
            nn.ReLU(inplace=True),
            nn.Linear(out_channel, out_channel),
            nn.LayerNorm(out_channel))
        # 防止出现nan值
        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # return self.embed(x)
        x = self.preproj(x)
        return self.embed(x)

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
    

########################################################################################################################
########################################### Encoder     ################################################################
########################################################################################################################


########################################################################################################################
###########################################Reprogramming################################################################
########################################################################################################################
class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        # d_model代表的就是输入ReprogrammingLayer中embedding的维度
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)

        out = out.reshape(B, L, -1)

        return self.out_projection(out)


    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape

        scale = 1. / sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding



########################################################################################################################
###########################################  Decoder    ################################################################
########################################################################################################################
class LinearDecoder(nn.Module):
    def __init__(self,in_channel,out_channel,dropout_rate=0.1):
        super(LinearDecoder, self).__init__()
        self.x_projection = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(in_channel, out_channel),
            nn.Dropout(dropout_rate))
        self.y_projection = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(in_channel, out_channel),
            nn.Dropout(dropout_rate))
    def forward(self, input):
        x = self.x_projection(input)
        y = self.y_projection(input)
        return torch.cat([x.unsqueeze(-1),y.unsqueeze(-1)],-1)
    
########################################################################################################################
###########################################  map_encoder    ############################################################
########################################################################################################################
# class CNNMapEncoder(nn.Module):
#     def __init__(self, map_channels=3, output_size=4096):
#         super(CNNMapEncoder, self).__init__()
        
#         # 卷积+激活+池化层模块（按顺序堆叠）
#         self.convs = nn.Sequential(
#             nn.Conv2d(map_channels, 16, kernel_size=5, stride=1),  # -> (16, 96, 96)
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2),                           # -> (16, 48, 48)

#             nn.Conv2d(16, 32, kernel_size=3),                      # -> (32, 46, 46)
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2),                           # -> (32, 23, 23)

#             nn.Conv2d(32, 64, kernel_size=3),                      # -> (64, 21, 21)
#             nn.ReLU()
#         )

#         # 线性层将 Flatten 后的特征变为 4096 向量
#         self.fc = nn.Linear(64 * 21 * 21, output_size)

#     def forward(self, x, return_feature_maps=False):
#         """
#         :param x: 输入图像 (batch_size, 3, 100, 100)
#         :param return_feature_maps: 是否返回中间的 feature maps（用于可视化）
#         """
#         feature_maps = []

#         for layer in self.convs:
#             x = layer(x)
#             if isinstance(layer, nn.ReLU) and return_feature_maps:
#                 feature_maps.append(x.clone())  # 保存 ReLU 激活后的特征图

#         x = torch.flatten(x, start_dim=1)  # 展平
#         x = self.fc(x)  # 全连接输出4096维向量

#         if return_feature_maps:
#             return x, feature_maps
#         else:
#             return x

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

########################################################################################################################
###########################################  CrossModalAttention    #############################################
########################################################################################################################
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
    

########################################################################################################################
###########################################  Model    #############################################
########################################################################################################################
class Model(nn.Module):
    def __init__(self, configs, llm_model, tokenizer):
        super(Model, self).__init__()

        ################## 属性读取
        self.seq_len = configs.seq_len  # observed sequence length
        self.pred_len = configs.pred_len  # predicted sequence length

        self.llm_model = llm_model
        self.tokenizer = tokenizer

        self.d_llm = configs.llm_dim  # LLM 输入token的维度，对于llama来说是4096

        self.d_model = configs.d_model  # 在进行embedding的过程中所考虑的维度
        self.dropout_rate = configs.dropout
        self.n_heads = configs.n_heads

        ################## 设置LLM Tokenizer的 pad_token
        # eos_token (end of sequence) 是一个代表序列结束的特殊字符'</s>', pad_token 作为填充字符统一batch内不同序列长度
        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        ################## 对encoder 进行初始化
        self.encoder = GateAttnSceneEmbedder(dropout_rate=self.dropout_rate,
                                             num_head=8, agent_input_dim=2, nbr_input_dim=4,
                                             embed_dim=self.d_model, obs_len=self.seq_len, pred_len=self.pred_len)
        self.use_map = configs.use_map
        if self.use_map:
            
            ################## 对map_encoder 进行初始化
            # 修改地图编码器初始化
            self.map_encoder = EnhancedMapEncoder(
                d_model=configs.d_model,
                llm_dim=configs.llm_dim
            )
            
            # 修改交叉注意力初始化
            self.cross_attn = EnhancedCrossModalAttention(
                d_model=configs.llm_dim,
                n_heads=configs.n_heads,
                llm_dim=configs.llm_dim
            )
            
            # 统一特征融合层
            self.feature_fusion = nn.Sequential(
                nn.Linear(2 * configs.llm_dim, configs.llm_dim),
                nn.LayerNorm(configs.llm_dim)
            )

        ################## 冻结LLM的参数
        # for param in self.llm_model.parameters():
        #     param.requires_grad = False
        # ❸ 若需要微调 Llama 部分层，可通过 configs.llm_ft_layers 指定
        ft_layers = getattr(configs, "llm_ft_layers", 0)
        if ft_layers == 0:
            for p in self.llm_model.parameters():
                p.requires_grad = False
        else:
            for name, p in self.llm_model.named_parameters():
                # 只解冻最后 N 层
                p.requires_grad = any(f".layers.{i}." in name
                                      for i in range(configs.llm_layers - ft_layers,
                                                     configs.llm_layers))
        ################## 初始化prompt
        # self.description = 'Given the historical trajectory and local-map of vehicles, predict their future trajectory.'
        self.description = 'Given the past 2 seconds of vehicle trajectories and the local road geometry, predict the next 6 seconds of motion for the ego vehicle.'

        ################## 初始化 prototype learning
        # word_embedding, size of (vocabulary size, embedding_dim), a vocabulary of token embeddings
        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        # Attention: 这就是LLM中说的，利用linear layer，降低vocabulary size
        self.num_tokens = 1000
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)

        ################## 初始化 reprogramming
        self.reprogramming_layer = ReprogrammingLayer(d_model=self.d_model, n_heads=self.n_heads, d_keys=None,
                                                      d_llm=self.d_llm)

        ################## 初始化 decoder
        self.decoder = LinearDecoder(in_channel=self.seq_len * self.d_llm, out_channel=self.pred_len,
                                     dropout_rate=self.dropout_rate)
        

    # def forward(self, batch_agent_vec, batch_nbr_vec, batch_nbr_vec_mask, batch_nbr_rlpos, batch_nbr_rlpos_mask):
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

        ####################################################### 将轨迹/场景/地图 编码
        # -> [batch_size, obs_len, embed_dim]
        encoded_rep = self.encoder(batch_agent_vec, batch_nbr_vec, batch_nbr_vec_mask, batch_nbr_rlpos, batch_nbr_rlpos_mask)

        #################################################### 学习prototype word
        # [vocabulary size, 4096] -> [1000, 4096]
        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)

        #################################################### Reprogramming
        # [batch_size, obs_len, d_model] -> [batch_size, obs_len, llm_dim]
        reprogrammed_rep = self.reprogramming_layer(encoded_rep, source_embeddings, source_embeddings)

        if self.use_map:
            map_encoded = self.map_encoder(batch_vehicle_map).unsqueeze(1)  # [B, 1, 4096]
            
            # 直接使用 reprogrammed_rep 作为输入到交叉注意力
            cross_attn_out = self.cross_attn(reprogrammed_rep, map_encoded)
            
            # 特征融合
            fused_features = self.feature_fusion(
                torch.cat([reprogrammed_rep, cross_attn_out], dim=-1)
            )
            llm_in = torch.cat([prompt_embeddings, fused_features], dim=1)
        else:
            llm_in = torch.cat([prompt_embeddings, reprogrammed_rep], dim=1) # 拼接map_reprogrammed_rep
            
        with amp.autocast("cuda", dtype=torch.float16):        # ① 暂停混合精度  
            llm_raw = self.llm_model(                    #    推理 (FP32 GEMM)  
                inputs_embeds        = llm_in,
                output_hidden_states = True,
                return_dict          = True,
            )

        # ③ 兼容两种返回类型
        if isinstance(llm_raw, CausalLMOutputWithPast):
            hidden_llm = llm_raw.hidden_states[-1]       
        else:
            hidden_llm = llm_raw.last_hidden_state       
        # 隐藏状态（hidden_llm）包含了所有之前的输入信息（通过 attention 机制累积的上下文信息），解码器利用这些信息生成最终的输出。
        # ------------------------------------------------ decoder ------------------
        decoder_in  = hidden_llm[:, -self.seq_len:, :]   # <-- 改这里，用 hidden_llm
        decoder_out = self.decoder(decoder_in)
        return decoder_out

