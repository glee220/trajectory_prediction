from math import sqrt
import transformers
transformers.logging.set_verbosity_error()
from utils.init_weight import init_weights
import torch
from torch import nn
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast
from torch import amp

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

    def forward(self, batch_agent_vec):

        """
        :param batch_agent_vec: tensor of shape [batch_size, obs_len, 2], 此处第3维的2代表central agent沿着x,y两个方向的位移向量
        :return:
        """


        ########## central agent的embedding
        # [batch_size, obs_len, 2] -> [batch_size, obs_len, embed_dim]
        batch_agent_embeddings = self.agent_embedder(batch_agent_vec)
        # 用learnable token填充agent embedding的首个时刻
        batch_agent_embeddings[:, 0, :] = self.NA_token_seq[0]  # 它里面有NaN


        return batch_agent_embeddings


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
        self.description = 'Given the historical trajectory and local-map of vehicles, predict their future trajectory.'

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
        encoded_rep = self.encoder(batch_agent_vec)

        #################################################### 学习prototype word
        # [vocabulary size, 4096] -> [1000, 4096]
        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)

        #################################################### Reprogramming
        # [batch_size, obs_len, d_model] -> [batch_size, obs_len, llm_dim]
        reprogrammed_rep = self.reprogramming_layer(encoded_rep, source_embeddings, source_embeddings)

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

