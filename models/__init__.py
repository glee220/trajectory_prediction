from transformers import (
    LlamaConfig, LlamaModel, AutoTokenizer,
    GPT2Config, GPT2Model, GPT2Tokenizer
)
import transformers
import torch
from transformers import BitsAndBytesConfig
###Qwen-1.5
from transformers import AutoModelForCausalLM, AutoTokenizer
##llama-3-8b分词器
from transformers import PreTrainedTokenizerFast

transformers.logging.set_verbosity_error()

def llm_load(configs):
    """
    :func: used to load llm model
    :param configs:
    :return: llm_model, tokenizer
    """
    # load model and tokenizer
    #######################################LLAMA###############################################
    if configs.llm_model == "LLAMA":
        # -------- 1. 解析权重路径 --------
        pretrained_model_path = (
            configs.llm_model_path
            if configs.llm_model_path is not None      # ← 用 is not None
            else "/models/Llama-2-7b-hf"
        )

        # -------- 2. 读取配置，切 SDPA kernel --------
        print("LLM (1/3) Loading Llama-2-7b-hf config ...")
        llama_config = LlamaConfig.from_pretrained(pretrained_model_path)
        llama_config._attn_implementation = "sdpa"      # ← xFormers / pytorch-sdpa
        llama_config.num_hidden_layers = configs.llm_layers
        llama_config.output_attentions = llama_config.output_hidden_states = False
        print("LLM (1/3) Config loaded ✔")

        # -------- 3. 加载权重（先尝试本地，只 catch FileNotFoundError）--------
        print("LLM (2/3) Loading Llama-2-7b-hf weights ...")
        try:
            llm_model = LlamaModel.from_pretrained(
                pretrained_model_path,
                config      = llama_config,
                torch_dtype = torch.float16,
                local_files_only = True,                # ← 先走本地
            )
        except FileNotFoundError:                       # ← 精准捕获
            print("Local weights not found, downloading from HuggingFace …")
            llm_model = LlamaModel.from_pretrained(
                pretrained_model_path,
                config      = llama_config,
                torch_dtype = torch.float16,
                trust_remote_code = True,
            )
        print(f"LLM (2/3) Weights loaded ✔  device={next(llm_model.parameters()).device}")

        # -------- 4. 加载 tokenizer（同样先本地） --------
        print("LLM (3/3) Loading tokenizer ...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_path,
                local_files_only = True,
                use_fast = True,                        # ← fast tokenizer 更快
                trust_remote_code = True,
            )
        except FileNotFoundError:
            print("Local tokenizer not found, downloading …")
            tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_path,
                use_fast = True,
                trust_remote_code = True,
            )
        print("LLM (3/3) Tokenizer loaded ✔")

    
    #######################################LLAMA3###############################################

    elif configs.llm_model == 'LLAMA3':
        if configs.llm_model_path is None:
            # 默认路径设置为 Meta-Llama-3-8B-Instruct
            pretrained_model_path = '/models/Meta-Llama-3-8B-Instruct'
        else:
            pretrained_model_path = configs.llm_model_path

        print('LLM Initialization Process (1/2): loading Meta-Llama-3-8B-Instruct ...... ')
        # 使用 AutoModelForCausalLM 加载模型
        llm_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_path,
            device_map=None,  # 自动选择设备
            torch_dtype="auto",  # 自动选择数据类型（例如 FP16）
            # torch_dtype=torch.float16,  # 通常用于混合精度
            trust_remote_code=True  # 如果需要从远程下载代码
        )

        print('LLM Initialization Process (2/2): Meta-Llama-3-8B-Instruct has been loaded successfully!')

        print('LLM Initialization Process (1/2): loading the tokenizer for Meta-Llama-3-8B-Instruct ...... ')
        # 使用 AutoTokenizer 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)
        print('LLM Initialization Process (2/2): the tokenizer for Meta-Llama-3-8B-Instruct has been loaded successfully!')

    
    #######################################Qwen2.5###############################################
    elif configs.llm_model == 'QWEN':

        if configs.llm_model_path == None:
            # LLM的保存路径
            pretrained_model_path = '/models/Qwen2.5-7B-Instruct' 
        else:
            pretrained_model_path = configs.llm_model_path

        # 2. 加载模型（自动设备分配+自动精度）
        print("Step 1/2: Loading Qwen1.5-7B-Instruct model...")
        llm_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_path,
            torch_dtype="auto",          # 自动选择BF16/FP16
            trust_remote_code=True,      # 必须启用！
            # 量化配置（可选，显存不足时启用）
            # quantization_config=BitsAndBytesConfig(
            #    load_in_4bit=True,
            #    bnb_4bit_compute_dtype=torch.bfloat16
            # )
        )
        print(f"Model loaded | Device: {next(llm_model.parameters()).device}")

        # 3. 加载分词器
        print("Step 2/2: Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_path,
            trust_remote_code=True,
            use_fast=True,               # 启用快速分词器
            padding_side='left'          # 生成任务建议左填充
        )
        print("Tokenizer ready!")
        print(f'[3/3] Config valid | Hidden_size: {llm_model.config.hidden_size}')
        
    #######################################Mistral###############################################
    elif configs.llm_model == 'Mistral':

        if configs.llm_model_path == None:
            # LLM的保存路径
            pretrained_model_path = '/models/Mistral-7B-Instruct-v0.2' 
        else:
            pretrained_model_path = configs.llm_model_path

        print('LLM Initialization Process (1/2): loading Mistral ...... ')
        llm_model = AutoModelForCausalLM.from_pretrained(pretrained_model_path, torch_dtype="auto", trust_remote_code=True)# !!!!!!!!!!!!!!!!!!
        print('LLM Initialization Process (2/3): Mistral has been loaded successfully!')

        print('LLM Initialization Process (2/2): loading the tokenizer for Mistral ...... ')
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)
        print('LLM Initialization Process (2/2): the tokenizer for Mistral has been loaded successfully!')
        
    #######################################vicuna###############################################
    elif configs.llm_model == 'vicuna':

        if configs.llm_model_path == None:
            # LLM的保存路径
            pretrained_model_path = '/models/vicuna-7b-v1.5' 
        else:
            pretrained_model_path = configs.llm_model_path

        # ===== 1. 加载模型 =====
        print('[1/3] Loading Vicuna-7B model...')
        llm_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_path,
            torch_dtype="auto",         # 自动选择BF16/FP16
            trust_remote_code=False     # Vicuna基于LLaMA，通常不需要
        )
        print(f'[1/3] Model loaded | Device: {next(llm_model.parameters()).device}')

        # ===== 2. 加载分词器 =====
        print('[2/3] Loading vicuna-7b-v1.5 tokenizer...')
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_path,
            use_fast=False              # Vicuna需禁用fast tokenizer
        )
        print('[2/3] Tokenizer loaded')

        # ===== 3. 验证配置 =====
        print(f'[3/3] Hidden_size: {llm_model.config.hidden_size}')  
    #######################################WizardLM###############################################
    elif configs.llm_model == 'WizardLM':

        if configs.llm_model_path == None:
            # LLM的保存路径
            pretrained_model_path = '/models/WizardLM-7B-V1.0' 
        else:
            pretrained_model_path = configs.llm_model_path

        print('LLM Initialization Process (1/2): loading WizardLM ...... ')
        llm_model = AutoModelForCausalLM.from_pretrained(pretrained_model_path, torch_dtype="auto", trust_remote_code=True)# !!!!!!!!!!!!!!!!!!
        print('LLM Initialization Process (2/3): WizardLM has been loaded successfully!')

        print('LLM Initialization Process (2/2): loading the tokenizer for WizardLM ...... ')
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)
        print('LLM Initialization Process (2/2): the tokenizer for WizardLM has been loaded successfully!')

    #######################################deepseek-llm-7b-base###############################################
    elif configs.llm_model == 'deepseek':

        if configs.llm_model_path == None:
            # LLM的保存路径
            pretrained_model_path =  '/models/deepseek-ai/deepseek-llm-7b-base'
        else:
            pretrained_model_path = configs.llm_model_path

        print('LLM Initialization Process (1/2): loading LLAMA3-8b  ...... ')

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        llm_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_path,
            quantization_config=bnb_config,
            device_map="auto"
        )
        print('LLM Initialization Process (2/2): LLAMA3-8b  has been loaded successfully!')

        print('LLM Initialization Process (1/2): loading the tokenizer for LLAMA3-8b  ...... ')
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path, trust_remote_code=True)
        print('LLM Initialization Process (2/2): the tokenizer for LLAMA3-8b has been loaded successfully!')

    #######################################return###############################################
    return llm_model, tokenizer
   