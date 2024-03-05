## Prerequisites

- 理论基础
    - 机器学习基础
    - 深度学习基础
    - 概率论
    - 线性代数

- 实操基础
    - Python
    - PyTorch


## LLM basics

- 模型结构
    - Transformer-based LLM类型
        - 从训练角度说
            - CLM
            - MLM
        - 从Transformer类型来说
            - Encoder-Decoder
            - Decoder-only
    - Transformer-based LLM结构基础
        - 激活函数
        - MLP
        - Attention
    - 典型LLM架构介绍
        - LLaMA系列
        - MoE(代表:Mixtral)
        - ...
    - 其他
        - LoRA
    - PyTorch框架下的LLM结构
        - state_dict
        - weight
        - forward function

-   输入(embedding)
    - 文本
        - tokenizer
    - 图像
        - CNN-based encoding

## Development phase

    - Preparing dataset
    - Training LLM
        - Pretraining
        - Supervised finetuning (sft)
        - RLHF
        - sft with LoRA

    - Training-related framework
        - transformers
            - hf trainer
        - peft
        - accerlrate
        - deepspeed
        - fsdp (ZeRO series)
        - others
            - precision
                - bfloat16
                - float32
                - ...
            - mixed precision
        - LLaMA factory

## Application phase

    - LLM + RAG
    - LLM as Agent
    - NL2SQL
    - Code generation
    - World model/foundation model
    
