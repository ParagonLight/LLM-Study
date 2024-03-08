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
            - CLM: Causal Language Modeling
            - MLM: Masked Language Modeling
        - 从Transformer类型来说
            - Encoder-Decoder: BERT
            - Decoder-only: GPT系列
    - Transformer-based LLM结构基础
        - 激活函数: ReLU以及各种变种
        - MLP: 全连接层，即Dense(x, Wx+b)，LLM里是FFN
        - Attention: 注意力机制，核心的核心，讲解QKV含义，计算过程，以及优化部分
    - 典型LLM架构介绍
        - LLaMA系列: 基于HF的transformers的实现讲解LLM结构
        - MoE(代表:Mixtral): 基于HF的transformers的实现讲解LLM结构
        - ...
    - 其他
        - LoRA: 讲解LoRA技术原理，基于HF的peft实现讲解LoRA的实现方式
    - PyTorch框架下的LLM结构
        - state_dict: 模型参数数据结构
        - weight: 模型参数
        - forward function: 模型前向计算实现

-   输入(embedding)
    - 文本
        - tokenizer: 文本编码相关技术(语言模型标配流程)
    - 图像
        - CNN-based encoding: 图像如何编码的相关技术

## Development phase
涉及如何训练/微调LLM
- Preparing dataset: 如何准备数据集，包括数据集结构的介绍，以及针对训练任务方面数据集的准备工作
- Training LLM
    - Pretraining: 预训练
    - Supervised finetuning (sft): 监督微调tutorial
    - RLHF: 带人类反馈的微调手段介绍
    - sft with LoRA: 结合LoRA的监督微调tutorial

- Training-related framework
围绕Huggingface的LLM相关框架，讲解如何使用HF提供的工具开展LLM的应用
    - transformers: LLM结构相关
    - hf trainer: 训练相关
    - peft: LoRA相关
    - accerlrate: 训练相关加速
    - deepspeed: 训练相关加速
    - fsdp (ZeRO series): 训练相关加速
    - others
        - precision
            - bfloat16
            - float32
            - ...
        - mixed precision
    - LLaMA factory

## Application phase

- LLM + RAG: 检索增强生成，包括LangChain框架，抽象可理解为为LLM外挂知识库
- LLM as Agent: LLM理解state，生成action，开展交互
- NL2SQL: 基于自然语言查询需求生成SQL语句
- Code generation: 代码生成
- World model/foundation model: 世界模型
    
