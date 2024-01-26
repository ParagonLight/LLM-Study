# LLM-Study

This repository is dedicated to storing and sharing materials focused on the study of Large Language Models (LLMs) from both theoretical and practical implementation perspectives. Our primary focus is on resources from [Hugging Face](https://huggingface.co/docs) along with other relevant papers and repositories.

Hugging Face is renowned for its extensive libraries that facilitate the training and deployment of LLMs. The materials shared here are related to these libraries. Some tutorials in this repository have been adapted from Hugging Face's original tutorials to better suit our specific needs.

## LLM Basics

### Transformers

[Transformers](https://huggingface.co/docs/transformers/) is a leading library for working with LLMs, offering functionalities for loading pre-trained models, tokenizers, processors, etc. It enables users to fine-tune pre-trained models with the Trainer API and to pre-train models from scratch.

### Datasets

The Datasets library is designed for converting and assembling text, images, or audio into batches of tensors, streamlining the process of feeding data into models.

### PEFT

PEFT (Parameter-Efficient Fine-Tuning) focuses on adapting large pre-trained models to various downstream applications without the need to fine-tune all parameters— a task that can be prohibitively expensive. PEFT implements [LoRA](https://arxiv.org/abs/2106.09685), a technique for the efficient training of large language models.

### Tokenizer

Tokenizers segment input text into manageable pieces, allowing models to process and understand the information more effectively for tasks like language comprehension.

## LLM Training

### Accelerate

Accelerate simplifies the process of training models using multiple GPUs, either on a single machine or distributed across several machines.

### DeepSpeed

DeepSpeed, provided by Microsoft, is a library similar to Accelerate but offers unique optimizations and scalability for training deep learning models.

### FairScale

FairScale is a Python library aimed at efficient and scalable training of large-scale models, facilitating distributed and parallel processing.

### FSDP

FSDP (Fully Sharded Data Parallel) is a technique that shards a model’s parameters, gradients, and optimizer states across available GPUs to enhance training efficiency.

### Megatron-LM

Megatron-LM is focused on model parallelism, similar to the Zero Redundancy Optimizer, optimizing the training of very large models.

### Optimum

Optimum extends the Transformers library to provide a suite of performance optimization tools designed to maximize efficiency on targeted hardware.

## Performance Optimization

### FlashAttention-2

FlashAttention-2 offers a faster and more efficient alternative to the standard attention mechanism, significantly accelerating model inference.

### Triton

The Triton Inference Server is optimized for cloud and edge inferencing, providing an efficient solution for deploying models in production environments.

### Bitsandbytes

Bitsandbytes is a quantization library supporting 4-bit and 8-bit quantization, aimed at reducing model size and improving inference speed.

### GPTQ

GPTQ introduces a technique for quantizing parameters in large language models like GPT, optimizing them for performance without significant loss of accuracy.

### VLLM

VLLM is designed for high-throughput and memory-efficient inference and serving of LLMs, addressing the challenges of deploying large models.