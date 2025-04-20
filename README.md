# MedReason-LLM-Fine-Tuning with Phi-2 🏥

Implementation of QLoRA fine-tuning on the Phi-2 model using the MedReason medical reasoning dataset for enhanced clinical decision support.

## Overview 📋
This project demonstrates fine-tuning the Microsoft Phi-2 model on MedReason dataset using PEFT (Parameter Efficient Fine-Tuning) with QLoRA in 8-bit precision.

## Features ✨
- **QLoRA Implementation** 🔧
    - 8-bit quantization
    - LoRA rank = 32, alpha = 64
    - Gradient checkpointing
    - Optimized memory usage
    
- **Training Configuration** ⚙️
    - Batch size: 2
    - Learning rate: 2.5e-5
    - Max steps: 500
    - Evaluation steps: 25

- **Dataset Processing** 📊
    - Custom prompt formatting
    - Train/test split (90/10)
    - Maximum sequence length: 400 tokens

## Usage Guide 📚
1. **Environment Setup** 🛠️
    ```bash
    pip install virtualenv
    virtualenv finetune
    pip install accelerate bitsandbytes trl peft transformers datasets huggingface_hub
    ```

2. **Model Training** 🚀
    - Load Phi-2 base model
    - Apply QLoRA configuration
    - Train with custom prompt format
    - Save adapter checkpoints

3. **Inference** 🔍
    - Load base model with PEFT adapters
    - Generate responses with reasoning
    - Support for medical queries

## Performance 📈
- Efficient parameter training (~0.1% of total params)
- 8-bit quantization for memory efficiency
- Evaluation every 25 steps

