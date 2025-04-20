# 🏥 MedReason-LLM Fine-Tuning with Phi-2

An end-to-end implementation of fine-tuning Microsoft's Phi-2 Large Language Model using the **MedReason** dataset for clinical reasoning tasks. This project leverages **QLoRA** (Quantized Low-Rank Adaptation) with 8-bit quantization to enable parameter-efficient training on limited hardware resources.

---

## 🧠 Project Overview

This repository showcases how to fine-tune the **Phi-2 LLM** using **QLoRA** with the **MedReason** dataset—a high-quality medical reasoning corpus designed to enhance clinical decision support systems (CDSS). The entire pipeline is built using [Hugging Face Transformers](https://huggingface.co/docs/transformers/index), [PEFT](https://huggingface.co/docs/peft/index), and [Accelerate](https://huggingface.co/docs/accelerate/index).

---

## ✨ Key Features

### 🔧 QLoRA Implementation
- 8-bit quantization for memory optimization
- Low-Rank Adaptation (LoRA) with:
  - Rank (`r`) = 32
  - Alpha = 64
  - Dropout = 0.05
- Targeted modules: `Wqkv`, `fc1`, `fc2`
- Gradient checkpointing for memory-efficient backpropagation

### ⚙️ Training Configuration
- **Model**: `microsoft/phi-2`
- **Batch Size**: 2 (per device)
- **Learning Rate**: 2.5e-5
- **Total Steps**: 500
- **Evaluation Frequency**: Every 25 steps
- **Gradient Accumulation**: 1

### 📊 Dataset Preprocessing
- Dataset: [`UCSC-VLAA/MedReason`](https://huggingface.co/datasets/UCSC-VLAA/MedReason)
- Subset of 200 records (due to hardware limitations)
- Train/Test Split: 90% / 10%
- Tokenization: Custom prompt-based format
- Max sequence length: 400 tokens

---

## 📚 Usage Guide

### 1️⃣ Environment Setup

```bash
# Create and activate virtual environment
pip install virtualenv
virtualenv finetune
source finetune/bin/activate

# Install dependencies
pip install accelerate bitsandbytes trl peft transformers datasets huggingface_hub
```

---

### 2️⃣ Training the Model

- Load and tokenize the dataset
- Format input prompts (Question → Answer with Reasoning)
- Load the base Phi-2 model with 8-bit precision
- Apply LoRA configuration
- Train using Hugging Face's `Trainer` API

```python
from peft import LoraConfig, get_peft_model
from transformers import Trainer, TrainingArguments

# LoRA setup
config = LoraConfig(r=32, lora_alpha=64, target_modules=["Wqkv", "fc1", "fc2"],
                    bias="none", lora_dropout=0.05, task_type="CAUSAL_LM")
model = get_peft_model(model, config)

# Training setup
args = TrainingArguments(output_dir="./train-dir", per_device_train_batch_size=2,
                         max_steps=500, learning_rate=2.5e-5, save_steps=25,
                         eval_steps=25, logging_steps=25, do_eval=True)

trainer = Trainer(model=model, args=args, ...)
trainer.train()
```

---

### 3️⃣ Inference and Evaluation

After training, use the PEFT adapter to merge with the base model for inference.

```python
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", ...)
ft_model = PeftModel.from_pretrained(base_model, "./train-dir/checkpoint-500")

output = ft_model.generate(...)
```

Use prompt-based inference to validate model responses on medical reasoning queries.

---

## 📈 Performance & Optimization

- **QLoRA Efficiency**: Fine-tunes only ~0.1% of model parameters
- **Memory Optimization**: 8-bit quantization allows training large models on consumer GPUs
- **Evaluation**: Periodic testing (every 25 steps) ensures model convergence

---

## 📦 Save & Export

To export your fine-tuned adapters:

```bash
zip -r phi2_qlora_adapter.zip ./train-dir/checkpoint-500
```

These can later be reused with the base model for inference in any deployment pipeline.

---

## ✅ Requirements

- Python ≥ 3.8  
- PyTorch ≥ 2.0  
- CUDA-compatible GPU (16GB+ recommended)  
- Dependencies: `transformers`, `datasets`, `peft`, `trl`, `bitsandbytes`, `accelerate`

---

## 📜 License

This project uses open-source libraries. Please refer to each library’s license for usage rights. The MedReason dataset is provided by UCSC-VLAA under their licensing terms.

---

## 🙏 Acknowledgements

- [Hugging Face](https://huggingface.co) ecosystem  
- Microsoft [Phi-2](https://huggingface.co/microsoft/phi-2) model  
- [MedReason](https://huggingface.co/datasets/UCSC-VLAA/MedReason) dataset by UCSC-VLAA  
- QLoRA by Tim Dettmers et al.