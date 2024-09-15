## Kiwi 🥝 Optimization Library

Kiwi helps you optimize AI models by reducing inference costs and speeding up performance.
Whether you want to quantize, prune, or compile your models,wi simplifies it and provides immediate gains.

### Table of Contents

1. [Installation](#installation)
2. [Key Features](#key-features)
3. [Quick Start](#quick-start)
    - [Quantization](#quantization)
    - [Pruning](#pruning)
    - [Compilation](#compilation)
4. [Configuration](#configuration)
5. [Example Workflow](#example-workflow)

---

### 1. Installation

Install Kiwi 🥝:

```bash
git clone https://github.com/your-repo/kiwi.git
cd kiwi
pip install -r requirements.txt
```

To use quantization and pruning:

```bash
pip install bitsandbytes timm torchao
```

---

### 2. Key Features

- **Quantization**: Lower the precision of model weights for 4x memory savings and up to **50% faster inference**.
- **Pruning**: Reduce model size by up to **50%** while maintaining accuracy, and see up to **30% faster inference**.
- **Compilation**: Use `torch.compile()` to get **2x faster inference** on A100 GPUs by optimizing the forward pass.
- **Diffusers Optimization**: Speed up image and video generation models.

---

### 3. Quick Start

#### Quantization

Use Kiwi to load quantized models with `TorchAoConfig` and reduce memory usage by 4x with **int4_weight_only** quantization.

```python
from kiwi_engine.KiwiModel import KiwiModel
from kiwi_engine.SmashConfig import SmashConfig

smash_config = SmashConfig()
smash_config['quant_type'] = 'int4_weight_only'
smash_config['group_size'] = 128
smash_config['compile'] = True  # Compile for faster inference

model_id = 'meta-llama/Meta-Llama-3.1-8B-Instruct'

optimized_model = KiwiModel.load_model(model_id, smash_config)
```

---

#### Pruning

Prune both `timm` and LLaMA models. Reduce the number of parameters while keeping performance high. This gives **up to 30% faster inference** and saves memory.

```python
# Prune a TIMM model
smash_config['pruning_ratio'] = 0.5  # Prune 50% of the weights
optimized_model = KiwiModel.load_model('convnext_xxlarge', smash_config)
```

#### Compilation

Compile the model’s forward function to unlock **2x faster inference** on A100 GPUs.

```python
smash_config['compile'] = True
optimized_model = KiwiModel.load_model('meta-llama/Meta-Llama-3.1-8B-Instruct', smash_config)
```

---

### 4. Configuration

**SmashConfig** allows you to control quantization, pruning, and compilation.

| Parameter         | Description                                      | Default           |
|-------------------|--------------------------------------------------|-------------------|
| `quant_type`      | Type of quantization (e.g., `int4_weight_only`)   | `int4_weight_only`|
| `pruning_ratio`   | Percentage of model weights to prune              | `0.0`             |
| `compile`         | Whether to compile the model for faster inference | `False`           |
| `group_size`      | Group size for quantization                       | `128`             |
| `device`          | Device for running the model (`cuda`, `cpu`)      | `cuda`            |

---

### 5. Example Workflow

Here’s how you can use Kiwi to quantize, prune, and compile a model for fast and cost-effective inference.

```python
from kiwi_engine.KiwiModel import KiwiModel
from kiwi_engine.SmashConfig import SmashConfig

smash_config = SmashConfig()
smash_config['quant_type'] = 'int8_weight_only'
smash_config['pruning_ratio'] = 0.3
smash_config['compile'] = True

model_id = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
optimized_model = KiwiModel.load_model(model_id, smash_config)

# Inference example
tokenizer = optimized_model.tokenizer
inputs = tokenizer("Why are dogs so cute?", return_tensors="pt").to('cuda')
outputs = optimized_model.generate(**inputs)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f"Generated response: {response}")
```

---

By following these steps, you’ll see **up to 50% memory savings** and **2x faster inference** on your AI models using Kiwi.

---

This version keeps it simple, highlights the inference gains, and follows a straightforward style. Let me know if it works for you!