# Kiwi: Smash Those Models, Go Kiwiiii 🥝💨

**Kiwi** is your ultimate sidekick for optimizing large language models (LLMs) without breaking a sweat. It smashes through inference costs, speeds up performance, and trims those bloated LLMs like a pro barber giving your models a fresh cut. Whether you're tired of paying AWS bills or you just want your model to go brrrr, **Kiwi** is here to save the day!

### Why Kiwi?
- **Bye-bye AWS costs!** 🍍 No more outrageous bills – Kiwi optimizes your models to make them run like Usain Bolt, even on the tiniest GPUs.
- **Blazing fast!** 🚀 Smash those big LLaMA, BLOOM, or Vicuna models down and make them go *kiwiii* with **2x** faster inference.
- **Easy as 1-2-3!** 🎉 Plug in your model, pick your optimization method, and let Kiwi do its thing. It’s that simple.

## 🔥 What’s New?
1. **LLM-Pruner Integration** – Trim your LLaMA, BLOOM, Vicuna, or Baichuan models with up to **30%** fewer parameters without sacrificing performance.
2. **TorchAo Quantization** – Turn those 32-bit whales into sleek, quantized machines with **4x** memory savings. Less memory, more kiwiii 🏄‍♂️.
3. **Diffusers & torch.compile Support** – Now supports **image and video generation** models with **53.88%** speedup on inference! Just throw in some `torch.compile()` magic, and watch your model go *brrrrrr*.

## ✨ How Kiwi Works (Go Kiwiiii Mode)
1. **Quantize**: Shrink your models down to int8 or int4 with zero effort, using `bitsandbytes` or `TorchAoConfig`.
2. **Prune**: Get rid of those unused model weights with **LLM-Pruner** and still keep your model sharp. 🍍 Trim it, tune it, and get ready to flex.
3. **Compile**: Throw in some **torch.compile()** magic and reduce inference time like a boss.

---

## 🍍 Quick Start (Go Kiwiiii Mode)
### 1. **Install Kiwi**
First things first, let’s get you set up:

```bash
pip install kiwi-smash
```

---

### 2. **Prune Your LLM (LLM-Pruner)**
Sick of bloated models slowing you down? Let Kiwi do the pruning for you! We now support **LLaMA, BLOOM, Vicuna, Baichuan**, and more! 💥

#### Example: Prune LLaMA-7B 🍍
Here’s how you can prune **25%** of a LLaMA-7B model’s parameters and still make it crush those tasks:

```python
from kiwi_engine.KiwiModel import KiwiModel
from kiwi_engine.SmashConfig import SmashConfig

# Prune LLaMA-7B model by 25%
smash_config = SmashConfig()
smash_config['pruning'] = True  # Activate pruning!
smash_config['pruning_ratio'] = 0.25  # Remove 25% of the weights
smash_config['model_type'] = 'llama'  # Smash LLaMA
smash_config['block_wise'] = True  # Block-wise pruning strategy
smash_config['device'] = 'cuda'  # Run on CUDA

model_id = 'meta-llama/Llama-2-7b-hf'
optimized_model = KiwiModel.load_model(model_id, smash_config)

# Now your model is lean and fast! 🍍
```

#### Example: Prune BLOOM by 30% 🍍
BLOOM models can also get the Kiwi treatment:

```python
# Prune BLOOM model by 30%
smash_config = SmashConfig()
smash_config['pruning'] = True
smash_config['pruning_ratio'] = 0.30  # Remove 30% of the weights
smash_config['model_type'] = 'bloom'
smash_config['channel_wise'] = True  # Channel-wise pruning

model_id = 'bigscience/bloom-7b1'
optimized_model = KiwiModel.load_model(model_id, smash_config)
```

---

### 3. **Quantize Your Model (TorchAo, bitsandbytes)**

Get your models slimmed down to int4 or int8 with **TorchAo** or **bitsandbytes**. Save up to **4x** on memory. Here’s how you can quantize **LLaMA-7B** to run at max speed without AWS eating your wallet! 💸💨

```python
import kiwi

# Quantize LLaMA-7B model using Kiwi (TorchAo int8 quantization)
model_id = 'meta-llama/Llama-2-7b-hf'

# Let Kiwi do the quantization for you! 🍍
model, tokenizer = kiwi.KiwiModel.quantize_model(model_id, quantization_type="int8_weight_only", group_size=64)

# Prepare input for the model
prompt = "What is the meaning of life?"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# Generate the o🤗utput and print the result
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0]))  # BOOM! 🍍 Faster and leaner inference.
```

---

### 4. **Compile Your Model with Diffusers Support**

Ready to speed up **image generation** and **video generation** models? Kiwi now supports **Flux** and **CogVideoX** pipelines for **53.88%** speed boosts. Throw in **torch.compile()** and watch your models go *brrr* like never before. 💥

```python
import kiwi

# compile and optimize a diffusers 🤗 model using kiwi 🥝 (flux pipeline)
pipeline_id = "black-forest-labs/FLUX.1-dev"

# Let Kiwi speed it up! 🥝
pipeline = kiwi.KiwiModel.compile_diffusers_model(pipeline_id)

image = pipeline("a kiwi surfing on a rainbow", guidance_scale=3.5, num_inference_steps=50).images[0]
image.show()  # Look at that speed. 🍍
```

---

## 🍍 Full Features of Kiwi

### Smash Your Model with:
- **LLM-Pruner**: Prune your **LLaMA, BLOOM, Vicuna** models for up to **30%** weight reduction, all while keeping that killer performance.
- **TorchAo & bitsandbytes**: Quantize models down to **int4** or **int8**, saving you **4x** in memory usage.
- **Diffusers with torch.compile()**: Boost your image and video generation models with **53.88%** faster inference.
- **Full Gradio Interface**: Ready-to-deploy models with just one line.

---

## Go Kiwiiiiii! 🥝🥝🥝

With Kiwi, the days of high AWS bills and slow LLMs are over. Whether you're pruning, quantizing, or compiling, Kiwi will make your models run faster, better, and cheaper. Ready to go brrrr? Let’s Kiwiiiii! 🎉

---

### Installation
```bash
!git clone https://github.com/jadechoghari/kiwi.git
```

And you’re off to the races! 🚀 Go Kiwiiii and watch your models fly.
