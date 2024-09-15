# example_usage.py

from kiwi_engine.KiwiModel import KiwiModel
from kiwi_engine.SmashConfig import SmashConfig

# Define your SmashConfig
smash_config = SmashConfig()
smash_config['task'] = 'text_text_generation'
smash_config['compilers'] = ['gptq']  # Optional: specify compilers to use
smash_config['quantization_bits'] = 4  # Desired quantization bits
smash_config['device'] = 'cuda'  # Target device
smash_config['device_map'] = 'auto'
smash_config['torch_dtype'] = torch.float16
# Add other parameters as needed

model_path = 'facebook/opt-350m'  # Replace with your desired model

# Load and optimize the model
optimized_model = KiwiModel.load_model(model_path, smash_config)

# Now you can use the optimized_model for inference
