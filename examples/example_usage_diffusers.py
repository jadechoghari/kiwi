# example_usage_diffusers.py

from kiwi_engine.KiwiModel import KiwiModel
from kiwi_engine.SmashConfig import SmashConfig
import torch
from diffusers import StableDiffusionPipeline

# define your SmashConfig // this is an example
smash_config = SmashConfig()
smash_config['task'] = 'text_image_generation'
smash_config['compilers'] = ['autoquant']  # use 'autoquant' or other methods
smash_config['device'] = 'cuda'  # Target device
smash_config['torch_dtype'] = torch.bfloat16
smash_config['diffusers_pipeline'] = StableDiffusionPipeline  # Specify the pipeline class
smash_config['diffusers_model_id'] = 'CompVis/stable-diffusion-v1-4'  # Model ID

# load and optimize the Diffusers pipeline
optimized_pipeline = KiwiModel.load_model(None, smash_config)

# generate an image
prompt = "a photograph of an astronaut riding a horse"
with torch.autocast("cuda"):
    image = optimized_pipeline(prompt).images[0]

# save or display the image
image.save("output.png")

# ENJOY! 🚀