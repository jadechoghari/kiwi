## Kiwi: The New Optimization Library for Hugging Face Models 🚀

### Plug in any HF model and watch it get faster while cutting down inference costs!

## Enjoy the speed and savings with Kiwi 🥝


```python
from kiwi_engine.KiwiModel import KiwiModel
from kiwi_engine.SmashConfig import SmashConfig

# Define your SmashConfig
smash_config = SmashConfig()
smash_config['task'] = 'text_text_generation'
smash_config['compilers'] = ['gptq']  # pptional: specify compilers to use
smash_config['quantization_bits'] = 4  # desired quantization bits
smash_config['device'] = 'cuda'  # target device
smash_config['device_map'] = 'auto'
smash_config['torch_dtype'] = torch.float16
# TODO: add other parameters as needed

model_path = 'facebook/opt-350m'  # replace with a desired model

# load and optimize the model
optimized_model = KiwiModel.load_model(model_path, smash_config)

# Now you can use the optimized_model for inference

```
## TODO:
- [ ] Add Diffusers support
- [ ] Add CV (Computer Vision) support
- [ ] Add Speech support
