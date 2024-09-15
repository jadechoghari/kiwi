# kiwi_engine/SmashConfig.py
import torch

class SmashConfig(dict):
    """
    Configuration class for optimizing models.

    Example:
        from kiwi_engine.SmashConfig import SmashConfig
        smash_config = SmashConfig()
        smash_config['task'] = 'text_text_generation'
        smash_config['compilers'] = ['gptq']
        smash_config['quantization_bits'] = 4
        smash_config['device'] = 'cuda'
    """

    def __init__(self):
        super().__init__()
        self['task'] = None
        self['compilers'] = []
        self['quantization_bits'] = 8  # Default quantization bits
        self['device_map'] = 'auto'
        self['torch_dtype'] = torch.float16
        self['diffusers_pipeline'] = None
        self['diffusers_model_id'] = None 
        self['quant_type'] = None,  # Default quantization type
        self['group_size'] = 128  # Default group size for quantization
        self['torch_dtype'] = torch.bfloat16  # Default precision
        self['max_length'] = 128  # Default max length for generation
        self['cache_implementation'] = 'static'  # Default cache implementation
        self['device'] = 'cuda'
        # Add other default parameters as needed
