# kiwi_engine/KiwiModel.py

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GPTQConfig,
    AwqConfig,
    AutoConfig,
)
from .optimizer import Optimizer
from .SmashConfig import SmashConfig

class KiwiModel:
    @staticmethod
    def load_model(model_path, smash_config=None):
        """
        Load and optimize the model based on the provided SmashConfig.

        Args:
            model_path (str): Path to the model or HuggingFace model ID.
            smash_config (SmashConfig): Configuration for optimization.

        Returns:
            model: Optimized model or pipeline.
        """
        optimizer = Optimizer(model_path, smash_config)
        optimizer.optimize()
        # Return the optimized model or pipeline
        if hasattr(optimizer, 'pipeline'):
            return optimizer.pipeline
        else:
            return optimizer.get_model()
