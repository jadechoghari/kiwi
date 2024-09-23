# kiwi_engine/KiwiModel.py
# TODO: add license

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GPTQConfig,
    AwqConfig,
    AutoConfig,
    AutoModelForCausalLM, AutoTokenizer, TorchAoConfig
)
from .optimizer import Optimizer
from .SmashConfig import SmashConfig
from diffusers import FluxPipeline
from torchao.quantization import autoquant
import torch


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
        
    @staticmethod
    def compile_flux_model(pipeline_id, device="cuda"):
        """
        Loads and compiles a diffusers model for faster inference.

        Args:
            pipeline_id (str): Hugging Face Diffusers pipeline ID.
            device (str): Device to run the model on (e.g., 'cuda').

        Returns:
            pipeline: Compiled and optimized pipeline.
        """
        # Load the pipeline
        pipeline = FluxPipeline.from_pretrained(pipeline_id, torch_dtype=torch.bfloat16).to(device)
        
        # Apply autoquant and torch.compile for speedups
        pipeline.transformer = autoquant(pipeline.transformer)
        pipeline.transformer = torch.compile(pipeline.transformer, mode="max-autotune", fullgraph=True)
        
        return pipeline
    

    @staticmethod
    def quantize_model(model_id, quantization_type="int8_weight_only", group_size=64, device="cuda"):
        """
        Quantizes the model using TorchAoConfig.

        Args:
            model_id (str): Hugging Face model ID.
            quantization_type (str): Type of quantization (int4_weight_only, int8_weight_only, int8_dynamic_activation_int8_weight).
            group_size (int): Group size for weight quantization.
            device (str): Device to run the model on (e.g., 'cuda', 'cpu').
        
        Returns:
            Tuple: Quantized model and tokenizer.
        """
        # Set up TorchAoConfig for quantization
        quantization_config = TorchAoConfig(quant_type=quantization_type, group_size=group_size)
        
        # Load model with quantization config
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
            device_map=device
        )
        
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        return model, tokenizer
