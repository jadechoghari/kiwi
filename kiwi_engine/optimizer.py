# kiwi_engine/optimizer.py

import torch
import logging
import sys
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GPTQConfig,
    AwqConfig,
    AutoConfig,
)

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

class Optimizer:
    def __init__(self, model_id, smash_config):
        """
        Initialize the optimizer with the given model_id and smash_config.

        Args:
            model_id (str): HuggingFace model ID or local path.
            smash_config (SmashConfig): Smash configuration dictionary.
        """
        self.model_id = model_id
        self.smash_config = smash_config
        self.model = None
        self.tokenizer = None
        self.supported_quant_methods = ['gptq', 'awq', 'bitsandbytes']
        self.quant_method = None

    def optimize(self):
      """
      Optimize the model by automatically selecting the best quantization method.
      """
      # Check if the task is related to Diffusers
      if self.smash_config.get('task') in [
          'text_image_generation',
          'image_image_generation',
          'text_video_generation',
          # Add other tasks as needed
      ]:
          logger.info("Detected Diffusers task. Optimizing Diffusers pipeline.")
          self._optimize_diffusers_pipeline()
      else:
          logger.info("Optimizing non-Diffusers model.")
          # Existing code for handling other models
          # Check for available quantization methods for the model
          quant_methods = self._get_supported_quant_methods()

          if not quant_methods:
              raise ValueError("No supported quantization methods available for this model.")

          # Use compilers specified in smash_config or auto-select
          if self.smash_config.get('compilers'):
              quant_methods = [method for method in self.smash_config['compilers'] if method in quant_methods]
              if not quant_methods:
                  raise ValueError("Specified compilers are not supported for this model.")
              self.quant_method = quant_methods[0]
          else:
              # Automatically select the best quantization method
              self.quant_method = self._select_best_quant_method(quant_methods)

          logger.info(f"Selected quantization method: {self.quant_method}")

          # Perform quantization based on the selected method
          if self.quant_method == 'gptq':
              self._quantize_with_gptq()
          elif self.quant_method == 'awq':
              self._quantize_with_awq()
          elif self.quant_method == 'bitsandbytes':
              self._quantize_with_bitsandbytes()
          else:
              raise ValueError(f"Unsupported quantization method: {self.quant_method}")

    def _get_supported_quant_methods(self):
        """
        Determine which quantization methods are supported for the given model.
        """
        supported_methods = []

        # Load the model config to check compatibility
        try:
            config = AutoConfig.from_pretrained(self.model_id)
        except Exception as e:
            logger.warning(f"Could not load model config: {e}")
            return ['bitsandbytes']  # Fallback to bitsandbytes

        # Check for GPTQ compatibility
        if config.model_type in ['gpt2', 'opt', 'bloom', 'llama']:
            supported_methods.append('gptq')

        # Check for AWQ compatibility
        if config.model_type in ['llama', 'mistral']:
            supported_methods.append('awq')

        # Bitsandbytes is generally supported
        supported_methods.append('bitsandbytes')

        return supported_methods

    def _select_best_quant_method(self, methods):
        """
        Select the best quantization method based on priority.
        """
        # Prioritize methods based on efficiency and compatibility
        priority = ['gptq', 'awq', 'bitsandbytes']
        for method in priority:
            if method in methods:
                return method
        return methods[0]  # Fallback

    def _quantize_with_gptq(self):
        """
        Quantize the model using GPTQ.
        """
        try:
            from transformers import AutoModelForCausalLM, GPTQConfig
            quantization_bits = self.smash_config.get('quantization_bits', 4)
            quantization_config = GPTQConfig(bits=quantization_bits)
            device_map = self.smash_config.get('device_map', 'auto')
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                device_map=device_map,
                quantization_config=quantization_config,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            logger.info("Model quantized with GPTQ successfully.")
        except ImportError:
            logger.error("Failed to import GPTQ libraries. Falling back to bitsandbytes.")
            self.quant_method = 'bitsandbytes'
            self._quantize_with_bitsandbytes()

    def _quantize_with_awq(self):
        """
        Quantize the model using AWQ.
        """
        try:
            from transformers import AutoModelForCausalLM, AwqConfig
            quantization_bits = self.smash_config.get('quantization_bits', 4)
            quantization_config = AwqConfig(bits=quantization_bits)
            device_map = self.smash_config.get('device_map', 'auto')
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                device_map=device_map,
                quantization_config=quantization_config,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            logger.info("Model quantized with AWQ successfully.")
        except ImportError:
            logger.error("Failed to import AWQ libraries. Falling back to bitsandbytes.")
            self.quant_method = 'bitsandbytes'
            self._quantize_with_bitsandbytes()

    def _optimize_diffusers_pipeline(self):
        """
        Optimize a Diffusers pipeline using torchao and torch.compile().
        """
        try:
            from diffusers import DiffusionPipeline
            import torch
            from torchao.quantization import autoquant

            device = self.smash_config.get('device', 'cuda')
            torch_dtype = self.smash_config.get('torch_dtype', torch.float16)

            # Load the Diffusers pipeline
            pipeline_class = self.smash_config.get('diffusers_pipeline', DiffusionPipeline)
            model_id = self.smash_config.get('diffusers_model_id', self.model_id)

            # Load the pipeline
            self.pipeline = pipeline_class.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
            ).to(device)

            # Apply torchao quantization
            quantization_method = self.smash_config.get('compilers', ['autoquant'])[0]
            if quantization_method == 'autoquant':
                logger.info("Applying autoquant to the transformer.")
                self.pipeline.transformer = autoquant(self.pipeline.transformer, error_on_unseen=False)
            elif quantization_method == 'int8_weight_only':
                logger.info("Applying int8 weight-only quantization.")
                from torchao.quantization.quantizer import int8_weight_only, quantize_
                quantize_(self.pipeline.transformer, int8_weight_only(), device=device)
            # Add other quantization methods as needed

            # Apply torch.compile()
            logger.info("Applying torch.compile to the transformer.")
            self.pipeline.transformer.to(memory_format=torch.channels_last)
            self.pipeline.transformer = torch.compile(
                self.pipeline.transformer,
                mode='max-autotune',
                fullgraph=True,
            )

            # Optionally compile the VAE decoder if present
            if hasattr(self.pipeline, 'vae'):
                logger.info("Applying torch.compile to the VAE decoder.")
                self.pipeline.vae.decode = torch.compile(
                    self.pipeline.vae.decode,
                    mode='max-autotune',
                    fullgraph=True,
                )

            logger.info("Diffusers pipeline optimized successfully.")
        except ImportError as e:
            logger.error(f"Failed to import required libraries for Diffusers optimization: {e}")
            raise ImportError("Please install diffusers and torchao to use this feature.")

    def _quantize_with_bitsandbytes(self):
        """
        Quantize the model using bitsandbytes.
        """
        quantization_bits = self.smash_config.get('quantization_bits', 8)
        load_in_8bit = quantization_bits == 8
        load_in_4bit = quantization_bits == 4

        quantization_config = BitsAndBytesConfig(
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            llm_int8_enable_fp32_cpu_offload=self.smash_config.get('llm_int8_enable_fp32_cpu_offload', False),
            llm_int8_threshold=self.smash_config.get('llm_int8_threshold', 6.0),
            llm_int8_skip_modules=self.smash_config.get('llm_int8_skip_modules', None),
            bnb_4bit_compute_dtype=self.smash_config.get('bnb_4bit_compute_dtype', torch.float16),
            bnb_4bit_use_double_quant=self.smash_config.get('bnb_4bit_use_double_quant', False),
            bnb_4bit_quant_type=self.smash_config.get('bnb_4bit_quant_type', 'fp4'),
        )

        device_map = self.smash_config.get('device_map', 'auto')
        torch_dtype = self.smash_config.get('torch_dtype', torch.float16)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=quantization_config,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        logger.info("Model quantized with bitsandbytes successfully.")

    def get_model(self):
        """
        Get the optimized model.

        Returns:
            model: The optimized model.
        """
        if self.model is None:
            raise ValueError("Model not optimized yet. Call optimize() first.")
        return self.model

    def get_tokenizer(self):
        """
        Get the tokenizer corresponding to the model.

        Returns:
            tokenizer: The tokenizer associated with the model.
        """
        if self.tokenizer is None:
            raise ValueError("Model not optimized yet. Call optimize() first.")
        return self.tokenizer

    def push_to_hub(self, repo_name):
        """
        Push the optimized model to the HuggingFace Hub.

        Args:
            repo_name (str): Repository name to push the model to.
        """
        if self.model is not None and self.tokenizer is not None:
            self.model.push_to_hub(repo_name)
            self.tokenizer.push_to_hub(repo_name)
            logger.info(f"Model and tokenizer pushed to hub under {repo_name}.")
        else:
            raise ValueError("Model not optimized yet. Call optimize() first.")

    def save_pretrained(self, save_directory):
        """
        Save the optimized model and tokenizer to the specified directory.

        Args:
            save_directory (str): Directory to save the model and tokenizer.
        """
        if self.model is not None and self.tokenizer is not None:
            self.model.save_pretrained(save_directory)
            self.tokenizer.save_pretrained(save_directory)
            logger.info(f"Model and tokenizer saved to {save_directory}.")
        else:
            raise ValueError("Model not optimized yet. Call optimize() first.")
