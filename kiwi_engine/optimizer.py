# kiwi_engine/optimizer.py

import argparse
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
    AutoModelForCausalLM, AutoTokenizer, TorchAoConfig
)
import timm
from .pruner import prune_llama
from .pruner import prune_timm
from .pruner.hf_prune import main as prune_llm_main
import argparse
import subprocess


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

      # Check if pruning is requested
      # Check if pruning is requested
      quant_type = self.smash_config.get('quant_type', None)
      if quant_type:
            logger.info(f"Quantization requested with type: {quant_type}. Starting quantization process.")
            self._quantize_model()
            return
      pruning_ratio = self.smash_config.get('pruning_ratio', 0.0)
      timm_model_list = timm.list_models()
      if self.model_id in timm_model_list:
            logger.info(f"Detected TIMM model: {self.model_id}. Starting TIMM model pruning.")
            self._prune_timm_model()
            return  # Exit after TIMM model pruning
      if pruning_ratio > 0.0:
            logger.info("Pruning requested. Starting pruning process.")
            self._prune_model()
            return  # Exit after pruning
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

    def parse_args():
      parser = argparse.ArgumentParser(description='Pruning LLaMA (huggingface version)')

      # argument for parsing
      parser.add_argument('--base_model', type=str, default="decapoda-research/llama-7b-hf", help='base model name')
      parser.add_argument('--save_ckpt_log_name', type=str, default="llama_prune", help='the path for saving the checkpoint and the log.')
      parser.add_argument('--pruning_ratio', type=float, default=0.5, help='pruning ratio')
      parser.add_argument('--pruner_type', type=str, default='l2', help='pruner type')

      # argument for generation
      parser.add_argument('--temperature', type=float, default=1.0, help='temperature')
      parser.add_argument('--top_p', type=float, default=0.95, help='top p')
      parser.add_argument('--max_seq_len', type=int, default=128, help='max sequence length')

      # argument for pruning
      parser.add_argument('--channel_wise', action='store_true', help='channel wise')
      parser.add_argument('--block_wise', action='store_true', help='block wise')
      parser.add_argument('--layer_wise', action='store_true', help='layer wise')
      parser.add_argument('--layer', type=int, default=12, help='remain the previous n layers')

      parser.add_argument('--block_attention_layer_start', type=int, default=3, help='start layer of block attention layers')
      parser.add_argument('--block_attention_layer_end', type=int, default=31, help='end layer of block attention layers')
      parser.add_argument('--block_mlp_layer_start', type=int, default=3, help='start layer of block mlp layers')
      parser.add_argument('--block_mlp_layer_end', type=int, default=31, help='end layer of block mlp layers')

      parser.add_argument('--iterative_steps', type=int, default=1, help="Iteration step for pruning. Default=1")
      parser.add_argument('--grouping_strategy', type=str, default='sum', help='Reduce method for grouping')
      parser.add_argument('--global_pruning', action='store_true', help='whether global pruning')
      parser.add_argument('--taylor', type=str, default='param_first', help='choose from [vectorize, param_second, param_first, param_mix]')
      parser.add_argument('--num_examples', type=int, default=10)

      # general argument
      parser.add_argument('--device', type=str, default="cuda", help='device')
      parser.add_argument('--test_before_train', action='store_true', help='whether test before training')
      parser.add_argument('--eval_device', type=str, default="cuda", help='evaluation device')
      parser.add_argument('--test_after_train', action='store_true', help='whether test after training')
      parser.add_argument('--seed', type=int, default=42, help='random seed')
      parser.add_argument('--save_model', action='store_true', help='whether to save model')

      # Parse only known arguments
      args, unknown = parser.parse_known_args()

      return args

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
    

    def _prune_model(self):
        """
        Prune the model using the external prune_llama.py script.
        """
        import sys
        import pruner.prune_llama as prune_llama

        # Prepare arguments for prune_llama
        args = argparse.Namespace(
            model=self.model_id or self.smash_config.get('model_id'),
            seed=self.smash_config.get('seed', 0),
            nsamples=self.smash_config.get('nsamples', 128),
            pruning_ratio=self.smash_config.get('pruning_ratio', 0),
            save=None,  # Optionally set a path to save results
            save_model=self.smash_config.get('save_model'),
            eval_zero_shot=self.smash_config.get('eval_zero_shot', False)
        )
        model=self.model_id or self.smash_config.get('model_id')
        pruning_ratio=self.smash_config.get('pruning_ratio', 0)
        seed=self.smash_config.get('seed', 0)

        # Call the main function from prune_llama
        prune_llama.main(args, model_name=model, pruning_ratio=pruning_ratio, seed=seed)

        logger.info(f"Pruning completed for model {args.model}.")
        
        if args.save_model:
            logger.info(f"Pruned model saved to {args.save_model}")

    def _prune_timm_model(self):
        """
        Prune a TIMM model using prune_timm.py.
        """
        import prune_timm

        # Extract parameters from SmashConfig
        model_name = self.model_id
        pruning_ratio = self.smash_config.get('pruning_ratio', 0.5)
        global_pruning = self.smash_config.get('global_pruning', False)
        save_model_path = self.smash_config.get('save_model', None)

        # Perform pruning via prune_timm
        prune_timm.prune_timm_model(model_name, pruning_ratio, global_pruning, save_model=save_model_path)

        logger.info(f"Pruned TIMM model {model_name} with ratio {pruning_ratio}")
        if save_model_path:
            logger.info(f"Pruned model saved to {save_model_path}")


    def _prune_llm_model(self):
        """
        Prune a large language model (LLM) using LLM-Pruner.
        """
        model_type = self.smash_config.get('model_type')

        # Add support for various LLMs based on the selected model type
        if model_type in ['llama', 'bloom', 'vicuna']:
            self._prune_llm_pruner()
        else:
            raise ValueError(f"Unsupported LLM model type for pruning: {model_type}")

    def _prune_llm_pruner(self):
      """
      Use LLM-Pruner to prune LLaMA, BLOOM, Vicuna, etc. models.
      """
      # Setup the argument parser and pass the args programmatically
      parser = argparse.ArgumentParser(description='Pruning LLaMA (huggingface version)')

      args = {
          'base_model': self.model_id,
          'pruning_ratio': self.smash_config.get('pruning_ratio', 0.5),
          'pruner_type': self.smash_config.get('pruner_type', 'l2'),
          'block_wise': self.smash_config.get('block_wise', False),
          'channel_wise': self.smash_config.get('channel_wise', False),
          'layer_wise': self.smash_config.get('layer_wise', False),
          'device': self.smash_config.get('device', 'cuda'),
          'eval_device': self.smash_config.get('eval_device', 'cuda'),
          'save_model': self.smash_config.get('save_model', True),
          'test_before_train': self.smash_config.get('test_before_train', False),
          'test_after_train': self.smash_config.get('test_after_train', True),
          'block_attention_layer_start': self.smash_config.get('block_attention_layer_start', 3),
          'block_attention_layer_end': self.smash_config.get('block_attention_layer_end', 31),
          'block_mlp_layer_start': self.smash_config.get('block_mlp_layer_start', 3),
          'block_mlp_layer_end': self.smash_config.get('block_mlp_layer_end', 31),
          'layer': self.smash_config.get('layer', 12),
          'iterative_steps': self.smash_config.get('iterative_steps', 1),
          'grouping_strategy': self.smash_config.get('grouping_strategy', 'sum'),
          'global_pruning': self.smash_config.get('global_pruning', False),
          'taylor': self.smash_config.get('taylor', 'param_first'),
          'seed': self.smash_config.get('seed', 42),
      }

      # Convert dictionary args to namespace
      args_namespace = argparse.Namespace(**args)

      # Call the main pruning function from hf_prune.py
      # Parsing only known args to avoid Jupyter/Colab unrecognized arguments
      prune_llm_main(args_namespace, model=self.model_id, pruning_ratio=pruning_ratio, seed=seed)



    def _evaluate_zero_shot(self):
        """
        Evaluate the pruned model in a zero-shot setting.
        """
        logger.info("Evaluating the pruned model in zero-shot setting.")
        # TODO: Implement evaluation logic here

        # This is highly dependent on the specific task and dataset
        # For demonstration, we'll perform a simple generation

        input_text = "Once upon a time"
        input_ids = self.tokenizer.encode(input_text, return_tensors='pt').to(self.smash_config.get('device', 'cpu'))
        output = self.model.generate(input_ids, max_length=50)
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        logger.info(f"Generated text: {generated_text}")


    def _quantize_model(self):
        """
        Quantize the model using TorchAoConfig and optionally compile it for further optimization.
        """
        model_id = self.model_id or self.smash_config.get('model_id')
        quant_type = self.smash_config.get('quant_type', 'int4_weight_only')
        group_size = self.smash_config.get('group_size', 128)
        torch_dtype = self.smash_config.get('torch_dtype', torch.bfloat16)
        device = self.smash_config.get('device', 'cuda')
        
        # Set up the quantization configuration
        quantization_config = TorchAoConfig(
            quant_type=quant_type,
            group_size=group_size
        )

        logger.info(f"Loading quantized model: {model_id} with {quant_type} and group size {group_size}")

        # Load the quantized model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            quantization_config=quantization_config,
            device_map=device
        )

        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        logger.info(f"Model loaded and quantized. Applying further optimizations...")

        # Apply compilation (optional)
        compile_model = self.smash_config.get('compile', False)
        if compile_model:
            logger.info(f"Compiling model forward function with torch.compile...")
            self.model.forward = torch.compile(
                self.model.forward, 
                mode="reduce-overhead", 
                fullgraph=True
            )

        logger.info(f"Model optimized and ready for inference.")

        # Optionally save the quantized model
        save_model_path = self.smash_config.get('save_model', None)
        if save_model_path:
            self.model.save_pretrained(save_model_path)
            self.tokenizer.save_pretrained(save_model_path)
            logger.info(f"Quantized model saved to {save_model_path}")


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
