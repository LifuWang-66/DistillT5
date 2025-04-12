# This script is modified from https://github.com/huggingface/diffusers. Thanks for their great work!

#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import copy
import gc
import itertools
import logging
import math
import os
import random
import shutil
import warnings
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from huggingface_hub.utils import insecure_hashlib
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm
from transformers import CLIPTextModelWithProjection, CLIPTokenizer, PretrainedConfig, T5EncoderModel, T5TokenizerFast

import sys
sys.path.insert(0, './diffusers/src')
import diffusers
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxPipeline,
    FluxTransformer2DModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
from diffusers.utils import (
    check_min_version,
    is_wandb_available,
)
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.torch_utils import is_compiled_module

import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.T5_encoder import T5EncoderWithProjection, T5ProjectionConfig

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.31.0.dev0")

logger = get_logger(__name__)


class CombinedDataset(Dataset):
    def __init__(self, data_path1=None, data_path2=None, data_path3=None):
        """
        Args:
            data_dir (string): Directory with all the training files.
        """
        self.prompts = []
        
        if data_path1 != None:
            with open(data_path1, 'r') as file:
                data = json.load(file)
                self.prompts.extend(data)
        
        if data_path2 != None:
            for file_name in os.listdir(data_path2):
                if 'train' in file_name:
                    file_path = os.path.join(data_path2, file_name)
                    with open(file_path, 'r') as f:
                        if data_path1 != None and data_path3 != None:
                            self.prompts.extend(f.readlines() * 10)
                        else:
                            self.prompts.extend(f.readlines())

        if data_path3 != None:
            with open(data_path3, 'r') as file:
                data = json.load(file)
                self.prompts.extend(data)


    def __len__(self):
        """Returns the total number of prompts."""
        return len(self.prompts)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the prompt to retrieve.
        
        Returns:
            prompt (string): The prompt at the specified index.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        prompt = self.prompts[idx]
        return prompt
        
def load_text_encoders(class_one, class_two):
    text_encoder_one = class_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    text_encoder_two = class_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant
    )
    return text_encoder_one, text_encoder_two


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="black-forest-labs/FLUX.1-dev",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--train_transformer", action="store_true")
    parser.add_argument("--train_text_encoder", action="store_true")
    parser.add_argument('--student_encoder', type=str, default="google/flan-t5-base")
    parser.add_argument('--load_student_checkpoint', type=str)
    parser.add_argument('--save_name', type=str, default="experiment")
    parser.add_argument('--load_transformer', type=str)
    parser.add_argument('--num_inference_steps', type=int, default=20)
    parser.add_argument("--text_encoder_lr", type=float, default=1e-4, help="Text encoder learning rate to use.")
    parser.add_argument("--min_guidance_scale", type=float, default=2.0)
    parser.add_argument("--max_guidance_scale", type=float, default=5.0)
    parser.add_argument("--interval", type=float, default=0.5, help="interval for guidance values")
    parser.add_argument('--laion_path', type=str, default="data/laion_6.5.json")
    parser.add_argument('--compbench_path', type=str, default="data/T2I-CompBench")
    parser.add_argument('--commontext_path', type=str, default="data/CommonText_Train.json")
    

    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )


    parser.add_argument("--repeats", type=int, default=1, help="How many times to repeat the training data.")
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=77,
        help="Maximum sequence length to use with with the T5 text encoder",
    )

    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="flux-dreambooth",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )

    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )

    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="none",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
        help=('We default to the "none" weighting scheme for uniform sampling and uniform loss'),
    )
    parser.add_argument(
        "--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        help=('The optimizer type to use. Choose between ["AdamW", "prodigy"]'),
    )

    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW",
    )

    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--prodigy_beta3",
        type=float,
        default=None,
        help="coefficients for computing the Prodigy stepsize using running averages. If set to None, "
        "uses the value of square root of beta2. Ignored if optimizer is adamW",
    )
    parser.add_argument("--prodigy_decouple", type=bool, default=True, help="Use AdamW style decoupled weight decay")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-04, help="Weight decay to use for unet params")
    parser.add_argument(
        "--adam_weight_decay_text_encoder", type=float, default=1e-03, help="Weight decay to use for text_encoder"
    )

    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )

    parser.add_argument(
        "--prodigy_use_bias_correction",
        type=bool,
        default=True,
        help="Turn on Adam's bias correction. True by default. Ignored if optimizer is adamW",
    )
    parser.add_argument(
        "--prodigy_safeguard_warmup",
        type=bool,
        default=True,
        help="Remove lr from the denominator of D estimate to avoid issues during warm-up stage. True by default. "
        "Ignored if optimizer is adamW",
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
   
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if not args.train_text_encoder and not args.train_transformer:
        ValueError("Set at least one of `--train_text_encoder` and `--train_transformer`")

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def _encode_prompt_with_t5(
    text_encoder,
    tokenizer,
    max_sequence_length=512,
    prompt=None,
    num_images_per_prompt=1,
    device=None,
    text_input_ids=None,
    dtype=torch.float32
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    prompt_embeds = text_encoder(text_input_ids.to(device))[0]
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape

    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds, text_input_ids


def _encode_prompt_with_clip(
    text_encoder,
    tokenizer,
    prompt: str,
    device=None,
    text_input_ids=None,
    num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=False)

    # Use pooled output of CLIPTextModel
    prompt_embeds = prompt_embeds.pooler_output
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

    return prompt_embeds


def encode_prompt(
    text_encoders,
    tokenizers,
    prompt: str,
    max_sequence_length,
    device=None,
    num_images_per_prompt: int = 1,
    text_input_ids_list=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)
    dtype = text_encoders[0].dtype
    device = device if device is not None else text_encoders[1].device
    pooled_prompt_embeds = _encode_prompt_with_clip(
        text_encoder=text_encoders[0],
        tokenizer=tokenizers[0],
        prompt=prompt,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        text_input_ids=text_input_ids_list[0] if text_input_ids_list else None,
    )

    prompt_embeds, input_ids = _encode_prompt_with_t5(
        text_encoder=text_encoders[1],
        tokenizer=tokenizers[1],
        max_sequence_length=max_sequence_length,
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        device=device,
        text_input_ids=text_input_ids_list[1] if text_input_ids_list else None,
        dtype=dtype
    )

    student_prompt_embeds, input_ids = _encode_prompt_with_t5(
        text_encoder=text_encoders[2],
        tokenizer=tokenizers[1],
        max_sequence_length=max_sequence_length,
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        device=device,
        text_input_ids=input_ids,
        dtype=dtype
    )
    

    text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)

    return prompt_embeds, pooled_prompt_embeds, text_ids, student_prompt_embeds


def main(args):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
            ).repo_id

    # Load the tokenizers
    tokenizer_one = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )
    tokenizer_two = T5TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
    )

    # import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )

    # Load scheduler and models
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    text_encoder_one, text_encoder_two = load_text_encoders(text_encoder_cls_one, text_encoder_cls_two)

    
    text_encoder_two_student = T5EncoderModel.from_pretrained(args.student_encoder)
    config = T5ProjectionConfig.from_pretrained("config.json")
    text_encoder_two_student = T5EncoderWithProjection(config).to(accelerator.device)

    if args.load_student_checkpoint != None:
        state_dict = torch.load(os.path.join('ckpt', args.load_student_checkpoint + ".pth"), map_location='cpu')
        text_encoder_two_student.load_state_dict(state_dict)

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )
    

    if args.load_transformer:
        transformer = FluxTransformer2DModel.from_pretrained(
            args.load_transformer, subfolder="transformer", revision=args.revision, variant=args.variant
        )
    else:
        transformer = FluxTransformer2DModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="transformer", revision=args.revision, variant=args.variant
        )
    
    if args.train_transformer:
        transformer.requires_grad_(True)
    else:
        transformer.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    if args.train_text_encoder:
        text_encoder_two_student.requires_grad_(True)
    else:
        text_encoder_two_student.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weights (vae, text_encoder and transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two_student.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)
    
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder_one.gradient_checkpointing_enable()

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    params_to_optimize = []
    if args.train_transformer:
        transformer_parameters_with_lr = {"params": transformer.parameters(), "lr": args.learning_rate}
        params_to_optimize.append(transformer_parameters_with_lr)
    if args.train_text_encoder:
        # different learning rate for text encoder and unet
        text_parameters_with_lr = {
            "params": text_encoder_two_student.parameters(),
            "weight_decay": args.adam_weight_decay_text_encoder,
            "lr": args.text_encoder_lr if args.text_encoder_lr else args.learning_rate,
        }
        params_to_optimize.append(text_parameters_with_lr)

    # Optimizer creation
    if not (args.optimizer.lower() == "prodigy" or args.optimizer.lower() == "adamw"):
        logger.warning(
            f"Unsupported choice of optimizer: {args.optimizer}.Supported optimizers include [adamW, prodigy]."
            "Defaulting to adamW"
        )
        args.optimizer = "adamw"

    if args.use_8bit_adam and not args.optimizer.lower() == "adamw":
        logger.warning(
            f"use_8bit_adam is ignored when optimizer is not set to 'AdamW'. Optimizer was "
            f"set to {args.optimizer.lower()}"
        )

    if args.optimizer.lower() == "adamw":
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    if args.optimizer.lower() == "prodigy":
        try:
            import prodigyopt
        except ImportError:
            raise ImportError("To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`")

        optimizer_class = prodigyopt.Prodigy

        if args.learning_rate <= 0.1:
            logger.warning(
                "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
            )
        if args.train_text_encoder and args.text_encoder_lr:
            logger.warning(
                f"Learning rates were provided both for the transformer and the text encoder- e.g. text_encoder_lr:"
                f" {args.text_encoder_lr} and learning_rate: {args.learning_rate}. "
                f"When using prodigy only learning_rate is used as the initial learning rate."
            )
            # changes the learning rate of text_encoder_parameters_one and text_encoder_parameters_two to be
            # --learning_rate
            params_to_optimize[1]["lr"] = args.learning_rate
            params_to_optimize[2]["lr"] = args.learning_rate

        optimizer = optimizer_class(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            beta3=args.prodigy_beta3,
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
            decouple=args.prodigy_decouple,
            use_bias_correction=args.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup,
        )    
    
    dataset = CombinedDataset(
        data_path1=args.laion_path,
        data_path2=args.compbench_path,
        data_path3=args.commontext_path
    )
    train_dataloader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) * args.num_inference_steps / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )
    
    # Prepare everything with our `accelerator`.
    if args.train_text_encoder and args.train_transformer:
        (
            transformer,
            text_encoder_two_student,
            optimizer,
            train_dataloader,
            lr_scheduler,
        ) = accelerator.prepare(
            transformer,
            text_encoder_two_student,
            optimizer,
            train_dataloader,
            lr_scheduler,
        )
    elif args.train_text_encoder:
        text_encoder_two_student, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            text_encoder_two_student, optimizer, train_dataloader, lr_scheduler
        )
    elif args.train_transformer:
        transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            transformer, optimizer, train_dataloader, lr_scheduler
        )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) * args.num_inference_steps / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_name = "flux-dev"
        accelerator.init_trackers(tracker_name, config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0
    initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    pipeline = FluxPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        text_encoder=accelerator.unwrap_model(text_encoder_one),
        text_encoder_2=accelerator.unwrap_model(text_encoder_two),
        transformer=accelerator.unwrap_model(transformer),
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )
    pipeline.text_encoder_2 = accelerator.unwrap_model(text_encoder_two_student)
    pipeline = pipeline.to(accelerator.device)


    def calculate_shift(
        image_seq_len,
        base_seq_len: int = 256,
        max_seq_len: int = 4096,
        base_shift: float = 0.5,
        max_shift: float = 1.16,
    ):
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        mu = image_seq_len * m + b
        return mu
        
    models_to_accumulate = []
    if args.train_transformer:
        models_to_accumulate.extend([transformer])
    if args.train_text_encoder:
        models_to_accumulate.extend([text_encoder_two_student])


    guidance_values = [args.min_guidance_scale + i * args.interval for i in range(int((args.max_guidance_scale - args.min_guidance_scale) / args.interval) + 1)]

    for epoch in range(first_epoch, args.num_train_epochs):
        if args.train_transformer:
            transformer.train()
        if args.train_text_encoder:
            text_encoder_two_student.train()
        for step, prompt in enumerate(train_dataloader):
            shape = (
                args.train_batch_size,
                transformer.config.in_channels // 4,
                2 * int(args.resolution) // pipeline.vae_scale_factor,
                2 * int(args.resolution) // pipeline.vae_scale_factor,
            )
            model_input = torch.randn(shape, dtype=text_encoder_two.dtype, device=accelerator.device)
            latent_image_ids = FluxPipeline._prepare_latent_image_ids(
                model_input.shape[0],
                model_input.shape[2],
                model_input.shape[3],
                accelerator.device,
                weight_dtype,
            )

            packed_model_input = FluxPipeline._pack_latents(
                model_input,
                batch_size=model_input.shape[0],
                num_channels_latents=model_input.shape[1],
                height=model_input.shape[2],
                width=model_input.shape[3],
            )

            num_inference_steps = args.num_inference_steps 
            image_seq_len = packed_model_input.shape[1]
            mu = calculate_shift(
                image_seq_len,
                noise_scheduler.config.base_image_seq_len,
                noise_scheduler.config.max_image_seq_len,
                noise_scheduler.config.base_shift,
                noise_scheduler.config.max_shift,
            )
            noise_scheduler.set_timesteps(num_inference_steps, device=accelerator.device, mu=mu)
            timesteps = noise_scheduler.timesteps
           
            if transformer.config.guidance_embeds:
                guidance = torch.tensor([random.choice(guidance_values)], device=accelerator.device)
                guidance = guidance.expand(packed_model_input.shape[0])
            else:
                guidance = None
            for i, t in enumerate(timesteps):
                with accelerator.accumulate(models_to_accumulate):
                    prompt_embeds, pooled_prompt_embeds, text_ids, student_prompt_embeds = encode_prompt(
                        text_encoders=[text_encoder_one, text_encoder_two, text_encoder_two_student],
                        tokenizers=[tokenizer_one, tokenizer_two],
                        max_sequence_length=args.max_sequence_length,
                        prompt=prompt,
                        device=accelerator.device
                    )

                    pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds, device=accelerator.device)

                    current_timestep = torch.tensor([t], device=accelerator.device, dtype=weight_dtype)
                    current_timestep = current_timestep.expand(packed_model_input.shape[0])
                    # Predict the noise residual
                    with torch.no_grad():
                        teacher_model_pred = transformer(
                            hidden_states=packed_model_input,
                            timestep=current_timestep / 1000,
                            guidance=guidance,
                            pooled_projections=pooled_prompt_embeds,
                            encoder_hidden_states=prompt_embeds,
                            txt_ids=text_ids,
                            img_ids=latent_image_ids,
                            return_dict=False,
                        )[0]

                    student_model_pred = transformer(
                        hidden_states=packed_model_input,
                        timestep=current_timestep / 1000,
                        guidance=guidance,
                        pooled_projections=pooled_prompt_embeds,
                        encoder_hidden_states=student_prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=latent_image_ids,
                        return_dict=False,
                    )[0]
                    
                    target = teacher_model_pred

                    # Compute regular loss.
                    loss = torch.mean(
                        ((student_model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                        1,
                    )
                    
                    loss = loss.mean()

                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        params_to_clip = (
                            itertools.chain(transformer.parameters(), text_encoder_one.parameters())
                            if args.train_text_encoder
                            else transformer.parameters()
                        )
                        accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    
                    packed_model_input = noise_scheduler.step(teacher_model_pred, t, packed_model_input, return_dict=False)[0]

                        
                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                    if accelerator.is_main_process:
                        if global_step % args.checkpointing_steps == 0:
                            if args.train_transformer:
                                unwrapped_transformer = accelerator.unwrap_model(transformer)
                                save_path = os.path.join('ckpt', f'transformer_iter{global_step}.pth')
                                torch.save(unwrapped_transformer.state_dict(), save_path)
                                logger.info(f"Saved transformer state to {save_path}")

                            if args.train_text_encoder:
                                unwrapped_text_encoder = accelerator.unwrap_model(text_encoder_two_student)
                                save_path = os.path.join('ckpt', f'{args.save_name}_iter{global_step}.pth')
                                torch.save(unwrapped_text_encoder.state_dict(), save_path)
                                logger.info(f"Saved text encoder state to {save_path}")

                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
            
            if global_step >= args.max_train_steps:
                break


    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if args.train_transformer:
                unwrapped_transformer = accelerator.unwrap_model(transformer)
                save_path = os.path.join('ckpt', f'transformer_{args.save_name}.pth')
                torch.save(unwrapped_transformer.state_dict(), save_path)
                logger.info(f"Saved transformer state to {save_path}")
        if args.train_text_encoder:
            unwrapped_text_encoder = accelerator.unwrap_model(text_encoder_two_student)
            save_path = os.path.join('ckpt', f'{args.save_name}.pth')
            torch.save(unwrapped_text_encoder.state_dict(), save_path)
            logger.info(f"Saved text encoder state to {save_path}")
    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)