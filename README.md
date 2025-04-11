# Scaling Down Text Encoder of Text-To-Image Diffusion Models [[Paper](https://arxiv.org/abs/2503.19897)] [[ComfyUI](https://github.com/LifuWang-66/DistillT5ComfyUI)] [[HuggingFace](https://huggingface.co/LifuWang/DistillT5)]
This repository provides the implementation for our paper "Scaling Down Text Encoder of Text-To-Image Diffusion Models". We replace the large T5-XXL in Flux with T5-Base, achieving 50 times reduction in model size.

## Install environment
```shell
conda create -n distillt5 python=3.12
conda activate distillt5
pip install -r requirements.txt
```

Diffusers may occasionally raise a 'no attribute _execution_device' error when using custom pipelines. For more details, refer to this [issue](https://github.com/huggingface/diffusers/issues/9180). To resolve this, we recommend replacing all instances of _execution_device in your pipeline with self.transformer.device, or installing our modified version of Diffusers.
```shell
pip install ./diffusers
```

## Inference Script
```python
python inference_flux.py --ckpt_name $ckpt_name
``` 
## Example Usage
```python
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.T5_encoder import T5EncoderWithProjection
import torch
from diffusers import FluxPipeline


pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.float16)
text_encoder = T5EncoderWithProjection.from_pretrained('LifuWang/DistillT5', torch_dtype=torch.float16)
pipe.text_encoder_2 = text_encoder
pipe = pipe.to('cuda')

prompt = "Photorealistic portrait of a stylish young woman wearing a futuristic golden sequined bodysuit that catches the light, creating a metallic, mirror-like effect. She is wearing large, reflective blue-tinted aviator sunglasses. Over her head, she wears headphones with metallic accents, giving a modern, cyber aesthetic."

image = pipe(prompt=prompt, num_images_per_prompt=1, guidance_scale=3.5, num_inference_steps=20).images[0]

image.save("t5_base.png")
``` 