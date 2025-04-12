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
Flux is now gated. Login with your HF token to get permission
```shell
huggingface-cli login
```
## Inference Script
Then run the inference script
```python
python inference_flux.py
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


## Training Script
Since the training has 3 stages, you should use the  datasets accordingly. For stage 1, use T2I-CompBench only；for stage 2, use CommonText only (resolution 1024); for stage 3, use all 3 datasets.
```python
accelerate launch train_flux.py  \
    --train_text_encoder \
    --mixed_precision bf16 \
    --train_batch_size 4 \
    --resolution 512 \
    --text_encoder_lr 1e-4 \
    --laion_path data/laion_6.5.json \
    --compbench_path data/T2I-CompBench \
    --commontext_path data/CommonText_Train.json \
    --num_train_epochs 1 
