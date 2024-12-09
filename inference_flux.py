import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.T5_encoder import T5EncoderWithProjection
import argparse
import torch
from diffusers import FluxPipeline


parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default="cuda")
parser.add_argument('--seed', type=int, default=4396)   
parser.add_argument('--resolution', type=int, default=1024)
args = parser.parse_args()

device = args.device
pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.float16)
pipe = pipe.to(device)

torch.cuda.manual_seed(args.seed)
prompt = "Photorealistic portrait of a stylish young woman wearing a futuristic golden sequined bodysuit that catches the light, creating a metallic, mirror-like effect. She is wearing large, reflective blue-tinted aviator sunglasses. Over her head, she wears headphones with metallic accents, giving a modern, cyber aesthetic."

text_encoder = T5EncoderWithProjection.from_pretrained('lwang717/ScalingDownTextEncoder', torch_dtype=torch.float16)
pipe.text_encoder_2 = text_encoder.to(device)

torch.cuda.manual_seed(args.seed)
image = pipe(prompt=prompt, 
    height=args.resolution,
    width=args.resolution,
    num_images_per_prompt=1, 
    guidance_scale=3.5,
    num_inference_steps=20
).images[0]

image.save("t5_base.png")

