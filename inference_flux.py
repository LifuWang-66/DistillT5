import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from transformers import T5EncoderModel
from models.T5_encoder import T5EncoderWithProjection
import argparse
import torch

sys.path.insert(0, './diffusers/src')
from diffusers import FluxPipeline


parser = argparse.ArgumentParser()
parser.add_argument('--ckpt_name', type=str, default="flux_flan_t5_base_512_combined_continue_iter180000")
parser.add_argument('--device', type=str, default="cuda")
parser.add_argument('--seed', type=int, default=4396)
parser.add_argument('--use_bias', action='store_true')
parser.add_argument('--project_in_dim', type=int, default=768)
parser.add_argument('--resolution', type=int, default=1024)
args = parser.parse_args()

device = args.device
pipe = FluxPipeline.from_pretrained("/pfs/models/FLUX.1-dev", torch_dtype=torch.float16)
pipe = pipe.to(device)

torch.cuda.manual_seed(args.seed)
prompt = ["Snapshot of ferret wearing timberland boots, blue aviation bomber jacket, rayban sunglasses, mountain view, sunlight"]


text_encoder = T5EncoderModel.from_pretrained("/pfs/models/flan-t5-base", torch_dtype=torch.float16)
text_encoder = T5EncoderWithProjection(text_encoder, args)
state_dict = torch.load(os.path.join('../DistillT5/ckpt', args.ckpt_name + ".pth"))
text_encoder.load_state_dict(state_dict)
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

