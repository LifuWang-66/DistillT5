import torch
from models.T5_encoder import T5EncoderWithProjection, T5ProjectionConfig

state_dict = torch.load("ckpt/flux_flan_t5_base_570kiter.pth")

config = T5ProjectionConfig.from_pretrained('/pfs/models/flan-t5-base/config.json')
model = T5EncoderWithProjection(config)

# Load the state dictionary
model.load_state_dict(state_dict)

# Save to Hugging Face format
model.save_pretrained("./my_hf_model")