import torch
from models.T5_encoder import T5EncoderWithProjection, T5ProjectionConfig
state_dict = torch.load("ckpt/flux_flan_t5_base_512_combined_continue_iter180000.pth")
config = T5ProjectionConfig.from_pretrained('ckpt/config.json')
model = T5EncoderWithProjection(config)
model.load_state_dict(state_dict)
model.push_to_hub("ScalingDownTextEncoder")