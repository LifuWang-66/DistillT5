import torch
from models.T5_encoder import T5EncoderWithProjection, T5ProjectionConfig
state_dict = torch.load("ckpt/t5_xl/flux_flan_t5_xl_512_combined_iter200000.pth")
config = T5ProjectionConfig.from_pretrained('ckpt/t5_xl/config.json', project_in_dim=2048)
model = T5EncoderWithProjection(config)
model.load_state_dict(state_dict)
model.push_to_hub("DistillT5-XL")