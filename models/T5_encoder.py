import torch
from transformers import T5EncoderModel, T5Config, T5PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput
from typing import List, Optional, Tuple, Union
from torch import nn, Tensor

class T5ProjectionConfig(T5Config):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.project_in_dim = kwargs.get("project_in_dim", 768)
        self.project_out_dim = kwargs.get("out_dim", 4096)

class T5EncoderWithProjection(T5PreTrainedModel):
    config_class = T5ProjectionConfig

    def __init__(self, config):
        super().__init__(config)
        # self.encoder = encoder
        self.encoder = T5EncoderModel(config)
        
        self.final_projection = nn.Sequential(
            nn.Linear(config.project_in_dim, config.project_out_dim, bias=False), 
            nn.ReLU(), 
            nn.Dropout(0.1), 
            nn.Linear(config.project_out_dim, config.project_out_dim, bias=False) 
        )
        
        
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], BaseModelOutput]:
        r"""
        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, T5EncoderModel

        >>> tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
        >>> model = T5EncoderModel.from_pretrained("google-t5/t5-small")
        >>> input_ids = tokenizer(
        ...     "Studies have been shown that owning a dog is good for you", return_tensors="pt"
        ... ).input_ids  # Batch size 1
        >>> outputs = model(input_ids=input_ids)
        >>> last_hidden_states = outputs.last_hidden_state
        ```"""
        return_dict = return_dict if return_dict is not None else False

        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        last_hidden_state = self.final_projection(encoder_outputs[0])
        # last_hidden_state = self.final_block(last_hidden_state)[0]

        if not return_dict:
            return tuple(
                v for v in [last_hidden_state] if v is not None
            )
        
        return BaseModelOutput(
            last_hidden_state=last_hidden_state
        )
