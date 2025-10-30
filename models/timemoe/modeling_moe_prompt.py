import torch
from torch import nn
from transformers import PreTrainedModel, PretrainedConfig
from .modeling_time_moe import TimeMoeForPrediction
from models.cov_prompt_generator import CovariatePromptGenerator

class CovPromptConfig(PretrainedConfig):
    
    def __init__(self, foundation_model_path=None, prompt_config=None, **kwargs):
        self.foundation_model_path = foundation_model_path
        self.prompt_config = prompt_config
        super().__init__(**kwargs)

class CovPromptTimeMoE(PreTrainedModel):
    config_class = CovPromptConfig
    def __init__(self, config: CovPromptConfig):
        super().__init__(config)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.foundation_model = TimeMoeForPrediction.from_pretrained(
            config.foundation_model_path,
            device_map=device,
            attn_implementation='flash_attention_2',
            torch_dtype=torch.float16,  
        )
        for param in self.foundation_model.parameters():
            param.requires_grad = False
        self.foundation_model.eval()
        self.embedding_layer = self.foundation_model.model.embed_layer
        self.prompt_generator = CovariatePromptGenerator(**config.prompt_config)
        self.prompt_generator.to(device=device, dtype=torch.float32)
        self.foundation_dtype = torch.float16  
        self.prompt_dtype = torch.float32      
        self.model_device = device

    def _apply_embedding(self, inputs):
        inputs = inputs.to(device=self.model_device, dtype=self.foundation_dtype)
        emb_output = self.embedding_layer.emb_layer(inputs)
        gate_output = self.embedding_layer.gate_layer(inputs)
        result = self.embedding_layer.act_fn(gate_output) * emb_output
 
        return result

    def forward(self, inputs, all_variables_history, variable_idx, **kwargs):
        all_variables_history_f32 = all_variables_history.to(
            device=self.model_device, dtype=self.prompt_dtype
        )
        variable_idx = variable_idx.to(device=self.model_device)
        prompt_embeds, attn_weights = self.prompt_generator(all_variables_history_f32, variable_idx)
        inputs_f16 = inputs.to(device=self.model_device, dtype=self.foundation_dtype)
        target_embeds = self._apply_embedding(inputs_f16.unsqueeze(-1))
        prompt_embeds_f16 = prompt_embeds.to(dtype=self.foundation_dtype)
        inputs_embeds = torch.cat([prompt_embeds_f16, target_embeds], dim=1)
        batch_size = inputs_embeds.shape[0]
        attention_mask = torch.ones(
            batch_size, inputs_embeds.shape[1], 
            device=self.model_device, 
            dtype=self.foundation_dtype
        )

        hidden_states = self.foundation_model.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask
        ).last_hidden_state
        
        return hidden_states, attn_weights

    @property
    def device(self):
        return self.model_device
