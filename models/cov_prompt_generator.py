import torch
from torch import nn
import torch.nn.functional as F


class TimeAttentionPool(nn.Module):
    def __init__(self, input_dim: int, attn_dim: int):
        super().__init__()
        self.query_proj = nn.Linear(input_dim, attn_dim)
        self.key_proj = nn.Linear(input_dim, attn_dim)
        self.value_proj = nn.Linear(input_dim, input_dim)
        self.scale = attn_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, L, C = x.shape
        x_flat = x.view(B * N, L, C)  
        q_source = x_flat.mean(dim=1, keepdim=True)   

        q = self.query_proj(q_source)                    
        k = self.key_proj(x_flat)                        
        v = self.value_proj(x_flat)                       

        attn_scores = torch.bmm(q, k.transpose(1, 2)) * self.scale  
        attn_weights = F.softmax(attn_scores, dim=-1)               
        summary_vector = torch.bmm(attn_weights, v).squeeze(1)     

        return summary_vector.view(B, N, C)


class CovariatePromptGenerator(nn.Module):
    def __init__(
        self,
        num_variables: int,
        id_dim: int = 32,
        ts_dim: int = 64,
        attn_dim: int = 128,
        prompt_len: int = 10,
        model_dim: int = 384,
        num_attn_heads: int = 4,
    ) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.prompt_len = prompt_len
        self.num_variables = num_variables
        self.ts_dim = ts_dim


        self.id_embeddings = nn.Embedding(num_variables, id_dim)
        self.time_feature_extractor = nn.Sequential(
            nn.Conv1d(
                num_variables, 
                num_variables * ts_dim, 
                kernel_size=8, 
                padding="same", 
                groups=num_variables
            ),
            nn.GELU(),
        )
        
        self.time_summarizer = TimeAttentionPool(input_dim=ts_dim, attn_dim=ts_dim)

        self.context_dim = id_dim + ts_dim
        self.k_proj = nn.Linear(self.context_dim, attn_dim, bias=False)
        self.v_proj = nn.Linear(self.context_dim, attn_dim, bias=False)

        self.prompt_queries = nn.Parameter(torch.randn(1, self.prompt_len, attn_dim))
        self.query_modulator = nn.Sequential(
            nn.Linear(self.context_dim, attn_dim),
            nn.GELU(),
            nn.Linear(attn_dim, attn_dim)
        )

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=attn_dim, 
            num_heads=num_attn_heads, 
            batch_first=True
        )

        self.prompt_projector = nn.Sequential(
            nn.Linear(attn_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim) 
        )

    def forward(
        self,
        all_variables_history: torch.Tensor, 
        target_variable_indices: torch.Tensor,  
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, L, N = all_variables_history.shape
        device = all_variables_history.device

        all_var_indices = torch.arange(N, device=device)
        id_embeds = self.id_embeddings(all_var_indices)              

        ts_input = all_variables_history.transpose(1, 2)              
        time_features = self.time_feature_extractor(ts_input)         
        time_features = time_features.view(B, N, self.ts_dim, L).permute(0, 1, 3, 2) 
        ts_summary = self.time_summarizer(time_features)              

        id_embeds_expanded = id_embeds.unsqueeze(0).expand(B, -1, -1) 

        kv_source = torch.cat([id_embeds_expanded, ts_summary], dim=-1) 
        keys = self.k_proj(kv_source)                                  
        values = self.v_proj(kv_source)                                 

        batch_indices = torch.arange(B, device=device)
        tgt_id_embed = id_embeds_expanded[batch_indices, target_variable_indices]  
        tgt_ts_summary = ts_summary[batch_indices, target_variable_indices]        
        tgt_context = torch.cat([tgt_id_embed, tgt_ts_summary], dim=-1)          

        modulation_vector = self.query_modulator(tgt_context)                   
        queries = self.prompt_queries.expand(B, -1, -1) + modulation_vector.unsqueeze(1)  

        context_vectors, attn_weights = self.cross_attention(queries, keys, values)
        prompt_embeds = self.prompt_projector(context_vectors)                     

        return prompt_embeds, attn_weights
