import torch
import torch.nn as nn
import torch.nn.functional as F


class NoRGa_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, prompt, act_scale, gate_act):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)


        if prompt is not None:
            # prefix key, value
            prompt = prompt.permute(1, 0, 3, 2, 4).contiguous() # 2, B, num_heads, prompt_length, C // num_heads

            key_prefix = prompt[0] # B, num_heads, prompt_length, embed_dim // num_heads
            value_prefix = prompt[1] # B, num_heads, prompt_length, embed_dim // num_heads

            # k = torch.cat([key_prefix, k], dim=2)
            v = torch.cat([value_prefix, v], dim=2)

            prompt_attn = (q @ key_prefix.transpose(-2, -1)) * self.scale # B, num_heads, N, prompt_length

            prompt_attn = (prompt_attn + gate_act(prompt_attn * act_scale[0]) * act_scale[1])

            attn = (q @ k.transpose(-2, -1)) * self.scale # B, num_heads, N, N
            attn = torch.cat([prompt_attn, attn], dim=-1)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = attn.softmax(dim=-1)  # B, num_heads, N, N + prompt_length

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    

    # def get_transformed_prompt(self, prompt):
    #     prompt = prompt.permute(0, 2, 1, 3) # B, prompt_length, num_heads, C // num_heads
    #     B, prompt_length, _, _ = prompt.shape
    #     prompt = prompt.reshape(B, prompt_length, -1) # B, prompt_length, C

    #     C = prompt.shape[-1]

    #     qkv = self.qkv(prompt).reshape(B, prompt_length, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    #     q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

    #     return q, k, v
    
    # def normalize_prompt(self, prompt):
    #     prompt = prompt.permute(0, 2, 1, 3) # B, prompt_length, num_heads, C // num_heads
    #     B, prompt_length, _, _ = prompt.shape
    #     prompt = prompt.reshape(B, prompt_length, -1) # B, prompt_length, C
        
    #     prompt = prompt / prompt.norm(dim=-1, keepdim=True, p=2).detach()

    #     return prompt.reshape(B, prompt_length, self.num_heads, -1).permute(0, 2, 1, 3) # B, num_heads, prompt_length, C // num_heads
    


class PreT_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, prompt):

        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        # V: B, num_heads, N, C // num_heads

        ## Baseline ###
        if prompt is not None:
            # prefix key, value
            prompt = prompt.permute(1, 0, 3, 2, 4).contiguous() # 2, B, num_heads, prompt_length, C // num_heads

            key_prefix = prompt[0] # B, num_heads, prompt_length, embed_dim // num_heads
            value_prefix = prompt[1] # B, num_heads, prompt_length, embed_dim // num_heads

            k = torch.cat([key_prefix, k], dim=2)
            v = torch.cat([value_prefix, v], dim=2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)  # B, num_heads, N, N + prompt_length

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x