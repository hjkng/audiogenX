import torch
import math

# Efficient implementation equivalent to the following:
def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, is_training=False, text_mask=None) -> torch.Tensor:
    # Efficient implementation equivalent to the following:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype).to(query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0).to(query.device)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias

    attn_weight = torch.softmax(attn_weight, dim=-1).clone()

    # Multiply mask after softmax function
    if text_mask is not None:
        B, H, D, T = attn_weight.shape
        if not is_training:
            B = B // 2
        text_mask = text_mask.unsqueeze(1).unsqueeze(2)
        # (B, 1, 1, T)
        attn_weight[:B, :, :, :] *= text_mask

    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value, attn_weight