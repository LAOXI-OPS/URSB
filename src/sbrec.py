import math
import numpy as np
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from step_sample import create_named_schedule_sampler


def space_indices(num_steps: int, count: int):
    if count <= 0:
        raise ValueError(f"{count=} must be > 0")
    if count == 1:
        return [0]

    xs = np.linspace(0, num_steps - 1, count)
    out = sorted({int(round(x)) for x in xs})

    if out[0] != 0:
        out = [0] + out
    if out[-1] != num_steps - 1:
        out = out + [num_steps - 1]

    dedup = []
    last = -1
    for value in out:
        if value > last:
            dedup.append(value)
            last = value
    return dedup


class SiLU(nn.Module):
    def forward(self, x):
        return x * th.sigmoid(x)


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class SublayerConnection(nn.Module):
    def __init__(self, hidden_size, dropout):
        super().__init__()
        self.norm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class PositionwiseFeedForward(nn.Module):
    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(hidden_size, hidden_size * 4)
        self.w_2 = nn.Linear(hidden_size * 4, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.w_1.weight)
        nn.init.xavier_normal_(self.w_2.weight)

    def forward(self, hidden):
        hidden = self.w_1(hidden)
        activation = 0.5 * hidden * (1 + torch.tanh(
            math.sqrt(2 / math.pi) * (hidden + 0.044715 * torch.pow(hidden, 3))))
        return self.w_2(self.dropout(activation))


class MultiHeadedAttention(nn.Module):
    def __init__(self, heads, hidden_size, dropout):
        super().__init__()
        assert hidden_size % heads == 0
        self.size_head = hidden_size // heads
        self.num_heads = heads
        self.linear_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(3)])
        self.w_layer = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=dropout)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.w_layer.weight)

    def forward(self, q, k, v, mask=None):
        batch_size = q.shape[0]
        q, k, v = [l(x).view(batch_size, -1, self.num_heads, self.size_head).transpose(1, 2)
                    for l, x in zip(self.linear_layers, (q, k, v))]
        corr = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))

        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, corr.shape[1], 1).unsqueeze(-1).repeat(1, 1, 1, corr.shape[-1])
            corr = corr.masked_fill(mask == 0, -1e9)

        prob_attn = F.softmax(corr, dim=-1)
        if self.dropout is not None:
            prob_attn = self.dropout(prob_attn)
        hidden = torch.matmul(prob_attn, v)
        hidden = self.w_layer(hidden.transpose(1, 2).contiguous().view(
            batch_size, -1, self.num_heads * self.size_head))
        return hidden


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, attn_heads, dropout):
        super().__init__()
        self.attention = MultiHeadedAttention(heads=attn_heads, hidden_size=hidden_size, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(hidden_size=hidden_size, dropout=dropout)
        self.input_sublayer = SublayerConnection(hidden_size=hidden_size, dropout=dropout)
        self.output_sublayer = SublayerConnection(hidden_size=hidden_size, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, hidden, mask):
        hidden = self.input_sublayer(hidden, lambda _hidden: self.attention.forward(_hidden, _hidden, _hidden, mask=mask))
        hidden = self.output_sublayer(hidden, self.feed_forward)
        return self.dropout(hidden)


class TransformerRep(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.heads = 4
        self.dropout = args.dropout
        self.n_blocks = args.num_blocks
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(self.hidden_size, self.heads, self.dropout) for _ in range(self.n_blocks)])

    def forward(self, hidden, mask):
        for transformer in self.transformer_blocks:
            hidden = transformer.forward(hidden, mask)
        return hidden


class XAttn(nn.Module):
    """一对多交叉注意力: Q=[B,1,d] × K/V=[B,L,d] → [B,L,d]"""
    def __init__(self, d, heads=4, dropout=0.1):
        super().__init__()
        assert d % heads == 0
        self.d_head = d // heads
        self.heads = heads
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.out_proj = nn.Linear(d, d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, kv, mask):
        B, L, d = kv.shape
        q = self.q_proj(q).view(B, 1, self.heads, self.d_head).transpose(1, 2)  # [B,h,1,dh]
        k = self.k_proj(kv).view(B, L, self.heads, self.d_head).transpose(1, 2)  # [B,h,L,dh]
        v = self.v_proj(kv).view(B, L, self.heads, self.d_head).transpose(1, 2)  # [B,h,L,dh]

        attn = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)  # [B,h,1,L]
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, -1e9)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = attn.transpose(-2, -1) * v  # [B,h,L,1]×[B,h,L,dh] → [B,h,L,dh]
        out = out.transpose(1, 2).contiguous().view(B, L, d)
        return self.out_proj(out)


class X1Encoder(nn.Module):
    def __init__(self, d_model, n_heads=4, max_len=50, dropout=0.1):
        super().__init__()
        self.pos_emb = nn.Embedding(max_len, d_model)

        self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, seq):
        B, L, d = seq.shape

        pos = th.arange(L, device=seq.device)
        seq = seq + self.pos_emb(pos).unsqueeze(0)

        causal_mask = th.triu(th.full((L, L), float('-inf'), device=seq.device), diagonal=1)

        seq_new, _ = self.self_attn(seq, seq, seq, attn_mask=causal_mask)
        seq = self.norm1(seq + seq_new)

        seq = self.norm2(seq + self.ffn(seq))
        return seq[:, -1, :]  # [B, d]


class SBXstart(nn.Module):
    def __init__(self, hidden_size, args):
        super().__init__()
        self.hidden_size = hidden_size
        time_embed_dim = self.hidden_size * 4
        self.time_embed = nn.Sequential(
            nn.Linear(self.hidden_size, time_embed_dim),
            SiLU(),
            nn.Linear(time_embed_dim, self.hidden_size),
        )

        self.x_attn = XAttn(self.hidden_size, heads=4, dropout=args.dropout)
        self.post_trm = TransformerRep(args)
        self.post_norm = LayerNorm(self.hidden_size)
        self.x1_head = nn.Linear(self.hidden_size, self.hidden_size)

    def timestep_embedding(self, timesteps, dim, max_period=10000):
        half = dim // 2
        freqs = th.exp(-math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half)
        freqs = freqs.to(device=timesteps.device)
        args_t = timesteps[:, None].float() * freqs[None]
        embedding = th.cat([th.cos(args_t), th.sin(args_t)], dim=-1)
        if dim % 2:
            embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, rep_item, x_t, t, mask_seq):
        # 1. Time embedding
        emb_t = self.time_embed(self.timestep_embedding(t, self.hidden_size))
        x_t_emb = x_t + emb_t  # [B, d]

        # 2. 一对多交叉注意力: Q=[B,1,d], K/V=rep_item → [B,L,d]
        ca_rep = self.x_attn(x_t_emb, rep_item, mask_seq)  # [B, L, d]

        # 3. Post-Transformer
        out_seq = self.post_trm(rep_item, mask_seq)  # TEMP: skip x_t
        out_seq = self.post_norm(out_seq)

        # 4. Mean pool → [B, d]
        out = out_seq.mean(dim=1)
        pred_x1 = self.x1_head(out)
        return out, out_seq, pred_x1


class SBRec(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.num_timesteps = args.diffusion_steps
        self.sample_steps = max(2, min(args.sample_steps, self.num_timesteps))
        schedule_sampler_name = getattr(args, "schedule_sampler_name", "uniform")
        self.schedule_sampler = create_named_schedule_sampler(schedule_sampler_name, self.num_timesteps)
        self.xstart_model = SBXstart(self.hidden_size, args)
        self.use_sde = getattr(args, 'use_sde', False)

    def _expand(self, v, x):
        while v.dim() < x.dim():
            v = v.unsqueeze(-1)
        return v

    def _scale_timesteps(self, t):
        return t.float() * (1000.0 / self.num_timesteps)

    def q_sample(self, step, x0, x1):
        assert x0.shape == x1.shape
        t = step / (self.num_timesteps - 1)  # normalized [0, 1]
        if not isinstance(t, th.Tensor):
            t = th.tensor(t, device=x0.device, dtype=x0.dtype)
        t = self._expand(t, x0)
        xt = (1 - t) * x0 + t * x1
        if self.use_sde:
            noise_std = th.sqrt(t * (1 - t))
            xt = xt + noise_std * th.randn_like(xt)
        return xt.detach()

    def ode_step_reverse(self, t, s, x_s, pred_x0, x1):
        # Closed-form reverse: x_s = A·x_t + B·x_1 + C·x_0
        # Derived from forward interpolation x_t = (1-t)x_0 + t·x_1 + √(t(1-t))ε
        norm_s = s / (self.num_timesteps - 1)   # current (noisier)
        norm_t = t / (self.num_timesteps - 1)   # target (cleaner)

        if norm_s >= 1.0 - 1e-8:
            return self.q_sample(t, pred_x0, x1)

        A = math.sqrt(norm_t * (1 - norm_t) / (norm_s * (1 - norm_s)))
        B = norm_t - norm_s * A
        C = (1 - norm_t) - (1 - norm_s) * A

        A = self._expand(th.tensor(A, device=x_s.device, dtype=x_s.dtype), x_s)
        B = self._expand(th.tensor(B, device=x_s.device, dtype=x_s.dtype), x_s)
        C = self._expand(th.tensor(C, device=x_s.device, dtype=x_s.dtype), x_s)

        return A * x_s + B * x1 + C * pred_x0

    def reverse_p_sample(self, item_rep, item_rep1, noise_x_t, mask_seq):
        x1 = item_rep.mean(dim=1)

        steps = space_indices(self.num_timesteps, self.sample_steps)
        xt = x1.detach()
        rev = list(steps)[::-1]
        self._stats('reverse_init: x1 / xt', x1, xt)

        for idx, (prev_step, step) in enumerate(zip(rev[1:], rev[:-1])):
            t = th.full((xt.shape[0],), step, device=xt.device, dtype=th.long)
            pred_x0, _, _ = self.xstart_model(item_rep, xt, self._scale_timesteps(t), mask_seq)
            xt = self.ode_step_reverse(prev_step, step, xt, pred_x0, x1)
            if idx == 0 or idx == len(rev) - 2:
                self._stats(f'reverse_step{idx}: pred_x0 / xt_next', pred_x0, xt)
        self._stats('reverse_final: xt / x1 / cos_dist',
                    xt, x1,
                    F.cosine_similarity(xt, x1, dim=-1).mean().unsqueeze(0))
        return xt

    def set_epoch(self, epoch):
        self._epoch = epoch
        self._batch_in_epoch = 0

    def _stats(self, name, *tensors):
        """Monitor tensor values — prints first 3 batches of every 10th epoch."""
        if not hasattr(self, '_epoch'):
            return
        if self._epoch % 10 != 0:
            return
        self._batch_in_epoch += 1
        if self._batch_in_epoch > 3:
            return
        import logging
        logger = logging.getLogger(__name__)
        parts = [f'\n[SBStat epoch={self._epoch} batch={self._batch_in_epoch}] {name}:']
        for t in tensors:
            parts.append(f'  shape={list(t.shape)} mean={t.mean().item():.4f} std={t.std().item():.4f} '
                         f'min={t.min().item():.4f} max={t.max().item():.4f} '
                         f'nan={t.isnan().any().item()}')
        msg = ' '.join(parts)
        logger.info(msg)

    def forward(self, item_rep, item_tag, mask_seq):
        x1 = item_rep.mean(dim=1)
        x0 = item_tag
        t, _ = self.schedule_sampler.sample(item_rep.shape[0], item_tag.device)
        x_t = self.q_sample(t, x0.detach(), x1.detach())
        self._stats('q_sample: x0(item_tag) / x1(pooled) / x_t', x0, x1, x_t)
        pred_x0, seq_feat, pred_x1 = self.xstart_model(item_rep, x_t, self._scale_timesteps(t), mask_seq)
        self._stats('xstart_out: pred_x0 / pred_x1 / x1', pred_x0, pred_x1, x1)
        return pred_x0, x1, pred_x1
