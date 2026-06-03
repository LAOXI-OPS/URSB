import math
import numpy as np
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from step_sample import create_named_schedule_sampler


def compute_gaussian_product_coef(sigma1, sigma2):
    denom = sigma1 ** 2 + sigma2 ** 2
    coef1 = sigma2 ** 2 / denom
    coef2 = sigma1 ** 2 / denom
    var = (sigma1 ** 2 * sigma2 ** 2) / denom
    return coef1, coef2, var


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


class CrossAttention(nn.Module):
    def __init__(self, hidden_size, heads, dropout):
        super().__init__()
        assert hidden_size % heads == 0
        self.d_head = hidden_size // heads
        self.heads = heads
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key_value, kv_mask):
        B = query.shape[0]
        q = self.q_proj(query).view(B, -1, self.heads, self.d_head).transpose(1, 2)
        k = self.k_proj(key_value).view(B, -1, self.heads, self.d_head).transpose(1, 2)
        v = self.v_proj(key_value).view(B, -1, self.heads, self.d_head).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)

        if kv_mask is not None:
            mask = kv_mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(attn, dim=-1)
        attn_weights = self.dropout(attn_weights)
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, -1, self.heads * self.d_head)
        out = self.out_proj(out)
        return out


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
        self.seq_att = TransformerRep(args)
        self.seq_norm = LayerNorm(self.hidden_size)

        self.cross_attn = CrossAttention(self.hidden_size, heads=4, dropout=args.dropout)
        self.cross_norm1 = LayerNorm(self.hidden_size)
        self.cross_norm2 = LayerNorm(self.hidden_size)
        self.cross_ffn = PositionwiseFeedForward(self.hidden_size, args.dropout)

        self.lambda_uncertainty = args.lambda_uncertainty
        self.dropout = nn.Dropout(args.dropout)
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

    def forward(self, rep_item, x_t, t, mask_seq, side_seq=None):
        # 1. Time embedding on x_t (noise state)
        emb_t = self.time_embed(self.timestep_embedding(t, self.hidden_size))
        x_t_emb = x_t + emb_t  # [B, d]

        # 2. Uncertainty noise on x_t only (not mixed into sequence)
        x_t_emb = x_t_emb + self.lambda_uncertainty * th.randn_like(x_t_emb)

        # 3. Sequence through self-attention (condition encoding)
        seq_features = self.seq_att(rep_item, mask_seq)  # [B, L, d]
        seq_features = self.seq_norm(seq_features)

        # 4. Cross-attention: x_t (query) attends to seq_features (key/value)
        x_t_query = x_t_emb.unsqueeze(1)  # [B, 1, d]
        cross_out = self.cross_attn(self.cross_norm1(x_t_query), seq_features, mask_seq)
        x_t_out = x_t_emb.unsqueeze(1) + self.dropout(cross_out)  # [B, 1, d]

        # 5. FFN
        x_t_out = x_t_out + self.dropout(self.cross_ffn(self.cross_norm2(x_t_out)))

        out = x_t_out.squeeze(1)  # [B, d]
        pred_x1 = self.x1_head(out)
        return out, seq_features, pred_x1


class SBRec(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.rescale_timesteps = args.rescale_timesteps

        interval = int(getattr(args, "interval", max(int(getattr(args, "diffusion_steps", 32)), 2)))
        self.num_timesteps = interval

        beta_min = float(getattr(args, "beta_min", 0.01))
        beta_max = float(getattr(args, "beta_max", 50))

        # Linear noise schedule: g²(t) = beta0 + t*(beta1 - beta0),  f(t) = 0
        ts = np.arange(interval, dtype=np.float64) / max(interval - 1, 1)
        sigma_sq = beta_min * ts + (beta_max - beta_min) * ts ** 2 / 2.0
        sigma_bar_sq = sigma_sq[-1] - sigma_sq

        std_fwd = np.sqrt(sigma_sq)
        std_bwd = np.sqrt(sigma_bar_sq)
        sigma_1_sq = float(sigma_sq[-1])

        std_fwd_t = torch.tensor(std_fwd, dtype=torch.float32)
        std_bwd_t = torch.tensor(std_bwd, dtype=torch.float32)

        # f(t)=0 => alpha_t = alpha_bar_t = 1 for all t
        alpha_t = torch.ones(interval, dtype=torch.float32)
        alpha_bar_t = torch.ones(interval, dtype=torch.float32)

        mu_x0, mu_x1, _ = compute_gaussian_product_coef(std_fwd_t, std_bwd_t)

        self.register_buffer("std_fwd", std_fwd_t)
        self.register_buffer("std_bwd", std_bwd_t)
        self.register_buffer("alpha", alpha_t)
        self.register_buffer("alpha_bar", alpha_bar_t)
        self.register_buffer("mu_x0", mu_x0)
        self.register_buffer("mu_x1", mu_x1)
        self.register_buffer("sigma_1_sq", torch.tensor(sigma_1_sq, dtype=torch.float32))

        self.sample_steps = max(2, min(int(getattr(args, "sample_steps", 32)), self.num_timesteps))
        schedule_sampler_name = getattr(args, "schedule_sampler_name", "uniform")
        self.schedule_sampler = create_named_schedule_sampler(schedule_sampler_name, self.num_timesteps)

        self.xstart_model = SBXstart(self.hidden_size, args)
        self.x1_pool_query = nn.Parameter(torch.randn(self.hidden_size))
        self.use_sde = getattr(args, 'use_sde', False)
        self.reverse_noise_scale = float(getattr(args, 'reverse_noise_scale', 0.0))
        self.sde_reverse_noise = float(getattr(args, 'sde_reverse_noise', 0.0))

    def _expand(self, v, x):
        while v.dim() < x.dim():
            v = v.unsqueeze(-1)
        return v

    def _pool_sequence(self, seq, mask):
        attn_scores = th.matmul(seq, self.x1_pool_query)  # [B, L]
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(-1)  # [B, L, 1]
        return (seq * attn_weights).sum(dim=1)  # [B, d]

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def q_sample(self, step, x0, x1):
        assert x0.shape == x1.shape
        mu0 = self._expand(self.alpha[step] * self.mu_x0[step], x0)
        mu1 = self._expand(self.alpha_bar[step] * self.mu_x1[step], x0)
        xt = mu0 * x0 + mu1 * x1
        if self.use_sde:
            sigma_t = self._expand(self.std_fwd[step], x0)
            sigma_bar_t = self._expand(self.std_bwd[step], x0)
            noise_std = th.sqrt(sigma_t ** 2 * sigma_bar_t ** 2 / self.sigma_1_sq)
            xt = xt + noise_std * th.randn_like(xt)
        return xt.detach()

    def ode_step_reverse(self, t, s, x_s, pred_x0, x1):
        sigma_bar_s = self.std_bwd[s]

        # At endpoint (σ̄_s ≈ 0): shared-noise formula degenerates; use direct interpolation
        if sigma_bar_s < 1e-8:
            return self.q_sample(t, pred_x0, x1)

        sigma_t = self.std_fwd[t]
        sigma_s = self.std_fwd[s]
        sigma_bar_t = self.std_bwd[t]
        a_t = self.alpha[t]
        a_s = self.alpha[s]
        a_1 = self.alpha[-1]

        inv_sigma_1_sq = 1.0 / self.sigma_1_sq

        coef_xs = a_t * sigma_t * sigma_bar_t / (a_s * sigma_s * sigma_bar_s)
        coef_f = a_t * (sigma_bar_t ** 2 - sigma_bar_s * sigma_t * sigma_bar_t / sigma_s) * inv_sigma_1_sq
        coef_x1 = a_t * (sigma_t ** 2 - sigma_s * sigma_t * sigma_bar_t / sigma_bar_s) * inv_sigma_1_sq / a_1

        coef_xs = self._expand(coef_xs, x_s)
        coef_f = self._expand(coef_f, x_s)
        coef_x1 = self._expand(coef_x1, x_s)

        x_t = coef_xs * x_s + coef_f * pred_x0 + coef_x1 * x1

        if self.sde_reverse_noise > 0:
            noise_std = sigma_t * sigma_bar_t / th.sqrt(self.sigma_1_sq)
            noise_std = self._expand(noise_std, x_t)
            x_t = x_t + self.sde_reverse_noise * noise_std * th.randn_like(x_t)

        return x_t

    def reverse_p_sample(self, item_rep, item_rep1, noise_x_t, mask_seq):
        if item_rep.dim() == 3:
            x1 = self._pool_sequence(item_rep, mask_seq)
        elif item_rep.dim() == 2:
            x1 = item_rep
        else:
            raise ValueError("Unsupported tensor shape for SB endpoint x1")

        steps = space_indices(self.num_timesteps, self.sample_steps)
        xt = x1.detach() + self.reverse_noise_scale * noise_x_t
        if self.reverse_noise_scale > 0:
            self._stats('reverse_init_noisy: scale/ x1 / xt',
                        th.tensor([self.reverse_noise_scale]), x1, xt)
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

    def forward(self, item_rep, item_tag, mask_seq, side_seq=None):
        if item_rep.dim() == 3:
            x1 = self._pool_sequence(item_rep, mask_seq)
        elif item_rep.dim() == 2:
            x1 = item_rep
        else:
            raise ValueError("Unsupported tensor shape for SB endpoint x1")

        x0 = item_tag
        t, _ = self.schedule_sampler.sample(item_rep.shape[0], item_tag.device)
        x_t = self.q_sample(t, x0.detach(), x1.detach())
        self._stats('q_sample: x0(item_tag) / x1(pooled) / x_t', x0, x1, x_t)
        pred_x0, seq_feat, pred_x1 = self.xstart_model(item_rep, x_t, self._scale_timesteps(t), mask_seq, side_seq=side_seq)
        self._stats('xstart_out: pred_x0 / pred_x1 / x1', pred_x0, pred_x1, x1)
        return pred_x0, x1, pred_x1
