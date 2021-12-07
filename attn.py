import math

import numpy as np
import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    def __init__(self, args, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, args.hidden_dim).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, args.hidden_dim, 2).float() * -(math.log(10000.0) / args.hidden_dim)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).unsqueeze(-1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.shape[1], :, :]


class SpatAttnLayer(nn.Module):
    def __init__(self, args):
        super(SpatAttnLayer, self).__init__()
        self.args = args
        self.Q = nn.Linear(self.args.hidden_dim, self.args.hidden_dim)
        self.K = nn.Linear(self.args.hidden_dim, self.args.hidden_dim)
        self.V = nn.Linear(self.args.hidden_dim, self.args.hidden_dim)
        self.prob_spat_attn = ProbAttn(args)
        self.dropout = nn.Dropout(self.args.dropout)

    def forward(self, queries, keys, values):
        B, L, N, D = queries.shape
        H = self.args.num_heads
        queries = self.Q(queries).reshape(B, L, N, H, -1)
        keys = self.K(keys).reshape(B, L, N, H, -1)
        values = self.V(values).reshape(B, L, N, H, -1)

        spat_attn_out = self.prob_spat_attn(queries, keys, values).view(B, L, N, D)

        return self.dropout(spat_attn_out)


class ProbAttn(nn.Module):
    def __init__(self, args):
        super(ProbAttn, self).__init__()
        self.args = args

    def forward(self, queries, keys, values):
        B, L_Q, N, H, D = queries.shape  # [B,LQ,N,H,D]
        _, L_K, _, _, _ = keys.shape  # [B,LK,N,H,D]

        queries = queries.transpose(1, -2)  # [B,H,N,LQ,D]
        keys = keys.transpose(1, -2)  # [B,H,N,LQ,D]
        values = values.transpose(1, -2)  # [B,H,N,LQ,D]

        effect_N = self.args.spatial_factor * np.ceil(np.log(N)).astype('int').item()

        effect_N = effect_N if effect_N < N else N

        scores_top, index = self._prob_QK(queries, keys, effect_N, effect_N)

        scale = 1. / math.sqrt(D)
        scores_top = scores_top * scale

        values_mean = values.mean(dim=-3)
        ctx = values_mean.unsqueeze(-3).expand(B, H, N, L_Q, D).clone()
        attn = torch.softmax(scores_top, dim=-1)  # 32,4,168,3,8

        ctx = ctx.permute(0, 1, 3, 2, 4)
        attn_ret = torch.matmul(attn, values.transpose(-2, -3)).type_as(ctx)

        ctx[torch.arange(B)[:, None, None, None], torch.arange(H)[None, :, None, None],
        torch.arange(L_Q)[None, None, :, None], index, :] = attn_ret

        return ctx.permute(0, 2, 3, 1, 4).contiguous()  # [B,L,N,H,D]

    def _prob_QK(self, Q, K, sample_k, n_top):
        B, H, N, L_Q, D = Q.shape  # [B,H,N,L_Q,D]
        _, _, _, L_K, _ = K.shape  # [B,H,N,L_K,D]

        K_expand = K.unsqueeze(-4).expand(B, H, N, N, L_K, D)  # [B,H,N,N,L_K,D]
        index_sample = torch.randint(N, (N, sample_k))  # [N,sample_k]
        K_sample = K_expand[:, :, torch.arange(N).unsqueeze(1), index_sample, :, :]  # [B,H,N,sample_k,L,D]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.permute(0, 1, 2, 4, 5, 3)).squeeze(-2).transpose(-2, -3)
        # [B,H,L,N,sample_k]
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), N)  # [B,H,L,N]

        M_top = M.topk(n_top, sorted=False)[1]  # [B,H,L,sample_k]

        Q_reduce = Q.transpose(-2, -3)[torch.arange(B)[:, None, None, None],
                   torch.arange(H)[None, :, None, None],
                   torch.arange(L_Q)[None, None, :, None], M_top, :]  # factor*ln(L_q)
        # [B,H,L,sample_k,D]
        Q_K = torch.matmul(Q_reduce, K.permute(0, 1, 3, 4, 2))
        return Q_K, M_top  # [B,H,L,sample_k,N]


class TempAttnLayer(nn.Module):
    def __init__(self, args):
        super(TempAttnLayer, self).__init__()
        self.args = args
        self.Q = nn.Linear(self.args.hidden_dim, self.args.hidden_dim)
        self.K = nn.Linear(self.args.hidden_dim, self.args.hidden_dim)
        self.V = nn.Linear(self.args.hidden_dim, self.args.hidden_dim)
        self.prob_temp_attn = ProbAttn(args)
        self.dropout = nn.Dropout(self.args.dropout)

    def forward(self, queries, keys, values):
        B, L, N, D = queries.shape
        H = self.args.num_heads
        queries = self.Q(queries).reshape(B, L, N, H, -1).transpose(1, 2)
        keys = self.K(keys).reshape(B, L, N, H, -1).transpose(1, 2)
        values = self.V(values).reshape(B, L, N, H, -1).transpose(1, 2)

        temp_attn_out = self.prob_temp_attn(queries, keys, values).view(B, L, N, D)
        return self.dropout(temp_attn_out)
