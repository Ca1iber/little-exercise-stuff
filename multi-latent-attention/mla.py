import torch
import torch.nn as nn

from math import sqrt


'''
    multi-latent attention with low-rank compression
    input params: 
        x: [B, T, D]
    output params:
        y: [B, T, D]
'''


class MultiLatentAttention(nn.Module):

    def __init__(self, D, low_rank):
        super().__init__()

        self.D = D
        self.num_head = 8
        self.head_dim = self.D // self.num_head
        if self.D % self.num_head != 0:
            raise ValueError(f"please check the value of D")
        
        self.low_rank = low_rank

        # get q k v
        self.W_qkv = nn.Linear(self.D, self.D + self.low_rank)

        # low-rank compression
        self.rms_norm = nn.RMSNorm(self.low_rank)
        # low-rank restoration
        self.W_up_k = nn.Linear(self.low_rank, self.D)
        self.W_up_v = nn.Linear(self.low_rank, self.D)

        self.W_o = nn.Linear(self.D, self.D)

    def store_kv_cache(self, c):
        ...

    def load_kv_cache(self, c):
        ...

    def forward(self, x):

        # get qkv -> [B, T, D + low_rank]
        x = self.W_qkv(x)
        # get q, c -> [B, T, D] + [B, T, low_rank]
        # q, c = x.split([self.D, self.low_rank], dim=-1)
        q, c = x[:, :, :self.D], x[:, :, self.D:]

        c = self.rms_norm(c)

        self.store_kv_cache(c)
        self.load_kv_cache(c)

        # high-rank restoration
        k, v = self.W_up_k(c), self.W_up_v(c)


        # [B, T, D] -> [B, T, nh, d] -> [B, nh, T, d]
        q, k, v = map(
            lambda x: x.reshape(x.shape[0], x.shape[1], self.num_head, self.head_dim)
            .transpose(1, 2),
            (q, k, v)
        )

        s = q @ k.transpose(-1, -2) / sqrt(self.head_dim)

        T = q.shape[-2]
        mask = torch.ones(T, T, dtype=torch.bool, device=s.device).triu(diagonal=1)
        s = s.masked_fill(mask, float('-inf'))

        p = torch.softmax(s, dim=-1)
        o = p @ v

        o = o.transpose(1, 2).reshape(o.shape[0], o.shape[2], -1)
        o = self.W_o(o)

        x = o

        return x

