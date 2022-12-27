import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class Embbeding(nn.Module):
    def __init__(self, lexicon_size, embed_dim):
        super(Embbeding, self).__init__()

        # Emb
        self.embedding = nn.Embedding(
            num_embeddings=lexicon_size,
            embedding_dim=embed_dim)

    def forward(self, inp):
        """
        Arguments:
            inp (Long tensor(m, T))         : Input sequence, categorical Long
                m : batch size
                T : sequence length
        Returns:
            emb (tensor(m, T, embed_dim))   : Embbeding sequence
        """
        # (m, T) -> (m, T, emb_dim)
        return self.embedding(inp)


class PositionalEncoder(nn.Module):
    def __init__(self, T, embed_dim):
        super(PositionalEncoder, self).__init__()
        self.emb_dim = embed_dim

        # Init PE matrix
        #    (T, emb_dim)
        pe = torch.zeros(T, self.emb_dim)

        # Assign pe values
        for t in range(T):
            for i in range(0, self.emb_dim, 2):
                pe[t, i] = np.sin(
                    t / (10000 ** ((2 * i) / self.emb_dim)))
                pe[t, i+1] = np.cos(
                    t / (10000 ** ((2 * (i + 1)) / self.emb_dim)))

        # (T, emb_dim) -> (1, T, emb_dim)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, emb):
        """
        Add position encoding info to emb

        Arguments:
            emb (tensor(m, T, embed_dim))       : Embbeding sequence

        Returns:
            emb_pos (tensor(m, T, embed_dim))   : Embbeding + PE sequence
        """
        # Get dim
        T = emb.size(1)
        
        # Retrieve pe
        pe = Variable(self.pe[:,:T], requires_grad=False)
        if emb.is_cuda: pe.cuda()

        # emb_pos = rescaled*emb + position encoded
        emb_pos = np.sqrt(self.emb_dim)*emb + pe
        return emb_pos


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        # Params
        self.emb_dim = embed_dim
        self.h = num_heads

        # Split emb_dim = num_head * head_dim
        self.head_dim = self.emb_dim // self.h

        assert (
            self.emb_dim == self.h * self.head_dim
        ), "Embedding size needs to be divisible by num_heads"

        # Linear key, queries, values
        self.q_linears = nn.Linear(self.emb_dim, self.emb_dim)
        self.k_linears = nn.Linear(self.emb_dim, self.emb_dim)
        self.v_linears = nn.Linear(self.emb_dim, self.emb_dim)

        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(num_heads*self.head_dim, self.emb_dim)

    def __scaled_dot_product_att(self, q, k, v, mask=None, dropout=None):
        """
        Compute scaled dot product attention

        Arguments:
            q (tensor(m, T_query, num_heads, head_dim))        : query
            k (tensor(m, T_key, num_heads, head_dim))          : key
            v (tensor(m, T_val, num_heads, head_dim))          : value
            mask (tensor None or (m, 1, T, T) or (m, 1, 1, T)) : attention mask, 1=keep, 0=suppress att
            dropout (None or nn.Dropout)                       : dropout function
        Returns:
            attention (tensor(m, T_query, num_heads, head_dim)): scaled dot product attention
        """
        # energy = (m, T_query, num_heads, head_dim) x (m, T_key, num_heads, head_dim)
        #    -> (m, num_heads, T_query, T_key)
        energy = torch.einsum("mqhd,mkhd->mhqk", [q, k]) / np.sqrt(self.emb_dim)

        if mask is not None:
            # Masking, suppress energy/attention where mask == 0
            energy = energy.masked_fill(mask == 0, float('-1e20'))

        # softmax = (m, num_heads, T_query, T_key)
        energy_sm = F.softmax(energy, dim=3)
        if dropout is not None:
            energy_sm = dropout(energy_sm)

        # attention = softmax * v
        #   (m, T_query, num_heads, head_dim)
        attention = torch.einsum("mhqk,mvhd->mqhd", [energy_sm, v])
        return attention

    def forward(self, Q, K, V, mask=None):
        """
        Compute Multihead attention

        Arguments:
            Q (tensor(m, T_query, emb_dim))        : Query
            K (tensor(m, T_key, emb_dim))          : Key
            V (tensor(m, T_val, emb_dim))          : Value
            mask (tensor None or (m, 1, T, T) or (m, 1, 1, T)) : attention mask, 1=keep, 0=suppress att
        Returns:
            attention (tensor(m, T_query, emb_dim)): multihead attention
        """

        # Batch size
        m = Q.size(0)

        # Compute key, queries, values linears, then split q,k,v -> multiple heads
        #   (m, T_query, emb_dim) -> (m, T_query, num_heads, head_dim)
        #   (m, T_key, emb_dim) -> (m, T_key, num_heads, head_dim)
        #   (m, T_val, emb_dim) -> (m, T_val, num_heads, head_dim)
        queries = self.q_linears(Q).view(m, -1, self.h, self.head_dim)
        keys = self.k_linears(K).view(m, -1, self.h, self.head_dim)
        values = self.v_linears(V).view(m, -1, self.h, self.head_dim)

        # Compute scaled dot product attention
        #    (m, T_query, num_heads, head_dim)
        attention = self.__scaled_dot_product_att(
            q=queries, k=keys, v=values,
            mask=mask, dropout=self.dropout)

        # concatenate all heads
        #    (m, T_query, emb_dim)
        concat = attention.view(m, -1, self.h * self.head_dim)

        # Compute last linear
        attention = self.linear(concat)
        return attention


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.eps = eps

        # Trainable params
        self.gamma = nn.Parameter(torch.ones(normalized_shape), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)

    def forward(self, x):
        """
        Compute Norm across each example, agg (emb_dim), same as nn.LayerNorm(emb_dim)

        Arguments:
            x (tensor(m, T, emb_dim))  : Input
            affine (bool)              : If True apply transform to input x
        Returns:
            y (tensor(m, T, emb_dim))  : Norm output
        """
        E_x = x.mean(dim=-1, keepdim=True)
        Var_x = torch.var(x, dim=-1, keepdim=True)
        return (x - E_x) / torch.sqrt(Var_x + self.eps) * self.gamma + self.beta


class FeedForward(nn.Module):
    def __init__(self, embed_dim, forward_expansion_dim=2048, dropout=0.1):
        super(FeedForward, self).__init__()

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, forward_expansion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(forward_expansion_dim, embed_dim),
        )
    def forward(self, inp):
        """
        Apply feed forward toward last dimension

        Arguments:
            inp (tensor(m,*,embed_dim))  : Input tensor
        Returns:
            out (tensor(m,*,embed_dim))  : Output tensor
        """
        return self.feed_forward(inp)
