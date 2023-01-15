import torch
import torch.nn as nn
import torch.nn.functional as F
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
    def __init__(self, max_length, embed_dim):
        super(PositionalEncoder, self).__init__()
        self.emb_dim = embed_dim

        self.pos_encode = nn.Embedding(
            num_embeddings=max_length,
            embedding_dim=embed_dim)


    def forward(self, m, T, device='cpu'):
        """
        Add position encoding info to emb

        Arguments:
            m (int)     : batch_size
            T (int)     : sequence length
        Returns:
            pos (tensor(m, T, embed_dim))   : Pos Encoding
        """
        pe = torch.arange(0, T).to(device).expand(m, -1) 
        return self.pos_encode(pe)


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
        self.linear = nn.Linear(self.h * self.head_dim, self.emb_dim)

    def __scaled_dot_product_att(self, q, k, v, mask=None, dropout=None):
        """
        Compute scaled dot product attention

        Arguments:
            q (tensor(m, num_heads, T_query, head_dim))        : query
            k (tensor(m, num_heads, T_key, head_dim))          : key
            v (tensor(m, num_heads, T_val, head_dim))          : value
            mask (tensor None or (m, 1, T, T) or (m, 1, 1, T)) : attention mask, 1=keep, 0=suppress att
            dropout (None or nn.Dropout)                       : dropout function
        Returns:
            context_vector (tensor(m, T_query, num_heads, head_dim)): scaled dot product context vector
            attention (tensor(m, num_heads, T_query, T_key))        : scaled dot product attention
        """
        # energy = (m, num_heads, T_query, head_dim) x (m, num_heads, head_dim, T_key)
        #    -> (m, num_heads, T_query, T_key)
        energy = torch.matmul(q, k.permute(0, 1, 3, 2)) / np.sqrt(self.emb_dim)

        if mask is not None:
            # Masking, suppress energy/attention where mask == 0
            energy = energy.masked_fill(mask == 0, float('-1e20'))

        # attention(alpha_ij) = (m, num_heads, T_query, T_key)
        attention = F.softmax(energy, dim=3)
        if dropout is not None:
            attention = dropout(attention)

        # context_vector = attention x v
        #   (m, num_heads, T_query, T_key) x (m, num_heads, T_val, head_dim)
        #       -> (m, num_heads, T_query, head_dim) -> (m, T_query, num_heads, head_dim)
        context_vector = torch.matmul(attention, v) \
            .permute(0, 2, 1, 3).contiguous()
        return context_vector, attention

    def forward(self, Q, K, V, mask=None):
        """
        Compute Multihead attention

        Arguments:
            Q (tensor(m, T_query, emb_dim))        : Query
            K (tensor(m, T_key, emb_dim))          : Key
            V (tensor(m, T_val, emb_dim))          : Value
            mask (tensor None or (m, 1, T, T) or (m, 1, 1, T)) : attention mask, 1=keep, 0=suppress att
        Returns:
            context_vector (tensor(m, T_query, emb_dim))       : multihead context_vector
            attention (tensor(m, num_heads, T_query, T_key))   : scaled dot product attention
        """

        # Batch size
        m = Q.size(0)

        # Compute key, queries, values linears, then split q,k,v -> multiple heads
        #   (m, T_query, emb_dim) -> (m, T_query, num_heads, head_dim) -> (m, num_heads, T_query, head_dim)
        #   (m, T_key,   emb_dim) -> (m, T_key,   num_heads, head_dim) -> (m, num_heads, T_key, head_dim)
        #   (m, T_val,   emb_dim) -> (m, T_val,   num_heads, head_dim) -> (m, num_heads, T_val, head_dim)
        queries = self.q_linears(Q) \
            .view(m, -1, self.h, self.head_dim) \
            .permute(0, 2, 1, 3)
        keys = self.k_linears(K) \
            .view(m, -1, self.h, self.head_dim) \
            .permute(0, 2, 1, 3)
        values = self.v_linears(V) \
            .view(m, -1, self.h, self.head_dim) \
            .permute(0, 2, 1, 3)

        # Compute scaled dot product context_vector
        #    (m, T_query, num_heads, head_dim)
        context_vector, attention = self.__scaled_dot_product_att(
            q=queries, k=keys, v=values,
            mask=mask, dropout=self.dropout)

        # concatenate all heads
        #    (m, T_query, emb_dim)
        context_vector = context_vector.view(m, -1, self.h * self.head_dim)

        # Compute last linear
        context_vector = self.linear(context_vector)
        return context_vector, attention


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
