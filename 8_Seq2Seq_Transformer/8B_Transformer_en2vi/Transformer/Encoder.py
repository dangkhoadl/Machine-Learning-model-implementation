import torch
import torch.nn as nn
import numpy as np

from Transformer.Components import \
    Embbeding, PositionalEncoder, MultiHeadAttention, FeedForward, LayerNorm

class EncoderBlock(nn.Module):
    def __init__(self,
            embed_dim, num_heads,
            dropout=0.1, forward_expansion_dim=2048, eps=1e-5):
        super(EncoderBlock, self).__init__()

        # Layer 1
        self.multihead_attention = MultiHeadAttention(
            embed_dim=embed_dim, num_heads=num_heads,
            dropout=dropout)
        self.norm_1 = LayerNorm(
            normalized_shape=embed_dim,
            eps=eps)
        self.dropout_1 = nn.Dropout(dropout)

        # Layer 2
        self.feed_forward = FeedForward(
            embed_dim=embed_dim, forward_expansion_dim=forward_expansion_dim,
            dropout=dropout)
        self.norm_2 =LayerNorm(
            normalized_shape=embed_dim,
            eps=eps)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, X_emb_pos, X_mask):
        """
        Arguments:
            X_emb_pos (tensor(m, Tx, emb_dim) : Source sequence encoded with embbeding and position
            X_mask (Long tensor(m, 1, 1, T))  : Source sequence attention mask(val=1:keep or val=0:supress)
        Returns:
            X_enc (tensor(m, Tx, emb_dim))    : encoder output from source seq
        """
        # Layer 1
        X_att, _ = self.multihead_attention(
            Q=X_emb_pos, K=X_emb_pos, V=X_emb_pos,
            mask=X_mask)
        X_norm_1 = self.dropout_1(
            self.norm_1(X_emb_pos + X_att))

        # Layer 2
        ff = self.feed_forward(X_norm_1)
        X_enc = self.dropout_2(
            self.norm_2(X_norm_1 + ff))

        return X_enc


class Encoder(nn.Module):
    def __init__(self,
            Tx, X_lexicon_size, embed_dim,
            num_layers, num_heads,
            forward_expansion_dim=1024,
            dropout=0.1, eps=1e-5):
        super(Encoder, self).__init__()
        # Params
        self.embed_dim = embed_dim

        # X = emb + pe
        self.input_embedding = Embbeding(
            lexicon_size=X_lexicon_size, embed_dim=embed_dim)
        self.pos_encoding = PositionalEncoder(
            max_length=100,
            embed_dim=embed_dim)
        self.dropout = nn.Dropout(dropout)

        # Enc blocks
        self.enc_blocks = nn.ModuleList([
            EncoderBlock(
                embed_dim=embed_dim, num_heads=num_heads,
                dropout=dropout, forward_expansion_dim=forward_expansion_dim, eps=eps
            ) for _ in range(num_layers)
        ])

    def forward(self, X_seq, X_mask, device='cpu'):
        """
        Arguments:
            X_seq (Long tensor(m, Tx)         : Source sequence, categorical Long
            X_mask (Long tensor(m, 1, 1, T))  : Source sequence attention mask(val=1:keep or val=0:supress)
        Returns:
            X_enc (tensor(m, Tx, emb_dim))    : encoder output from source seq
        """
        # Input emb, pos encoding
        m, Tx = X_seq.size()
        X_emb = self.input_embedding(X_seq)
        X_pos = self.pos_encoding(m=m, T=Tx, device=device)

        # X_emb_pos = rescaled*emb + position encoded
        X_enc = self.dropout(np.sqrt(self.embed_dim)*X_emb + X_pos)

        # Enc blocks
        for enc_blk in self.enc_blocks:
            X_enc = enc_blk(
                X_emb_pos=X_enc, X_mask=X_mask)

        return X_enc
