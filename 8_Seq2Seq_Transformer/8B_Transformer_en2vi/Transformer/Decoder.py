import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from Transformer.Components import \
    Embbeding, PositionalEncoder, MultiHeadAttention, FeedForward, LayerNorm


class DecoderBlock(nn.Module):
    def __init__(self,
            embed_dim, num_heads,
            dropout=0.1, forward_expansion_dim=2048, eps=1e-5):
        super(DecoderBlock, self).__init__()
        # Layer 1
        self.masked_multihead_attention = MultiHeadAttention(
            embed_dim=embed_dim, num_heads=num_heads,
            dropout=dropout)
        self.norm_1 = LayerNorm(
            normalized_shape=embed_dim,
            eps=eps)
        self.dropout_1 = nn.Dropout(dropout)

        # Layer 2
        self.multihead_attention = MultiHeadAttention(
            embed_dim=embed_dim, num_heads=num_heads,
            dropout=dropout)
        self.norm_2 = LayerNorm(
            normalized_shape=embed_dim,
            eps=eps)
        self.dropout_2 = nn.Dropout(dropout)

        # Layer 3
        self.feed_forward = FeedForward(
            embed_dim=embed_dim, forward_expansion_dim=forward_expansion_dim,
            dropout=dropout)
        self.norm_3 = LayerNorm(
            normalized_shape=embed_dim,
            eps=eps)
        self.dropout_3 = nn.Dropout(dropout)

    def forward(self,
            Y_emb_pos, Y_mask,
            X_enc, X_mask):
        """
        Arguments:
            Y_emb_pos (tensor(m, Ty, emb_dim) : Target sequence encoded with embbeding and position
            Y_mask (Long tensor(m, 1, T, T))  : Target sequence attention mask(val=1:keep or val=0:supress)
            X_enc (tensor(m, Tx, emb_dim))    : encoder output from source seq
            X_mask (Long tensor(m, 1, 1, T))  : Source sequence attention mask(val=1:keep or val=0:supress)
        Returns:
            Y_dec (tensor(m, Ty, emb_dim))     : deccoder output
        """
        # Layer 1
        Y_masked_att, _ = self.masked_multihead_attention(
            Q=Y_emb_pos, K=Y_emb_pos, V=Y_emb_pos,
            mask=Y_mask)
        Y_norm_1 = self.dropout_1(
            self.norm_1(Y_emb_pos + Y_masked_att))

        # Layer 2
        Y_att, attention = self.multihead_attention(
            Q=Y_norm_1, K=X_enc, V=X_enc,
            mask=X_mask)
        Y_norm_2 = self.dropout_2(
            self.norm_2(Y_norm_1 + Y_att))

        # Layer 3
        ff = self.feed_forward(Y_norm_2)
        Y_dec = self.dropout_3(
            self.norm_3(Y_norm_2 + ff))

        return Y_dec, attention


class Decoder(nn.Module):
    def __init__(self,
            Ty, Y_lexicon_size, embed_dim,
            num_layers, num_heads,
            forward_expansion_dim=1024,
            dropout=0.1, eps=1e-5):
        super(Decoder, self).__init__()
        # Params
        self.embed_dim = embed_dim

        # Y = emb + pe
        self.input_embedding = Embbeding(
            lexicon_size=Y_lexicon_size, embed_dim=embed_dim)
        self.pos_encoding = PositionalEncoder(
            max_length=100,
            embed_dim=embed_dim)

        self.dropout = nn.Dropout(dropout)

        # Dec blocks
        self.dec_blocks = nn.ModuleList([
             DecoderBlock(
                embed_dim=embed_dim, num_heads=num_heads,
                dropout=dropout, forward_expansion_dim=forward_expansion_dim, eps=eps
            ) for _ in range(num_layers)
        ])

        # Classifier
        self.fc_out = nn.Linear(
            in_features=embed_dim,
            out_features=Y_lexicon_size)

    def forward(self,
            Y_seq, Y_mask,
            X_enc, X_mask, device='cpu'):
        """
        Arguments:
            Y_seq (Long tensor(m, Ty)         : Target sequence, categorical Long
            Y_mask (Long tensor(m, 1, T, T))  : Target sequence attention mask(val=1:keep or val=0:supress)
            X_enc (tensor(m, Tx, emb_dim))    : encoder output from source seq
            X_mask (Long tensor(m, 1, 1, T))  : Source sequence attention mask(val=1:keep or val=0:supress)
        Returns:
            out (tensor(m, Ty, emb_dim))      : log softmax probability predict category in Y_lexicon
        """
        # Input emb, pos encoding
        m, Ty = Y_seq.size()
        Y_emb = self.input_embedding(Y_seq)
        Y_pos = self.pos_encoding(m=m, T=Ty, device=device)

        # Y_emb_pos = rescaled*emb + position encoded
        Y_dec = self.dropout(np.sqrt(self.embed_dim)*Y_emb + Y_pos)

        # Dec blocks
        for dec_blk in self.dec_blocks:
            Y_dec, attention = dec_blk(
                Y_emb_pos=Y_dec, Y_mask=Y_mask,
                X_enc=X_enc, X_mask=X_mask)

        # Classifier
        out = self.fc_out(Y_dec)
        out = F.log_softmax(out, dim=-1)

        return out, attention