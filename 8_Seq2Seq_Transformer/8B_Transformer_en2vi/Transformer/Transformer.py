import torch
import torch.nn as nn

from Transformer.Decoder import Decoder
from Transformer.Encoder import Encoder

def create_mask_source(X_seq, pad_token, device='cpu'):
    """
    Create mask values (0 = masked, 1 = keep)

    Arguments:
        X_seq (Long tensor(m, T)) : Source Sequence, categorical Long
            m : batch size
            T : sequence length
        pad_token (int)           : token value of <pad>, get from X_lexicon['<pad>']

    Returns:
        mask (Long tensor(m, 1, 1, T)) : attention mask(val=1:keep or val=0:supress) for source sequence
    """
    # Get dim
    m, Tx = X_seq.size()

    # padding token Mask
    #    mask = 1 if seq != padding token else 0
    #    (m, T) -> (m, 1, 1, T)
    padding_mask = (X_seq != pad_token) \
        .unsqueeze(dim=1).unsqueeze(dim=2) \
        .to(device)

    return padding_mask \
        .to(torch.int8).requires_grad_(False)

def create_mask_target(Y_seq, pad_token, device='cpu'):
    """
    Create mask values (0 = masked, 1 = keep)

    Arguments:
        Y_seq (Long tensor(m, T)) : Target Sequence, categorical Long
            m : batch size
            T : sequence length
        pad_token (int)           : token value of <pad>, get from Y_lexicon['<pad>']

    Returns:
        mask (Long tensor(m, 1, T, T)) : attention mask(val=1:keep or val=0:supress) for target sequence
    """
    # Get dim
    m, Ty = Y_seq.size()

    # padding token Mask
    #    mask = 1 if seq != padding token else 0
    #    (m, T) -> (m, 1, 1, T) -> (m, 1, T, T)
    padding_mask = (Y_seq != pad_token) \
        .unsqueeze(dim=1).unsqueeze(dim=2) \
        .expand(m, 1, Ty, Ty).to(torch.bool) \
        .to(device)

    # No peaking forward Mask: (m, 1, T, T)
    #   (T, T) = Lower triangular matrix
    nopeak_mask = torch.tril(torch.ones( (Ty, Ty) )) \
        .unsqueeze(dim=0).unsqueeze(dim=1) \
        .expand(m, 1, Ty, Ty).to(torch.bool) \
        .to(device)

    return (padding_mask & nopeak_mask) \
        .to(torch.int8).requires_grad_(False)


class Transformer(nn.Module):
    def __init__(self,
            Tx, X_lexicon_size,
            Ty, Y_lexicon_size,
            embed_dim=512,
            num_layers=3, num_heads=4,
            forward_expansion_dim=1024,
            dropout=0.1, eps=1e-6):
        super().__init__()
        self.encoder = Encoder(
            Tx=Tx, X_lexicon_size=X_lexicon_size, embed_dim=embed_dim,
            num_layers=num_layers, num_heads=num_heads,
            forward_expansion_dim=forward_expansion_dim,
            dropout=dropout, eps=eps)
        self.decoder = Decoder(
            Ty=Ty, Y_lexicon_size=Y_lexicon_size, embed_dim=embed_dim,
            num_layers=num_layers, num_heads=num_heads,
            forward_expansion_dim=forward_expansion_dim,
            dropout=dropout, eps=eps)

    def forward(self, X_seq, X_mask, Y_seq, Y_mask, device='cpu'):
        """
        Arguments:
            X_seq (Long tensor(m, Tx)         : Source sequence, categorical Long
            X_mask (Long tensor(m, 1, 1, T))  : Source sequence attention mask(val=1:keep or val=0:supress)
            Y_seq (Long tensor(m, Ty)         : Target sequence, categorical Long
            Y_mask (Long tensor(m, 1, T, T))  : Target sequence attention mask(val=1:keep or val=0:supress)
        Returns:
            out (tensor(m, Ty, emb_dim))      : log softmax probability predict category in Y_lexicon
        """
        X_enc = self.encoder(
            X_seq=X_seq, X_mask=X_mask,
            device=device)
        out, attention = self.decoder(
            Y_seq=Y_seq, Y_mask=Y_mask,
            X_enc=X_enc, X_mask=X_mask,
            device=device)
        return out, attention
