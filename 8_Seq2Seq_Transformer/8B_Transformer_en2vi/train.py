#!/usr/bin/env python3
import sys, os
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from utils import Seq2SeqDataset, translate
from Transformer.Transformer import \
    Transformer, create_mask_source, create_mask_target

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def setup_logger(name):
    # Setup display to terminal stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s : %(name)s : %(levelname)s : %(message)s : (%(filename)s:%(lineno)d)')
    console.setFormatter(formatter)

    # Set up file logger
    logger_name=f'{name}'
    log_filepath=f'{name}.log'

    logging.basicConfig(
        filename=log_filepath,
        filemode='w+',
        format='%(asctime)s : %(name)s : %(levelname)s : %(message)s : (%(filename)s:%(lineno)d)',
        datefmt='%d-%m-%Y %H:%M:%S',
        level=logging.INFO)
    logger = logging.getLogger(logger_name)
    
    logger.addHandler(console)
    return logger


def fit(
        train_dset,
        en_tokenizer, vi_tokenizer,
        Tx, X_lexicon_size,
        Ty, Y_lexicon_size,
        alpha=1e-2, num_iters=20, batch_size=16,
        device='cpu',
        checkpoint_name='8B_Transformer_en2vi', SAMPLING=False):
    """
    """
    # Setup
    ckpts_f_path = f"ckpts/{checkpoint_name}.ckpt"
    model_f_path = f"ckpts/{checkpoint_name}.model.pth"
    logger = setup_logger('train')
    logger.info('='*10 + ' Init ' + '='*10)

    # Lexicon config
    en_pad_token = en_tokenizer.convert_tokens_to_ids('[PAD]')

    # dataloader
    train_loader = DataLoader(
        dataset=train_dset,
        batch_size=batch_size,
        shuffle=True, num_workers=128, pin_memory=True)

    # Model
    transformer = Transformer(
        Tx=Tx, X_lexicon_size=X_lexicon_size,
        Ty=Ty, Y_lexicon_size=Y_lexicon_size,
        embed_dim=512,
        num_layers=12, num_heads=32,
        forward_expansion_dim=2048,
        dropout=0.1, eps=1e-5)
    transformer = transformer.to(device)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Number of parameters: {count_parameters(transformer)}')

    # Criterions
    #    - Exclude padding token when compte loss
    #    - Only allow 1 EOS in the output seq
    vi_pad_token = vi_tokenizer.convert_tokens_to_ids('<pad>')
    vi_eos_token = vi_tokenizer.convert_tokens_to_ids('</s>')
    criterion = nn.NLLLoss(
        reduction='mean',
        ignore_index=vi_pad_token)
    def custom_lossfn(Y_hat, Y, loss_fn):
        '''Penalize > 1 EOS token'''
        # Cross Entropy cost
        Y_hat_reshaped = Y_hat.contiguous().view(-1, Y_lexicon_size)
        Y_target_reshaped = Y.contiguous().view(-1)
        cost = loss_fn(Y_hat_reshaped, Y_target_reshaped)

        # Calc num of eos in batch
        Y_pred = Y_hat.argmax(dim=-1)
        m, _ = Y_pred.size()
        num_eos_tokens = (Y_pred == vi_eos_token).sum().item()

        # Penalize more than 1 eos
        # cost += torch.tensor(np.abs(num_eos_tokens-m) * float('1'),
        #     requires_grad=True)
        return cost

    # Optimizer
    optimizer = torch.optim.Adam(transformer.parameters(),
        lr=alpha, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=10, verbose=True)

    # Pretrain or train from scratch
    if os.path.exists(ckpts_f_path):
        # Load checkpoints
        checkpoint = torch.load(ckpts_f_path)

        start_epoch = int(checkpoint['epoch']) + 1
        best_cost = checkpoint['best_cost']
        transformer.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        
        logger.info(f'Best cost = {best_cost:.3f}')
        logger.info(f'Starting Iter = {start_epoch:3}')
    else:
        best_cost = float('inf')
        start_epoch = 0

        # Xavier Weights Init
        def initialize_weights(m):
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                nn.init.xavier_uniform_(m.weight.data)
        transformer.apply(initialize_weights)


    # Train Iters
    def train_iter():
        iter_costs = []
        transformer.train()
        for b, batch in enumerate(tqdm(train_loader, desc=f'Iteration {i}')):
            # Batch:
            #    X_b = (batch_size, Tx)
            #    Y_b = (batch_size, Ty)
            Xb = batch['X_seq'].to(device)
            Yb = batch['Y_seq'].to(device)
            batch_size = Xb.size(0)

            # t-1 preidct t
            Yb_in = Yb[:, :-1]
            Yb_target = Yb[:, 1:]

            # Create maskes
            X_mask = create_mask_source(
                X_seq=Xb,
                pad_token=en_pad_token,
                device=device)

            Y_mask = create_mask_target(
                Y_seq=Yb_in,
                pad_token=vi_pad_token,
                device=device)
            # Forward
            #    Yb_hat = (batch_size, Ty, Y_lexicon_size)
            transformer.train()
            optimizer.zero_grad()
            Yb_hat, _ = transformer(
                X_seq=Xb, X_mask=X_mask,
                Y_seq=Yb_in, Y_mask=Y_mask,
                device=device)

            # Batch Cost compute
            cost_b = custom_lossfn(Yb_hat, Yb_target, criterion)

            # Track Iter Cost
            iter_costs.append(cost_b.item())

            # Back Propagation
            cost_b.backward()
            nn.utils.clip_grad_norm_(transformer.parameters(), max_norm=1)
                # Clip grad to avoid exploding gradient
            optimizer.step()

            # Sample Trainset
            if SAMPLING is True and b%100==0:
                transformer.eval()
                print('\n======= SAMPLING =========')
                sample_idx = np.random.randint(0, batch_size)

                x_utt = batch['X_sentence'][sample_idx]
                y_utt = batch['Y_sentence'][sample_idx]

                y_hat = Yb_hat.argmax(dim=-1)[sample_idx]
                y_hat_pred = vi_tokenizer.decode(y_hat)
                y_hat_decodes = translate(x_utt,
                    model=transformer,
                    en_tokenizer=en_tokenizer, vi_tokenizer=vi_tokenizer,
                    Tx=Tx, Ty=Ty, beam_width=5, device=device)

                print(f'{x_utt = }')
                print(f'{y_utt = }')
                print(f'{y_hat_pred = }')
                for x, (decode, att) in enumerate(y_hat_decodes):
                    print(f'y_hat_decode[{x}] = {decode}')
                print(f'Batch cost: {cost_b.item():.3f}')
                print('==========================')

        return iter_costs

    # Train - Epoch
    for i in range(start_epoch, num_iters):
        # Train iter
        iter_costs = train_iter()

        # Cost
        cost = sum(iter_costs) / len(iter_costs)
        scheduler.step(cost)
        if i % 1 == 0 or i == num_iters-1:
            logger.info(f"Cost after iteration {i:4}: {cost:.3f}")

        # Save model + checkpoints
        if cost < best_cost:
            best_cost = cost
            logger.info(f'[SAVING CKPT] Best cost update {best_cost:.3f}')

            if not os.path.exists('ckpts/'): os.makedirs('ckpts/')
            # Model
            torch.save(transformer, model_f_path)

            # Checkpoint
            checkpoint = {
                'epoch': i,
                'model': transformer.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_cost': best_cost
            }
            torch.save(checkpoint, ckpts_f_path)

if __name__ == "__main__":
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    # Init Tokenizers
    en_tokenizer = AutoTokenizer.from_pretrained(
        'bert-base-uncased')
    vi_tokenizer = AutoTokenizer.from_pretrained(
        'vinai/phobert-base')

    # Params
    Tx = 55
    Ty = 60

    # Init dset
    train_dset = Seq2SeqDataset(
        en_tokenizer=en_tokenizer, vi_tokenizer=vi_tokenizer,
        Tx=Tx, Ty=Ty,
        en_sentences="../datasets/train/en_150000",
        vi_sentences="../datasets/train/vi_150000",
        mode="train", device=device)

    # Train
    fit(
        train_dset,
        en_tokenizer=en_tokenizer, vi_tokenizer=vi_tokenizer,
        Tx=Tx, X_lexicon_size=30522,
        Ty=Ty-1, Y_lexicon_size=64000,
        alpha=1e-4, num_iters=1000, batch_size=128,
        device=device,
        checkpoint_name='8B_Transformer_en2vi', SAMPLING=True)
