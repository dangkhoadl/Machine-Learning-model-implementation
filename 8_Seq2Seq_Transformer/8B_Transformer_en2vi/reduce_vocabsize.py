#!/usr/bin/env python3
import sys, os

import torch
import numpy as np

from transformers import AutoTokenizer
from tqdm import tqdm
import pickle

'''
Reduce tokenizer vocab_size to dataset lexicon size
'''

def create_lexicon(tokenizer, data_fpath, batch_size=128):
    # Read text
    with open(data_fpath, 'r+', encoding='utf-8') as file_obj:
        lines = file_obj.readlines()
    lines = [ line.strip().lower() for line in lines ]

    # Get all tokens
    ## Special tokens
    token_ids = set([ tokenizer.convert_tokens_to_ids(token)
        for _, token in tokenizer.special_tokens_map.items() ])

    ## Get all tokens
    st = 0
    for en in tqdm(np.arange(start=st+batch_size, stop=len(lines)+batch_size, step=batch_size),
            desc='Tokenizing'):
        encoded = tokenizer(
            text=lines[st:en],               # the batch sentences to be encoded
            add_special_tokens=True,        # Add [CLS] and [SEP]
            padding='longest',              # Add [PAD]s
            return_attention_mask=True,     # Generate the attention mask
            return_tensors='pt',            # ask the function to return PyTorch tensors
            max_length=50,                  # maximum length of a sentence
            truncation=True
        )

        batch_ids = encoded['input_ids']
        for ids in batch_ids:
            token_ids.update( set(ids.tolist()) )

        # Next
        st = en
    
    # Create dictionary
    token_ids = sorted(list(token_ids))
    tkids_2_lexicon = { token_id:idx for idx, token_id in enumerate(token_ids) }
    lexicon_2_tkids = { idx:token_id for idx, token_id in enumerate(token_ids) }
    return tkids_2_lexicon, lexicon_2_tkids

if __name__ == "__main__":
    # X
    en_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    en_tkids_2_lexicon, en_lexicon_2_tkids = create_lexicon(
        tokenizer=en_tokenizer,
        data_fpath='datasets/train/en_1355603', batch_size=128)
    print(f'Reduce en vocab size from {en_tokenizer.vocab_size} to {len(en_tkids_2_lexicon)}')

    # Y
    vi_tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')
    vi_tkids_2_lexicon, vi_lexicon_2_tkids = create_lexicon(
        tokenizer=vi_tokenizer,
        data_fpath='datasets/train/vi_1355603', batch_size=128)
    print(f'Reduce vi vocab size from {vi_tokenizer.vocab_size} to {len(vi_tkids_2_lexicon)}')

    # Save lexicons
    if not os.path.exists('lexicons/'): os.makedirs('lexicons/')
    with open('lexicons/en_lexicon.pkl', 'wb') as file_obj:
        pickle.dump(
            (en_tkids_2_lexicon, en_lexicon_2_tkids),
            file_obj, protocol=pickle.HIGHEST_PROTOCOL)
    with open('lexicons/vi_lexicon.pkl', 'wb') as file_obj:
        pickle.dump(
            (vi_tkids_2_lexicon, vi_lexicon_2_tkids),
            file_obj, protocol=pickle.HIGHEST_PROTOCOL)
