import torch
from copy import deepcopy
from Transformer.Transformer import \
    create_mask_source, create_mask_target
from torch.utils.data import Dataset

class Seq2SeqDataset(Dataset):
    def __init__(self,
            en_tokenizer, vi_tokenizer,
            Tx, Ty,
            en_sentences="../datasets/train/en_175000",
            vi_sentences="../datasets/train/vi_175000",
            mode="train", device='cpu'):
        super(Seq2SeqDataset, self).__init__()
        # Mode
        assert mode in ["train", "validation", "test"]
        self.mode = mode
        self.device = device

        # Read data
        ## Read SRC
        with open(en_sentences, 'r+', encoding='utf-8') as file_obj:
            data_en = file_obj.readlines()
        self.data_en = [ line.strip().lower() for line in data_en ]

        ## Read TAR
        with open(vi_sentences, 'r+', encoding='utf-8') as file_obj:
            data_vi = file_obj.readlines()
        self.data_vi = [ line.strip().lower() for line in data_vi ]

        # Get length
        self.length = len(self.data_en)
        
        # Init tokenizers
        self.en_tokenizer = en_tokenizer
        self.vi_tokenizer = vi_tokenizer

        # Params
        self.Tx = Tx 
        self.Ty = Ty 

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # X
        X_encoded = self.en_tokenizer(
            text=self.data_en[idx],         # the batch sentences to be encoded
            add_special_tokens=True,        # Add [CLS] and [SEP]
            padding='max_length',              # Add [PAD]s
            return_attention_mask=True,     # Generate the attention mask
            return_tensors='pt',            # ask the function to return PyTorch tensors
            max_length=self.Tx,                 # maximum length of a sentence
            truncation=True
        )
        ## (Tx)
        X_seq = X_encoded['input_ids'].squeeze(dim=0)

        # Y
        Y_encoded = self.vi_tokenizer(
            text=self.data_vi[idx],         # the batch sentences to be encoded
            add_special_tokens=True,        # Add [CLS] and [SEP]
            padding='max_length',           # Add [PAD]s
            return_attention_mask=True,     # Generate the attention mask
            return_tensors='pt',            # ask the function to return PyTorch tensors
            max_length=self.Ty,                  # maximum length of a sentence
            truncation=True
        )
        ## (Ty)
        Y_seq = Y_encoded['input_ids'].squeeze(dim=0)
        ## Replace </s> with <pad>
        # vi_pad_token = vi_tokenizer.convert_tokens_to_ids('<pad>')
        # vi_eos_token = vi_tokenizer.convert_tokens_to_ids('</s>')
        # Y_seq = torch.masked_fill(Y_seq,
        #     mask=(Y_seq==vi_eos_token), value=vi_pad_token)

        return {
            'X_sentence': self.data_en[idx],
            'X_seq': torch.LongTensor(X_seq),
            'Y_sentence': self.data_vi[idx],
            'Y_seq': torch.LongTensor(Y_seq),
        }


def translate(input_sentence,
        model, en_tokenizer, vi_tokenizer,
        Tx=32, Ty=12, beam_width=5, device='cpu'):
    """
    Arguments:
        model (torch model)                 : trained transformer model
        input_sentence (str)                : Input human readable format
    Returns:
        output_sentence (str)               : Predicted machine readable format from model
    """
    en_pad_id = en_tokenizer.convert_tokens_to_ids('[PAD]')
    vi_pad_id = vi_tokenizer.convert_tokens_to_ids('<pad>')
    vi_sos_id = vi_tokenizer.convert_tokens_to_ids('<s>')
    vi_eos_id = vi_tokenizer.convert_tokens_to_ids('</s>')
    
    # str -> [37,2,1,56,38] -> tensor(1, Tx)
    X_seq = en_tokenizer(
        text=[input_sentence],                      # the batch sentences to be encoded
        add_special_tokens=True,        # Add [CLS] and [SEP]
        padding='max_length',           # Add [PAD]s
        return_attention_mask=True,     # Generate the attention mask
        return_tensors='pt',            # ask the function to return PyTorch tensors
        max_length=Tx,                  # maximum length of a sentence
        truncation=True)['input_ids'].to(device)
    X_mask = create_mask_source(
        X_seq=X_seq,
        pad_token=en_pad_id,
        device=device)

    # Init beams: log_prob, Y_seq
    Y_seq = torch.full((1, Ty),
        fill_value=vi_pad_id, dtype=torch.int64).to(device)
    Y_seq[:, 0] = vi_sos_id
    beams = [(0, Y_seq)]


    # Beam Search
    final_results = []
    for t in range(1, Ty):
        new_beams = []
        for log_prob, Y_seq in beams:

            # Infer
            Y_in = Y_seq[:,:t]
            Y_mask = create_mask_target(
                Y_seq=Y_in,
                pad_token=vi_pad_id,
                device=device)
            with torch.no_grad():
                # Y_hat = (1, T, Y_lexicon_size)
                Y_hat, attention = model(
                    X_seq=X_seq, X_mask=X_mask,
                    Y_seq=Y_in, Y_mask=Y_mask,
                    device=device)

            # Update beams
            top_log_probs, top_indices = Y_hat[0,t-1,:].topk(beam_width)
            for b in range(beam_width):
                ids = top_indices[b].item()
                log_prob_b = top_log_probs[b].item()
                
                Y_seq_b = deepcopy(Y_seq)
                Y_seq_b[:, t] = ids
                if ids == vi_eos_id or t == Ty-1:
                    final_results.append((
                        1/t*(log_prob*(t-1) + log_prob_b),
                        Y_seq_b,
                        attention))
                else:
                    new_beams.append((
                        1/t*(log_prob*(t-1) + log_prob_b),
                        Y_seq_b))
        # Relax beams
        beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:beam_width]

    # Retrieve best guess Y_seq
    outputs = []
    sent_set = set()
    
    if len(final_results) > 0:
        final_results = sorted(final_results, key=lambda x: x[0], reverse=True)
        for res in final_results[:2*beam_width]:
            ids = res[1].squeeze(dim=0).cpu().detach().tolist()
            ids = list(filter(lambda x : x!=vi_pad_id, ids))
            sent = vi_tokenizer.decode(ids)

            if sent not in sent_set:
                outputs.append( (sent, res[2]) )
                sent_set.add(sent)

    return list(outputs)[:beam_width]
