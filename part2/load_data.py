import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt', quiet=True)
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0
TOKENIZER_NAME = 'google-t5/t5-small'
TASK_PREFIX = 'translate English to SQL: '

# The T5 sentencepiece vocabulary has no token for '<' (it becomes <unk>), but the SQL
# uses it in comparisons such as "departure_time < 1200". We replace it with a placeholder
# word before tokenization and map it back after decoding.
SQL_REPLACEMENTS = [('<=', ' LTE_OP '), ('<', ' LT_OP ')]


def preprocess_sql(sql):
    for src, tgt in SQL_REPLACEMENTS:
        sql = sql.replace(src, tgt)
    return ' '.join(sql.split())


def postprocess_sql(sql):
    for src, tgt in SQL_REPLACEMENTS:
        sql = sql.replace(tgt.strip(), src)
    return ' '.join(sql.split())


def decode_sql(tokenizer, ids):
    '''Decode generated ids back into an executable SQL string.'''
    text = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return postprocess_sql(text)


def get_tokenizer():
    return T5TokenizerFast.from_pretrained(TOKENIZER_NAME)


class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        '''
        Skeleton for the class for performing data processing for the T5 model.

        Some tips for implementation:
            * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
              the encoder and decoder output.
            * You want to provide the decoder some beginning of sentence token. Any extra-id on the
              T5Tokenizer should serve that purpose.
            * Class behavior should be different on the test set.
        '''
        self.split = split
        self.tokenizer = get_tokenizer()
        # T5's decoder start token is the pad token (id 0); this is also what model.generate() uses
        self.bos_id = self.tokenizer.pad_token_id
        self.encoder_ids, self.decoder_targets = self.process_data(data_folder, split, self.tokenizer)

    def process_data(self, data_folder, split, tokenizer):
        nl_lines = load_lines(os.path.join(data_folder, f'{split}.nl'))
        encoder_texts = [TASK_PREFIX + line for line in nl_lines]
        # The tokenizer appends </s> to every sequence
        encoder_ids = tokenizer(encoder_texts, truncation=True, max_length=512)['input_ids']

        decoder_targets = None
        if split != 'test':
            sql_lines = load_lines(os.path.join(data_folder, f'{split}.sql'))
            assert len(sql_lines) == len(nl_lines)
            sql_lines = [preprocess_sql(s) for s in sql_lines]
            decoder_targets = tokenizer(sql_lines, truncation=True, max_length=512)['input_ids']

        return encoder_ids, decoder_targets

    def __len__(self):
        return len(self.encoder_ids)

    def __getitem__(self, idx):
        enc = torch.tensor(self.encoder_ids[idx], dtype=torch.long)
        if self.decoder_targets is None:
            return enc, self.bos_id
        tgt = torch.tensor(self.decoder_targets[idx], dtype=torch.long)
        # Teacher forcing: decoder input is the target shifted right with the BOS token in front
        dec_in = torch.cat([torch.tensor([self.bos_id], dtype=torch.long), tgt[:-1]])
        return enc, dec_in, tgt, self.bos_id


def _pad_encoder(encs):
    encoder_ids = pad_sequence(encs, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = (encoder_ids != PAD_IDX).long()
    return encoder_ids, encoder_mask


def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encs, dec_ins, tgts, bos = zip(*batch)
    encoder_ids, encoder_mask = _pad_encoder(encs)
    decoder_inputs = pad_sequence(dec_ins, batch_first=True, padding_value=PAD_IDX)
    decoder_targets = pad_sequence(tgts, batch_first=True, padding_value=PAD_IDX)
    initial_decoder_inputs = torch.tensor(bos, dtype=torch.long).unsqueeze(1)
    return encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs


def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns:
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encs, bos = zip(*batch)
    encoder_ids, encoder_mask = _pad_encoder(encs)
    initial_decoder_inputs = torch.tensor(bos, dtype=torch.long).unsqueeze(1)
    return encoder_ids, encoder_mask, initial_decoder_inputs


def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")

    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    # Not used in this assignment (only needed for an LLM-prompting variant)
    train_x = load_lines(os.path.join(data_folder, 'train.nl'))
    train_y = load_lines(os.path.join(data_folder, 'train.sql'))
    dev_x = load_lines(os.path.join(data_folder, 'dev.nl'))
    dev_y = load_lines(os.path.join(data_folder, 'dev.sql'))
    test_x = load_lines(os.path.join(data_folder, 'test.nl'))
    return train_x, train_y, dev_x, dev_y, test_x
