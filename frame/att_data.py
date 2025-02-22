import numpy as np
from transformers import BertTokenizer
import torch
import torch.utils.data as data
import tensorflow.keras.preprocessing.sequence as pad
from configs.configs import *

BERT_MAX_LEN = max_length

def seq_padding(batch, padding=0, maxlen=BERT_MAX_LEN):
    # length_batch = [len(seq) for seq in batch]
    # max_length = max(length_batch)
    max_length = maxlen
    return np.array([
        np.concatenate([seq, [padding] * (max_length - len(seq))]) if len(seq) < max_length else seq for seq in batch
    ])


class AttDataset(data.Dataset):
    def __init__(self, u_data, label):
        super().__init__()
        self.max_len = BERT_MAX_LEN  # 64
        self.device = torch.device("cuda")
        # self.dataset = sequence.pad_sequences(u_data, maxlen=self.max_len)
        self.dataset = u_data
        # self.labels = torch.from_numpy(label.astype(np.float64))
        self.labels = torch.from_numpy(label.astype(np.float64))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item1 = self.dataset[index]
        item2 = self.labels[index]
        return item1, item2


class BertDataset(data.Dataset):
    def __init__(self, u_data, label):
        super().__init__()
        self.max_len = BERT_MAX_LEN  # 1200
        self.bert_tokenizer = BertTokenizer.from_pretrained(pretrain_path)
        self.dataset = u_data
        self.labels = torch.from_numpy(label.astype(np.float64))
        # self.labels = label

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        item2 = self.labels[index]
        ret = self._tokenizer(item, item2)
        return ret,index

    def _tokenize(self, tokens):
        re_tokens = ['[CLS]']
        for token in tokens:
            re_tokens += self.bert_tokenizer.tokenize(token)
        re_tokens.append('[SEP]')
        if not len(tokens) + 2 == len(re_tokens):
            print(tokens)
            print("token 中有空格")
        return re_tokens

    def _tokenizer(self, item1, item2):
        text = item1
        tokens = self._tokenize(text)
        if len(tokens) > BERT_MAX_LEN:
            tokens = tokens[:BERT_MAX_LEN]
        elif len(tokens)< BERT_MAX_LEN:
            for i in range(0,BERT_MAX_LEN-len(tokens)):
                tokens.append('[PAD]')
        text_len = len(tokens)
        # token_id
        token_ids = self.bert_tokenizer.convert_tokens_to_ids(tokens)
        token_ids = torch.tensor(token_ids).numpy()
        if len(token_ids) > text_len:
            token_ids = token_ids[:text_len]
        #  mask 为全1的
        att_mask = torch.ones(len(token_ids)).numpy().tolist()
        att_mask2 = torch.ones(len(token_ids))
        return token_ids, item2, att_mask, att_mask2

    @staticmethod
    def collate_fn(ret_data):
        ret_data = list(zip(*ret_data))
        token_ids, label, att_mask, _ = ret_data
        label = torch.stack(label)
        tokens_batch = torch.from_numpy(seq_padding(token_ids, maxlen=BERT_MAX_LEN)).long()
        att_mask_batch = pad.pad_sequences(list(att_mask), maxlen=BERT_MAX_LEN, padding='post', value=0)
        att_mask_batch = torch.from_numpy(att_mask_batch).long()
        # att_mask_batch = pad_sequence(att_mask, batcpad_sequence(att_mask, batch_first=True, padding_value=0)
        return tokens_batch, label, att_mask_batch

    @staticmethod
    def collate_fn2(ret_data):
        ret_data = list(zip(*ret_data))
        ret_data2 = list(zip(*ret_data[0]))
        ids = list(ret_data[1])
        token_ids, label, _, att_mask = ret_data2
        # for i in att_mask:
        #
        # att_mask =
        label = torch.stack(label)
        tokens_batch = torch.from_numpy(seq_padding(token_ids, maxlen=BERT_MAX_LEN)).long()
        att_mask_batch = torch.stack(att_mask)
        return tokens_batch, label, att_mask_batch,ids
