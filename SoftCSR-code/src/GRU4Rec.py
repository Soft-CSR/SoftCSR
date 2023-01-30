# -*- coding: utf-8 -*-
import  math
import os
import pickle
from tqdm import tqdm
import random
import copy
from torch.nn.init import xavier_uniform_, xavier_normal_
import torch
import torch.nn as nn
import gensim

from modules import Encoder, LayerNorm

class GRU4RecModel(nn.Module):
    def __init__(self, args):
        super(GRU4RecModel, self).__init__()

        # load parameters info
        self.embedding_size = args.embedding_size
        self.hidden_size = args.hidden_size
        self.loss_type = args.loss_type
        self.num_layers = args.num_hidden_layers
        self.dropout_prob = args.hidden_dropout_prob

        # define layers and loss
        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        #self.item_embedding = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)
        self.emb_dropout = nn.Dropout(self.dropout_prob)
        self.gru_layers = nn.GRU(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=False,
            batch_first=True,
        )
        self.dense = nn.Linear(self.hidden_size, self.embedding_size)
        self.loss_fct = nn.BCELoss(reduction='none')

        # parameters initialization
        self.apply(self._init_weights_gru4rec)

    def _init_weights_gru4rec(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight)
        elif isinstance(module,nn.GRU):
            xavier_uniform_(self.gru_layers.weight_hh_l0)
            xavier_uniform_(self.gru_layers.weight_ih_l0)

    def forward(self, input_ids,VAT_embed):
        #item_seq_len = self.item_seq_len(input_ids)
        if len(VAT_embed) == 1:
            item_seq_emb = self.item_embeddings(input_ids)
        else:
            item_seq_emb =VAT_embed
        #item_seq_emb = self.item_embeddings(input_ids)
        item_seq_emb_dropout = self.emb_dropout(item_seq_emb)
        gru_output, _ = self.gru_layers(item_seq_emb_dropout)
        gru_output = self.dense(gru_output)
        # the embedding of the predicted item, shape of (batch_size, embedding_size)
        #seq_output = self.gather_indexes(gru_output, item_seq_len - 1)
        seq_output=gru_output
        return seq_output


    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embeddings(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embeddings.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B, n_items]
        return scores








if __name__ == '__main__':
    onlineitemsim = OnlineItemSimilarity(item_size=10)
    item_embeddings = nn.Embedding(10, 6, padding_idx=0)
    onlineitemsim.update_embedding_matrix(item_embeddings)
    item_idx = torch.tensor(2, dtype=torch.long)
    similiar_items = onlineitemsim.most_similar(item_idx=item_idx, top_k=1)
    print(similiar_items)