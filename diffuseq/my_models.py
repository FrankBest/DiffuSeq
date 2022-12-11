import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F

from .utils.nn import (
    SiLU,
    linear,
    timestep_embedding,
)


class LSTMNetModel(nn.Module):
    def __init__(self, input_dims, hidden_dims, output_dims, hidden_t_dim, vocab_size, num_layers=3, dropout=0.0, 
                 logits_mode=1):
        super(LSTMNetModel, self).__init__()
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.output_dims = output_dims
        self.hidden_t_dim = hidden_t_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.dropout_prob = dropout
        self.dropout = nn.Dropout(dropout)
        self.logits_mode = logits_mode
        time_embed_dim = hidden_t_dim * 4
        self.time_embed = nn.Sequential(
            linear(hidden_t_dim, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, hidden_dims),
        )
        self.LayerNorm = nn.LayerNorm(hidden_dims, eps=1e-6)
        self.word_embedding = nn.Embedding(vocab_size, self.input_dims)
        self.lm_head = nn.Linear(self.input_dims, vocab_size)
        with torch.no_grad():
            self.lm_head.weight = self.word_embedding.weight
        self.lstm = nn.LSTM(input_size=self.input_dims, hidden_size=self.hidden_dims, num_layers=self.num_layers)

        if self.input_dims != self.hidden_dims:
            self.input_up_proj = nn.Sequential(nn.Linear(input_dims, self.hidden_dims),
                                              nn.Tanh(), nn.Linear(self.hidden_dims, self.hidden_dims))
        if self.output_dims != self.hidden_dims:
            self.output_down_proj = nn.Sequential(nn.Linear(self.hidden_dims, self.hidden_dims),
                                                nn.Tanh(), nn.Linear(self.hidden_dims, self.output_dims))

    def get_embeds(self, input_ids):
        return self.word_embedding(input_ids)

    def get_logits(self, hidden_repr):
        if self.logits_mode == 1:
            return self.lm_head(hidden_repr)
        elif self.logits_mode == 2: # standard cosine similarity
            text_emb = hidden_repr
            emb_norm = (self.lm_head.weight ** 2).sum(-1).view(-1, 1)  # vocab
            text_emb_t = torch.transpose(text_emb.view(-1, text_emb.size(-1)), 0, 1)  # d, bsz*seqlen
            arr_norm = (text_emb ** 2).sum(-1).view(-1, 1)  # bsz*seqlen, 1
            dist = emb_norm + arr_norm.transpose(0, 1) - 2.0 * torch.mm(self.lm_head.weight,
                                                                     text_emb_t)  # (vocab, d) x (d, bsz*seqlen)
            scores = torch.sqrt(torch.clamp(dist, 0.0, np.inf)).view(emb_norm.size(0), hidden_repr.size(0),
                                                               hidden_repr.size(1)) # vocab, bsz*seqlen
            scores = -scores.permute(1, 2, 0).contiguous()
            return scores
        else:
            raise NotImplementedError

    def forward(self, x, timesteps):
        """
        Apply torche model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :return: an [N x C x ...] Tensor of outputs.
        """
        emb_t = self.time_embed(timestep_embedding(timesteps, self.hidden_t_dim))
        
        if self.input_dims != self.hidden_dims:
            emb_x = self.input_up_proj(x)
        else:
            emb_x = x

        emb_inputs = emb_x + emb_t.unsqueeze(1).expand(-1, emb_x.size(1), -1)
        emb_inputs = self.dropout(self.LayerNorm(emb_inputs))
        lstm_out, _ = self.lstm(emb_inputs)

        if self.output_dims != self.hidden_dims:
            h = self.output_down_proj(lstm_out)
        else:
            h = lstm_out
        h = h.type(x.dtype)
        return h


class GRUNetModel(nn.Module):
    def __init__(self, input_dims, hidden_dims, output_dims, hidden_t_dim, vocab_size, num_layers=3, dropout=0.0, 
                 logits_mode=1):
        super(GRUNetModel, self).__init__()
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.output_dims = output_dims
        self.hidden_t_dim = hidden_t_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.dropout_prob = dropout
        self.dropout = nn.Dropout(dropout)
        self.logits_mode = logits_mode
        time_embed_dim = hidden_t_dim * 4
        self.time_embed = nn.Sequential(
            linear(hidden_t_dim, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, hidden_dims),
        )
        self.LayerNorm = nn.LayerNorm(hidden_dims, eps=1e-6)
        self.word_embedding = nn.Embedding(vocab_size, self.input_dims)
        self.lm_head = nn.Linear(self.input_dims, vocab_size)
        with torch.no_grad():
            self.lm_head.weight = self.word_embedding.weight
        self.gru = nn.GRU(input_size=self.input_dims, hidden_size=self.hidden_dims, num_layers=self.num_layers)
        if self.input_dims != self.hidden_dims:
            self.input_up_proj = nn.Sequential(nn.Linear(input_dims, self.hidden_dims),
                                              nn.Tanh(), nn.Linear(self.hidden_dims, self.hidden_dims))
        if self.output_dims != self.hidden_dims:
            self.output_down_proj = nn.Sequential(nn.Linear(self.hidden_dims, self.hidden_dims),
                                                nn.Tanh(), nn.Linear(self.hidden_dims, self.output_dims))

    def get_embeds(self, input_ids):
        return self.word_embedding(input_ids)

    def get_logits(self, hidden_repr):
        if self.logits_mode == 1:
            return self.lm_head(hidden_repr)
        elif self.logits_mode == 2: # standard cosine similarity
            text_emb = hidden_repr
            emb_norm = (self.lm_head.weight ** 2).sum(-1).view(-1, 1)  # vocab
            text_emb_t = torch.transpose(text_emb.view(-1, text_emb.size(-1)), 0, 1)  # d, bsz*seqlen
            arr_norm = (text_emb ** 2).sum(-1).view(-1, 1)  # bsz*seqlen, 1
            dist = emb_norm + arr_norm.transpose(0, 1) - 2.0 * torch.mm(self.lm_head.weight,
                                                                     text_emb_t)  # (vocab, d) x (d, bsz*seqlen)
            scores = torch.sqrt(torch.clamp(dist, 0.0, np.inf)).view(emb_norm.size(0), hidden_repr.size(0),
                                                               hidden_repr.size(1)) # vocab, bsz*seqlen
            scores = -scores.permute(1, 2, 0).contiguous()
            return scores
        else:
            raise NotImplementedError

    def forward(self, x, timesteps):
        """
        Apply torche model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :return: an [N x C x ...] Tensor of outputs.
        """
        emb_t = self.time_embed(timestep_embedding(timesteps, self.hidden_t_dim))
        
        if self.input_dims != self.hidden_dims:
            emb_x = self.input_up_proj(x)
        else:
            emb_x = x

        emb_inputs = emb_x + emb_t.unsqueeze(1).expand(-1, emb_x.size(1), -1)
        emb_inputs = self.dropout(self.LayerNorm(emb_inputs))
        gru_out, _ = self.gru(emb_inputs)

        if self.output_dims != self.hidden_dims:
            h = self.output_down_proj(gru_out)
        else:
            h = gru_out
        h = h.type(x.dtype)
        return h
