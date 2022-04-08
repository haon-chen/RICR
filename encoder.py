import torch.nn as nn
import torch
import torch.nn.functional as F

from layers import TermLevelEncoder
from layers import RNNEncoder

class Encoder(nn.Module):

    def __init__(self, args) -> None:
        super(Encoder, self).__init__()

        # Embedding Layer
        self.d_word_vec = args.d_word_vec

        #Term Level Query-aware Attention
        self.term_level_attn = TermLevelEncoder(args)

        self.concat_qd = nn.Linear(2, 1)
        self.dropout = nn.Dropout(args.dropout)

        self.rnn_encoder = RNNEncoder(
            args.rnn_type,
            args.d_hid_qat,
            False,
            args.layer_num,
            args.d_hid_rnn,
            dropout=0.1,
            use_last=False
        )

        self.inner_attention = nn.Sequential(
            nn.Linear(args.d_hid_rnn, args.d_hid_rnn),
            nn.Tanh(),
            nn.Dropout(p=args.dropout),
            nn.Linear(args.d_hid_rnn, 1)
        )

        self.BatchNorm = nn.BatchNorm1d(args.d_hid_rnn, eps=1e-6)

        need_init = [self.inner_attention, self.concat_qd, self.term_level_attn]
        for layer in need_init:
            for p in layer.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    
    def forward(self, q_c_embed, history_len, q_h_embed, d_hc_embed):
        history_num = q_h_embed.size(1)

        q_h_encoded = self.term_level_attn(q_c_embed, q_h_embed)
        d_hc_encoded = self.term_level_attn(q_c_embed, d_hc_embed)

        history_rep = F.tanh(self.dropout(self.concat_qd(torch.stack([q_h_encoded, d_hc_encoded], dim=-1)).squeeze(-1)))

        h_n, outputs = self.rnn_encoder(history_rep, history_len)

        history_encoded = h_n.squeeze(0)

        return self.BatchNorm(history_encoded), outputs

