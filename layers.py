import imp
import numpy as np
import math
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from utils.misc import aeq

def kernel_mus(n_kernels):
    l_mu = [1]
    if n_kernels == 1:
        return l_mu
    bin_size = 2.0 / (n_kernels - 1)
    l_mu.append(1 - bin_size / 2) 
    for i in range(1, n_kernels - 1):
        l_mu.append(l_mu[i] - bin_size)
    return l_mu

def kernel_sigmas(n_kernels):
    bin_size = 2.0 / (n_kernels - 1)
    l_sigma = [0.001]
    if n_kernels == 1:
        return l_sigma
    l_sigma += [0.1] * (n_kernels - 1)
    return l_sigma

class knrm(nn.Module):
    def __init__(self, args):
        super(knrm, self).__init__()
        k = args.n_kernels
        tensor_mu = torch.FloatTensor(kernel_mus(k))
        tensor_sigma = torch.FloatTensor(kernel_sigmas(k))
        if args.use_cuda:
            tensor_mu = tensor_mu.cuda(args.cuda_id)
            tensor_sigma = tensor_sigma.cuda(args.cuda_id)
        self.mu = Variable(tensor_mu, requires_grad = False).view(1, 1, 1, k)
        self.sigma = Variable(tensor_sigma, requires_grad = False).view(1, 1, 1, k)
        self.dense = nn.Linear(k, 1, 1)

    def get_intersect_matrix(self, q_embed, d_embed, attn_q, attn_d):
        sim = torch.bmm(q_embed, torch.transpose(d_embed, 1, 2)).view(q_embed.size()[0], q_embed.size()[1], d_embed.size()[1], 1) # n*m*d*1
        pooling_value = torch.exp((- ((sim - self.mu) ** 2) / (self.sigma ** 2) / 2)) * attn_d
        pooling_sum = torch.sum(pooling_value, 2)
        log_pooling_sum = torch.log(torch.clamp(pooling_sum, min=1e-10)) * 0.01 * attn_q
        log_pooling_sum = torch.sum(log_pooling_sum, 1)#soft-TF
        return log_pooling_sum

    def forward(self, inputs_q, inputs_d, mask_q, mask_d):
        q_embed_norm = F.normalize(inputs_q, 2, 2)
        d_embed_norm = F.normalize(inputs_d, 2, 2)
        mask_d = mask_d.view(mask_d.size()[0], 1, mask_d.size()[1], 1)
        mask_q = mask_q.view(mask_q.size()[0], mask_q.size()[1], 1)
        log_pooling_sum = self.get_intersect_matrix(q_embed_norm, d_embed_norm, mask_q, mask_d)
        output = F.tanh(self.dense(log_pooling_sum)).squeeze(-1)
        return output

class conv_knrm(nn.Module):

    def __init__(self, args):
        """

        :param mu: |d| * 1 dimension mu
        :param sigma: |d| * 1 dimension sigma
        """
        super(conv_knrm, self).__init__()
        
        tensor_mu = torch.FloatTensor(kernel_mus(args.n_kernels))
        tensor_sigma = torch.FloatTensor(kernel_sigmas(args.n_kernels))
        if args.use_cuda:
            tensor_mu = tensor_mu.cuda(args.cuda_id)
            tensor_sigma = tensor_sigma.cuda(args.cuda_id)
        
        
        self.mu = Variable(tensor_mu, requires_grad = False).view(1, 1, 1, args.n_kernels)
        self.sigma = Variable(tensor_sigma, requires_grad = False).view(1, 1, 1, args.n_kernels)


        self.d_word_vec = args.d_word_vec

        self.dense_f = nn.Linear(args.n_kernels * 9, 1, 1)
        self.tanh = nn.Tanh()

        self.conv_uni = nn.Sequential(
            nn.Conv2d(1, 128, (1, args.d_word_vec)),
            nn.ReLU()
        )

        self.conv_bi = nn.Sequential(
            nn.Conv2d(1, 128, (2, args.d_word_vec)),
            nn.ReLU()
        )
        self.conv_tri = nn.Sequential(
            nn.Conv2d(1, 128, (3, args.d_word_vec)),
            nn.ReLU()
        )



    def get_intersect_matrix(self, q_embed, d_embed, atten_q, atten_d):

        sim = torch.bmm(q_embed, d_embed).view(q_embed.size()[0], q_embed.size()[1], d_embed.size()[2], 1)
        pooling_value = torch.exp((- ((sim - self.mu) ** 2) / (self.sigma ** 2) / 2)) * atten_d
        pooling_sum = torch.sum(pooling_value, 2)
        log_pooling_sum = torch.log(torch.clamp(pooling_sum, min=1e-10)) * 0.01 * atten_q
        log_pooling_sum = torch.sum(log_pooling_sum, 1)
        return log_pooling_sum



    def forward(self, inputs_qwt, inputs_dwt, inputs_qwm, inputs_dwm):
        qwu_embed = torch.transpose(torch.squeeze(self.conv_uni(inputs_qwt.view(inputs_qwt.size()[0], 1, -1, self.d_word_vec))), 1, 2) + 0.000000001
        qwb_embed = torch.transpose(torch.squeeze(self.conv_bi (inputs_qwt.view(inputs_qwt.size()[0], 1, -1, self.d_word_vec))), 1, 2) + 0.000000001
        qwt_embed = torch.transpose(torch.squeeze(self.conv_tri(inputs_qwt.view(inputs_qwt.size()[0], 1, -1, self.d_word_vec))), 1, 2) + 0.000000001
        dwu_embed = torch.squeeze(self.conv_uni(inputs_dwt.view(inputs_dwt.size()[0], 1, -1, self.d_word_vec))) + 0.000000001
        dwb_embed = torch.squeeze(self.conv_bi (inputs_dwt.view(inputs_dwt.size()[0], 1, -1, self.d_word_vec))) + 0.000000001
        dwt_embed = torch.squeeze(self.conv_tri(inputs_dwt.view(inputs_dwt.size()[0], 1, -1, self.d_word_vec))) + 0.000000001
        qwu_embed_norm = F.normalize(qwu_embed, p=2, dim=2, eps=1e-10)
        qwb_embed_norm = F.normalize(qwb_embed, p=2, dim=2, eps=1e-10)
        qwt_embed_norm = F.normalize(qwt_embed, p=2, dim=2, eps=1e-10)
        dwu_embed_norm = F.normalize(dwu_embed, p=2, dim=1, eps=1e-10)
        dwb_embed_norm = F.normalize(dwb_embed, p=2, dim=1, eps=1e-10)
        dwt_embed_norm = F.normalize(dwt_embed, p=2, dim=1, eps=1e-10)
        mask_qw = inputs_qwm.view(inputs_qwt.size()[0], inputs_qwt.size()[1], 1)
        mask_dw = inputs_dwm.view(inputs_dwt.size()[0], 1, inputs_dwt.size()[1], 1)
        #print(type(inputs_qwt.size()[1] - (1 - 1)), inputs_qwt.size()[1] - (1 - 1), inputs_qwm.shape)
        mask_qwu = mask_qw[:, :inputs_qwt.size()[1] - (1 - 1), :]
        mask_qwb = mask_qw[:, :inputs_qwt.size()[1] - (2 - 1), :]
        mask_qwt = mask_qw[:, :inputs_qwt.size()[1] - (3 - 1), :]
        #print(type(inputs_dwt.size()[1] - (1 - 1)), inputs_dwt.size()[1] - (1 - 1), inputs_dwm.shape)
        mask_dwu = mask_dw[:, :, :inputs_dwt.size()[1] - (1 - 1), :]
        mask_dwb = mask_dw[:, :, :inputs_dwt.size()[1] - (2 - 1), :]
        mask_dwt = mask_dw[:, :, :inputs_dwt.size()[1] - (3 - 1), :]
        log_pooling_sum_wwuu = self.get_intersect_matrix(qwu_embed_norm, dwu_embed_norm, mask_qwu, mask_dwu)
        log_pooling_sum_wwut = self.get_intersect_matrix(qwu_embed_norm, dwt_embed_norm, mask_qwu, mask_dwt)
        log_pooling_sum_wwub = self.get_intersect_matrix(qwu_embed_norm, dwb_embed_norm, mask_qwu, mask_dwb)
        log_pooling_sum_wwbu = self.get_intersect_matrix(qwb_embed_norm, dwu_embed_norm, mask_qwb, mask_dwu)
        log_pooling_sum_wwtu = self.get_intersect_matrix(qwt_embed_norm, dwu_embed_norm, mask_qwt, mask_dwu)

        log_pooling_sum_wwbb = self.get_intersect_matrix(qwb_embed_norm, dwb_embed_norm, mask_qwb, mask_dwb)
        log_pooling_sum_wwbt = self.get_intersect_matrix(qwb_embed_norm, dwt_embed_norm, mask_qwb, mask_dwt)
        log_pooling_sum_wwtb = self.get_intersect_matrix(qwt_embed_norm, dwb_embed_norm, mask_qwt, mask_dwb)
        log_pooling_sum_wwtt = self.get_intersect_matrix(qwt_embed_norm, dwt_embed_norm, mask_qwt, mask_dwt)
        log_pooling_sum = torch.cat([ log_pooling_sum_wwuu, log_pooling_sum_wwut, log_pooling_sum_wwub, log_pooling_sum_wwbu, log_pooling_sum_wwtu,\
            log_pooling_sum_wwbb, log_pooling_sum_wwbt, log_pooling_sum_wwtb, log_pooling_sum_wwtt], 1)
        output = F.tanh(self.dense_f(log_pooling_sum)).squeeze(-1)
        return output


class RNNEncoder(nn.Module):
    """ A generic recurrent neural network encoder.
    Args:
       rnn_type (:obj:`str`):
          style of recurrent unit to use, one of [RNN, LSTM, GRU, SRU]
       bidirectional (bool) : use a bidirectional RNN
       num_layers (int) : number of stacked layers
       hidden_size (int) : hidden size of each layer
       dropout (float) : dropout value for :obj:`nn.Dropout`
    # src: https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/encoders/rnn_encoder.py
    """

    def __init__(self,
                 rnn_type,
                 input_size,
                 bidirectional,
                 num_layers,
                 hidden_size,
                 dropout=0.0,
                 use_bridge=False,
                 use_last=True):
        super(RNNEncoder, self).__init__()

        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0
        hidden_size = hidden_size // num_directions

        # Saves preferences for layer
        self.nlayers = num_layers
        self.use_last = use_last

        self.rnns = nn.ModuleList()
        for i in range(self.nlayers):
            input_size = input_size if i == 0 else hidden_size * num_directions
            kwargs = {'input_size': input_size,
                      'hidden_size': hidden_size,
                      'num_layers': 1,
                      'bidirectional': bidirectional,
                      'batch_first': True}
            rnn = getattr(nn, rnn_type)(**kwargs)
            self.rnns.append(rnn)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        # Initialize the bridge layer
        self.use_bridge = use_bridge
        if self.use_bridge:
            nl = 1 if self.use_last else num_layers
            self._initialize_bridge(rnn_type, hidden_size, nl)
        
        need_init = [self.rnns]
        for layer in need_init:
            for p in layer.parameters():
                if p.dim() > 1:
                    nn.init.orthogonal_(p)

    def forward(self,
                emb,
                lengths=None,
                init_states=None):
        "See :obj:`EncoderBase.forward()`"
        self._check_args(emb, lengths)

        packed_emb = emb
        if lengths is not None:
            # Lengths data is wrapped inside a Tensor.
            lengths, indices = torch.sort(lengths, 0, True)  # Sort by length (keep idx)
            packed_emb = pack(packed_emb[indices], lengths.tolist(), batch_first=True)
            _, _indices = torch.sort(indices, 0)  # Un-sort by length

        istates = []
        if init_states is not None:
            if isinstance(init_states, tuple):
                hidden_states, cell_states = init_states
                hidden_states = hidden_states.split(self.nlayers, dim=0)
                cell_states = cell_states.split(self.nlayers, dim=0)
            else:
                hidden_states = init_states
                hidden_states = hidden_states.split(self.nlayers, dim=0)

            for i in range(self.nlayers):
                if isinstance(init_states, tuple):
                    istates.append((hidden_states[i], cell_states[i]))
                else:
                    istates.append(hidden_states[i])

        memory_bank, encoder_final = [], {'h_n': [], 'c_n': []}
        for i in range(self.nlayers):
            if i != 0:
                packed_emb = self.layer_norm(self.dropout(packed_emb))
                if lengths is not None:
                    packed_emb = pack(packed_emb, lengths.tolist(), batch_first=True)

            if init_states is not None:
                packed_emb, states = self.rnns[i](packed_emb, istates[i])
            else:
                packed_emb, states = self.rnns[i](packed_emb)

            if isinstance(states, tuple):
                h_n, c_n = states
                encoder_final['c_n'].append(c_n)
            else:
                h_n = states
            encoder_final['h_n'].append(h_n)

            packed_emb = unpack(packed_emb, batch_first=True)[0] if lengths is not None else packed_emb
            if not self.use_last or i == self.nlayers - 1:
                memory_bank += [packed_emb[_indices]] if lengths is not None else [packed_emb]

        assert len(encoder_final['h_n']) != 0
        if self.use_last:
            memory_bank = memory_bank[-1]
            if len(encoder_final['c_n']) == 0:
                encoder_final = encoder_final['h_n'][-1]
            else:
                encoder_final = encoder_final['h_n'][-1], encoder_final['c_n'][-1]
        else:
            memory_bank = torch.cat(memory_bank, dim=2)
            if len(encoder_final['c_n']) == 0:
                encoder_final = torch.cat(encoder_final['h_n'], dim=0)
            else:
                encoder_final = torch.cat(encoder_final['h_n'], dim=0), \
                                torch.cat(encoder_final['c_n'], dim=0)

        if self.use_bridge:
            encoder_final = self._bridge(encoder_final)

        # TODO: Temporary hack is adopted to compatible with DataParallel
        # reference: https://github.com/pytorch/pytorch/issues/1591
        if memory_bank.size(1) < emb.size(1):
            dummy_tensor = torch.zeros(memory_bank.size(0),
                                       emb.size(1) - memory_bank.size(1),
                                       memory_bank.size(2)).type_as(memory_bank)
            memory_bank = torch.cat([memory_bank, dummy_tensor], 1)

        return encoder_final, memory_bank

    def _initialize_bridge(self,
                           rnn_type,
                           hidden_size,
                           num_layers):

        # LSTM has hidden and cell state, other only one
        number_of_states = 2 if rnn_type == "LSTM" else 1
        # Total number of states
        self.total_hidden_dim = hidden_size * num_layers

        # Build a linear layer for each
        self.bridge = nn.ModuleList([nn.Linear(self.total_hidden_dim,
                                               self.total_hidden_dim,
                                               bias=True)
                                     for _ in range(number_of_states)])

    def _bridge(self, hidden):
        """
        Forward hidden state through bridge
        """

        def bottle_hidden(linear, states):
            """
            Transform from 3D to 2D, apply linear and return initial size
            """
            size = states.size()
            result = linear(states.view(-1, self.total_hidden_dim))
            return F.relu(result).view(size)

        if isinstance(hidden, tuple):  # LSTM
            outs = tuple([bottle_hidden(layer, hidden[ix])
                          for ix, layer in enumerate(self.bridge)])
        else:
            outs = bottle_hidden(self.bridge[0], hidden)

        return outs
    
    def _check_args(self,
                    src,
                    lengths=None,
                    hidden=None):
        n_batch, _, _ = src.size()
        if lengths is not None:
            n_batch_, = lengths.size()
            aeq(n_batch, n_batch_)

class TermLevelEncoder(nn.Module):
    '''
    Term level Query-aware attention + InnerAttention
    '''
    def __init__(self, args):
        super(TermLevelEncoder, self).__init__()

        
        self.query_term_aware_attention = BasicAttention(
            args.d_word_vec, args.d_word_vec, args.d_word_vec, 
            output_hidden_size=args.d_hid_qat, 
            is_q=False, is_k=False, is_v=False, num_heads=args.num_heads, 
            drop_rate=args.dropout)

        self.layer_norm = nn.LayerNorm(args.d_hid_qat, eps=1e-6)

        self.inner_attention = nn.Sequential(
            nn.Linear(args.d_hid_qat, args.d_hid_qat),
            nn.Tanh(),
            nn.Dropout(p=args.dropout),
            nn.LayerNorm(args.d_hid_qat, eps=1e-6),
            nn.Linear(args.d_hid_qat, 1),
            nn.Dropout(p=args.dropout)
        )

    def forward(self, q_c, history):
        '''
        q_c: (batch_size, len, embed_size)
        history: (batch_size, history_num, len, embed_size)
        '''

        h_attn = [[self.layer_norm(self.query_term_aware_attention(q_ct, his.squeeze(1), his.squeeze(1)))for q_ct in torch.split(q_c, 1, dim=1)] 
        for his in torch.split(history, 1, dim=1)]

        for index, history_rep in enumerate(h_attn):
            history_rep = torch.cat(history_rep, dim=1)
            att_weights = self.inner_attention(history_rep).squeeze(-1)
            att_weights = F.softmax(att_weights, 1)
            history_rep = torch.bmm(history_rep.transpose(1, 2), att_weights.unsqueeze(-1)).squeeze(-1)
            h_attn[index] = history_rep
        
        return torch.stack(h_attn, dim=1)


def orignal(x):
    return x

class BasicAttention(nn.Module):
    # src: https://github.com/sakuranew/attention-pytorch/blob/master/attention.py
    def __init__(self,
                 q_embd_size,
                 k_embd_size,
                 v_embd_size,
                 q_k_hidden_size=None,
                 output_hidden_size=None,
                 num_heads=1,  # for multi-head attention
                 score_func='scaled_dot',
                 drop_rate=0.,
                 is_q=False,  # let q_embd to be q or not,default not
                 is_k=False,
                 is_v=False,
                 bias=True
                 ):
        '''
        :param q_embd_size:
        :param k_embd_size:
        :param v_embd_size:
        :param q_k_hidden_size:
        :param output_hidden_size:
        :param num_heads: for multi-head attention
        :param score_func:
        :param is_q: let q_embd to be q or not,default not
        :param is_k: let k_embd to be k or not,default not
        :param is_v: let v_embd to be v or not,default not
        :param bias: bias of linear
        '''
        super(BasicAttention, self).__init__()
        if not q_k_hidden_size:
            q_k_hidden_size = q_embd_size
        if not output_hidden_size:
            output_hidden_size = v_embd_size
        assert q_k_hidden_size % num_heads == 0
        self.head_dim = q_k_hidden_size // num_heads
        assert self.head_dim * num_heads == q_k_hidden_size, "q_k_hidden_size must be divisible by num_heads"
        assert output_hidden_size % num_heads == 0, "output_hidden_size must be divisible by num_heads"
        if is_q:
            self.q_w = orignal
            assert q_embd_size == k_embd_size
        else:
            self.q_w = nn.Linear(q_embd_size, q_k_hidden_size,bias=bias)
        self.is_q = is_q
        self.q_embd_size = q_embd_size
        if is_k:
            self.k_w = orignal
            assert k_embd_size == q_k_hidden_size
        else:
            self.k_w = nn.Linear(k_embd_size, q_k_hidden_size,bias=bias)
        if is_v:
            self.v_w = orignal
            assert v_embd_size == output_hidden_size
        else:
            self.v_w = nn.Linear(v_embd_size, output_hidden_size,bias=bias)
        self.q_k_hidden_size = q_k_hidden_size
        self.output_hidden_size = output_hidden_size
        self.num_heads = num_heads
        self.score_func = score_func
        self.drop_rate = drop_rate
        

    def forward(self, q_embd, k_embd, v_embd, mask=None):
        '''
        batch-first is needed
        :param q_embd: [?,q_len,q_embd_size] or [?,q_embd_size]
        :param k_embd: [?,k_len,k_embd_size] or [?,k_embd_size]
        :param v_embd: [?,v_len,v_embd_size] or [?,v_embd_size]
        :return: [?,q_len,output_hidden_size*num_heads]
        '''
        if len(q_embd.shape) == 2:
            q_embd = torch.unsqueeze(q_embd, 1)
        if len(k_embd.shape) == 2:
            k_embd = torch.unsqueeze(k_embd, 1)
        if len(v_embd.shape) == 2:
            v_embd = torch.unsqueeze(v_embd, 1)
        batch_size = q_embd.shape[0]
        q_len = q_embd.shape[1]
        k_len = k_embd.shape[1]
        v_len = v_embd.shape[1]
        #     make sure k_len==v_len
        assert k_len == v_len

        q = self.q_w(q_embd).view(batch_size, q_len, self.num_heads, self.head_dim)
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, q_len, self.head_dim)
        k = self.k_w(k_embd).view(batch_size, k_len, self.num_heads, self.head_dim)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, k_len, self.head_dim)
        v = self.v_w(v_embd).view(batch_size, v_len, self.num_heads, self.output_hidden_size // self.num_heads)
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, v_len, self.output_hidden_size // self.num_heads)

        # get score
        if isinstance(self.score_func, str):
            if self.score_func == "dot":
                score = torch.bmm(q, k.permute(0, 2, 1))

            elif self.score_func == "scaled_dot":
                temp = torch.bmm(q, k.permute(0, 2, 1))
                score = torch.div(temp, math.sqrt(self.q_k_hidden_size))

            else:
                raise RuntimeError('invalid score function')
        elif callable(self.score_func):
            try:
                score = self.score_func(q, k)
            except Exception as e:
                print("Exception :", e)
        if mask is not None:
            mask = mask.bool().unsqueeze(1)
            score = score.masked_fill(~mask, -np.inf)
        score = nn.functional.softmax(score, dim=-1)
        score = nn.functional.dropout(score, p=self.drop_rate, training=self.training)

        # get output
        output = torch.bmm(score, v)
        heads = torch.split(output, batch_size)
        output = torch.cat(heads, -1)

        return output