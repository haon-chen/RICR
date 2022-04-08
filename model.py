import torch.nn as nn
import torch
import torch.nn.functional as F


from encoder import Encoder
from layers import knrm, conv_knrm
from utils.Constants import PAD
from enhancer import Enhancer

def get_non_pad_mask(seq):
    #assert seq.dim() == 2
    return seq.ne(PAD).type(torch.float).unsqueeze(-1)

class RICR(nn.Module):
    
    def load_embedding(self, args): # load the pretrained embedding
        weight = torch.zeros(len(self.src_vocab), self.d_word_vec)
        with open(args.emb_file, 'r') as fr:
            for line in fr:
                line = line.strip().split()
                wordid = self.src_vocab[line[0]]
                weight[wordid, :] = torch.FloatTensor([float(t) for t in line[1:]]) 
        print("Successfully load the word vectors...")
        return weight

    def __init__(self, args, src_vocab) -> None:
        super(RICR, self).__init__()

        # Embedding Layer
        
        self.pad_idx = src_vocab.get_id(src_vocab.pad_token)
        self.bos_idx = src_vocab.get_id(src_vocab.bos_token)
        self.eos_idx = src_vocab.get_id(src_vocab.eos_token)

        self.d_word_vec = args.d_word_vec
        self.src_vocab = src_vocab
        self.embedding = nn.Embedding(len(src_vocab.embeddings), self.d_word_vec, padding_idx = self.pad_idx)
        self.embedding.weight.data.copy_(torch.from_numpy(src_vocab.embeddings))

        # Encoder Layer

        self.encoder = Encoder(args)

        # Enhance current query, candidate document, and supplemental query with session history

        self.query_enhancement = Enhancer(args.d_word_vec, args.max_query_len, args.d_hid_rnn, dropout=0.1)
        self.document_enhancement = Enhancer(args.d_word_vec, args.max_doc_len, args.d_hid_rnn, dropout=0.1)
        self.select_query_enhancement = Enhancer(args.d_word_vec, args.max_query_len, args.d_hid_rnn, dropout=0.1)

        self.hidden2output = nn.Linear(args.d_hid_rnn, args.d_word_vec)

        # Supplemental Query Selection

        self.query_selector = nn.Linear(args.d_hid_rnn + args.d_word_vec, 1)

        # Ranking Layer

        self.knrm_layer1 = conv_knrm(args)
        self.knrm_layer2 = conv_knrm(args)

        self.knrm_layer3 = conv_knrm(args)
        self.knrm_layer4 = conv_knrm(args)

        self.knrm_layer5 = conv_knrm(args)
        self.knrm_layer6 = conv_knrm(args)
        
        self.knrm_layer7 = conv_knrm(args)
        self.knrm_layer8 = conv_knrm(args)

        self.ranker = nn.Linear(8, 1, 1)

        need_init = [self.ranker, self.hidden2output, self.query_selector]
        for layer in need_init:
            for p in layer.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p) 

    
    def forward(self, q_c, history_len, q_h, d_hc, d_cand, q_cands):
        # Embedding
        
        q_c_embed = self.embedding(q_c)
        q_h_embed = self.embedding(q_h)
        d_hc_embed = self.embedding(d_hc)
        d_cand_embed = self.embedding(d_cand)
        # (B, num_cands, num_words, d_emb) 512, 10, 9, 100
        q_cands_embed = self.embedding(q_cands)

        d_hc_embed = torch.mean(d_hc_embed, dim=2, keepdim=False)

        current_query_mask = get_non_pad_mask(q_c)
        candidate_mask = get_non_pad_mask(d_cand)

        history_encoded, hiddens = self.encoder(q_c_embed, history_len, q_h_embed, d_hc_embed)
        
        enhanced_query_embed, q_hiddens = self.query_enhancement(q_c_embed, hiddens, history_encoded)
        enhanced_doc_embed, d_hiddens = self.document_enhancement(d_cand_embed, hiddens, history_encoded)

        enhanced_query_embed = enhanced_query_embed.masked_fill(q_c.eq(self.pad_idx).unsqueeze(-1), value=0)
        enhanced_doc_embed = enhanced_doc_embed.masked_fill(d_cand.eq(self.pad_idx).unsqueeze(-1), value=0)
        
        score = self.knrm_layer1(q_c_embed, d_cand_embed, current_query_mask, candidate_mask)
        query_enhancement_score = self.knrm_layer2(enhanced_query_embed, d_cand_embed, current_query_mask, candidate_mask)

        enhancement_score = self.knrm_layer3(enhanced_query_embed, enhanced_doc_embed, current_query_mask, candidate_mask)
        doc_generation_score = self.knrm_layer4(q_c_embed, enhanced_doc_embed, current_query_mask, candidate_mask)
        
        # (B, num_cands, num_words, d_emb)
        q_cands_embed_1 = torch.mean(q_cands_embed, dim=2, keepdim=False)
        # (B, num_cands, d_emb)
        q_hidden = q_hiddens[:,-1,:].unsqueeze(1).expand(q_hiddens[:,-1,:].size(0), q_cands_embed_1.size(1), q_hiddens[:,-1,:].size(1))
        
        select_prob = (torch.relu(self.query_selector(torch.cat([q_hidden, q_cands_embed_1], dim=-1)))).squeeze(-1)
        selection = F.softmax(select_prob, dim=-1)
        tgt = torch.max(selection, dim=1, keepdim=True)[1].squeeze(-1)
        selected_query = []
        selected_query_embed = []

        for ind, x in enumerate(torch.split(q_cands,1, dim=0)):
            selected_query.append(x[:,tgt[ind],:])
        
        for ind, x in enumerate(torch.split(q_cands_embed,1, dim=0)):
            selected_query_embed.append(x[:,tgt[ind],:,:])
        
        selected_query = torch.cat(selected_query, dim=0)
        selected_query_embed = torch.cat(selected_query_embed, dim=0)
        select_query_mask = get_non_pad_mask(selected_query)

        enhanced_select_query_embed, refor_q_hiddens = self.select_query_enhancement(selected_query_embed, hiddens, history_encoded)
        enhanced_select_query_embed = enhanced_select_query_embed.masked_fill(selected_query.eq(self.pad_idx).unsqueeze(-1), value=0)
        

        sup_query_score1 = self.knrm_layer5(selected_query_embed, d_cand_embed, select_query_mask, candidate_mask)
        sup_query_score2 = self.knrm_layer6(selected_query_embed, enhanced_doc_embed, select_query_mask, candidate_mask)

        enhan_sup_query_score1 = self.knrm_layer7(enhanced_select_query_embed, d_cand_embed, select_query_mask, candidate_mask)
        enhan_sup_query_score2 = self.knrm_layer8(enhanced_select_query_embed, enhanced_doc_embed, select_query_mask, candidate_mask)

        scores = torch.stack((score, query_enhancement_score, sup_query_score1, sup_query_score2, enhan_sup_query_score1, enhan_sup_query_score2, enhancement_score, doc_generation_score), -1)

        score = F.tanh(self.ranker(scores)).squeeze(-1)
        
        return score