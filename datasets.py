import json
from tqdm import tqdm
import copy
import logging
import itertools

import torch

from utils.Constants import PAD, UNK_WORD
from utils.misc import count_file_lines

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO,
                    filename='./ricr.log')

logger = logging.getLogger(__name__)

def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(PAD).type(torch.float).unsqueeze(-1)

def collate_fn_train(insts):
    ''' Pad the instance to the max seq length in batch '''
    q_c, history_len, q_h, d_hc, d_pos, d_neg, q_cands = zip(*insts)

    
    q_c = torch.LongTensor(q_c)
    history_len = torch.LongTensor(history_len)
    q_h = torch.LongTensor(q_h)
    d_hc = torch.LongTensor(d_hc)
    d_pos = torch.LongTensor(d_pos)
    d_neg = torch.LongTensor(d_neg)
    q_cands = torch.LongTensor(q_cands)



    return q_c, history_len, q_h, d_hc, d_pos, d_neg, q_cands


class Dataset_train(torch.utils.data.Dataset):
    def __init__(self, args, vocab):
        if args.dataset == 'aol':
            in_path = args.data_path + '/aol'
            train_file = in_path + "/train_candidate.json"
        if args.dataset == 'tg':
            in_path = args.data_path + '/tiangong'
            train_file = in_path + "/train_candidate.json"
        train_data = []
        self.vocab = vocab

        self.sessions = {}

        with open(train_file) as f:
            exm_num = count_file_lines(train_file)
            for line in tqdm(f, total=exm_num, position=0):
                train_data.append(json.loads(line))
                if(len(train_data)>=exm_num):
                    break
        
        logger.info("Using %s examples as training data..." %len(train_data))

        logger.info("Loading vocab...")

        example_num = 0
        if args.dataset == 'aol':
            for session in tqdm(train_data, position=0):
                example_num += 1
                q_cs = []
                session_id = session['session_id']
                self.sessions[session_id] = []
                query = []

                for qindex, q in enumerate(session['query']):
                    query = q['text']
                    qids = sen2qid(args, query, self.vocab)
                    d_negs = []
                    d_poss = []
                    q_cands = [sen2qid(args, cand, self.vocab) for cand in q['candidates']]
                    assert len(q_cands) == 10

                    for dindex, candidate in enumerate(q['clicks']):
                        title = candidate['title']
                        dids = sen2did(args, title, self.vocab)
                        if isinstance(candidate['label'], bool):
                            if(candidate['label']):
                                d_poss.append(dids)
                            else:
                                d_negs.append(dids)
                        else:
                            if(candidate['label'] == 1):
                                d_poss.append(dids)
                            else:
                                d_negs.append(dids)
                    if(len(d_negs)==0 or len(d_poss)==0):
                        continue
                    else:
                        history_len = args.history_num
                        if(history_len == 0):
                            history_len+=1
                        assert history_len!=0

                        q_h = pad_train_history_head_aol(args, q_cs, self.vocab)
                        for d_pos in d_poss:
                            for d_neg in d_negs:

                                q_c = {
                                    'qids': qids,
                                    'q_h': q_h,
                                    'q_cands': q_cands,
                                    'history_len': history_len,
                                    'd_pos': d_pos,
                                    'd_neg': d_neg,
                                }
                                self.sessions[session_id].append(q_c)
                        
                        q_c = {
                            'qids': qids,
                            'd_pos': d_poss,
                            'd_neg': d_negs
                        }
                        q_cs.append(q_c)
            self.querys = list(itertools.chain(*(self.sessions.values())))

        else:
            for session in tqdm(train_data, position=0):
                example_num += 1
                q_cs = []
                session_id = session['session_id']
                self.sessions[session_id] = []
                query = []

                for qindex, q in enumerate(session['query']):
                    query = q['text']
                    qids = sen2qid(args, query, self.vocab)
                    d_negs = []
                    d_poss = []
                    q_cands = [sen2qid(args, cand, self.vocab) for cand in q['candidates']]
                    assert len(q_cands) == 10

                    for dindex, candidate in enumerate(q['clicks']):
                        title = candidate['title']
                        dids = sen2did(args, title, self.vocab)
                        if 'label' in candidate:
                            d_poss.append(dids)
                        else:
                            d_negs.append(dids)
                    if(len(d_negs)==0 or len(d_poss)==0):
                        continue
                    else:
                        history_len = args.history_num
                        if(history_len == 0):
                            history_len+=1
                        assert history_len!=0

                        q_h = pad_train_history_head_tg(args, q_cs, self.vocab)
                        for d_pos in d_poss:
                            for d_neg in d_negs:

                                q_c = {
                                    'qids': qids,
                                    'q_h': q_h,
                                    'q_cands': q_cands,
                                    'history_len': history_len,
                                    'd_pos': d_pos,
                                    'd_neg': d_neg,
                                }
                                self.sessions[session_id].append(q_c)
                        
                        q_c = {
                            'qids': qids,
                            'd_pos': d_poss,
                            'd_neg': d_negs
                        }
                        q_cs.append(q_c)
            self.querys = list(itertools.chain(*(self.sessions.values())))


    def __len__(self):
        return len(self.querys)

    def __getitem__(self, idx):
        query = self.querys[idx]
        q_h = []
        d_hc = []
        
        for ind, q in enumerate(query['q_h']):
            q_h.append(q['qids'])
            d_hc.append([d for d in q['d_pos']])
        return query['qids'], query['history_len'], q_h, d_hc, query['d_pos'], query['d_neg'], query['q_cands']


def collate_fn_score(insts):
    ''' Pad the instance to the max seq length in batch '''
    q_c, history_len, q_h, d_hc, d_cand, label, q_cands = zip(*insts)
    
    q_c = torch.LongTensor(q_c)
    history_len = torch.LongTensor(history_len)
    q_h = torch.LongTensor(q_h)
    d_hc = torch.LongTensor(d_hc)
    d_cand = torch.LongTensor(d_cand)
    label = torch.LongTensor(label)
    q_cands = torch.LongTensor(q_cands)

    return q_c, history_len, q_h, d_hc, d_cand, label, q_cands


class Dataset_score(torch.utils.data.Dataset):
    def __init__(self, args, vocab):
        if args.dataset == 'aol':
            in_path = args.data_path + '/aol'
            #test_file = in_path + "/dev_candidate.json"
            test_file = in_path + "/test_candidate.json"
        if args.dataset == 'tg':
            in_path = args.data_path + '/tiangong'
            #test_file = in_path + "/dev_candidate.json"
            test_file = in_path + "/test_candidate.json"
        
        test_data = []
        self.vocab = vocab

        self.last_querys = []
        self.querys = []
        self.sessions = {}

        with open(test_file) as f:
            exm_num = count_file_lines(test_file)
            for line in tqdm(f, total=exm_num, position=0):
                test_data.append(json.loads(line))
                if(len(test_data)>=exm_num):
                    break
        
        logger.info("Using %s examples as scoring data..." %len(test_data))

        logger.info("Loading vocab...")

        example_num = 0

        if args.dataset == 'aol':
            for session in tqdm(test_data, position=0):
                example_num += 1
                q_cs = []
                session_id = session['session_id']
                self.sessions[session_id] = []

                for qindex, q in enumerate(session['query']):
                    query = q['text']
                    qids = sen2qid(args, query, self.vocab)
                    d_negs = []
                    d_poss = []
                    q_cands = [sen2qid(args, cand, self.vocab) for cand in q['candidates']]
                    assert len(q_cands) == 10

                    for dindex, candidate in enumerate(q['clicks']):
                        title = candidate['title']
                        dids = sen2did(args, title, self.vocab)
                        if isinstance(candidate['label'], bool):
                            if(candidate['label']):
                                d_poss.append(dids)
                            else:
                                d_negs.append(dids)
                        else:
                            if(candidate['label'] == 1):
                                d_poss.append(dids)
                            else:
                                d_negs.append(dids)
                    if(len(d_negs)==0 or len(d_poss)==0):
                        continue
                    else:
                        q_h = pad_score_history_head_aol(args, q_cs, self.vocab)
                        history_len = args.history_num
                        if(history_len == 0):
                            history_len+=1
                        assert history_len!=0
                        self.sessions[session_id].extend([{
                            'qids': qids,
                            'q_cands': q_cands,
                            'q_h': q_h,
                            'history_len': history_len,
                            'd_cand': d,
                            'label': 1,
                            'candidates': q['candidates'],
                        } for d in d_poss])
                        self.sessions[session_id].extend([{
                            'qids': qids,
                            'q_cands': q_cands,
                            'q_h': q_h,
                            'history_len': history_len,
                            'd_cand': d,
                            'label': 0,
                            'candidates': q['candidates'],
                        } for d in d_negs])
                        q_c = {
                            'qids': qids,
                            'd_pos': d_poss,
                            'd_neg': d_negs
                        }
                        q_cs.append(q_c)
            self.querys = list(itertools.chain(*(self.sessions.values())))
        
        else:
            for session in tqdm(test_data, position=0):
                example_num += 1
                q_cs = []
                session_id = session['session_id']

                for qindex, q in enumerate(session['query']):
                    is_final_query = False
                    query = q['text']
                    qids = sen2qid(args, query, self.vocab)
                    d_negs = []
                    d_poss = []
                    last_q_d_negs = {}
                    last_q_d_poss = {}
                    q_cands = [sen2qid(args, cand, self.vocab) for cand in q['candidates']]
                    assert len(q_cands) == 10

                    for dindex, candidate in enumerate(q['clicks']):
                        title = candidate['title']
                        dids = sen2did(args, title, self.vocab)
                        dids_str = ''.join(list(map(str,dids)))
                        if 'label' in candidate:
                            if isinstance(candidate['label'], bool):
                                if(candidate['label']):
                                    d_poss.append(dids)
                                else:
                                    d_negs.append(dids)
                            else:
                                is_final_query = True
                                if(candidate['label'] == '0'):
                                    d_negs.append(dids)
                                    last_q_d_negs[dids_str] = int(candidate['label'])
                                else:
                                    d_poss.append(dids)
                                    last_q_d_poss[dids_str] = int(candidate['label'])
                        else:
                            d_negs.append(dids)
                    q_h = pad_score_history_head_tg(args, q_cs, self.vocab)
                    history_len = args.history_num
                    if(history_len == 0):
                        history_len+=1
                    assert history_len!=0
                    if is_final_query:
                        self.last_querys.extend([{
                            'qids': qids,
                            'q_cands': q_cands,
                            'q_h': q_h,
                            'history_len': history_len,
                            'd_cand': d,
                            'label': last_q_d_poss[''.join(list(map(str,d)))],
                            'candidates': q['candidates'],
                            'q_text': query
                        } for d in d_poss])
                        self.last_querys.extend([{
                            'qids': qids,
                            'q_cands': q_cands,
                            'q_h': q_h,
                            'history_len': history_len,
                            'd_cand': d,
                            'label': last_q_d_negs[''.join(list(map(str,d)))],
                            'candidates': q['candidates'],
                            'q_text': query
                        } for d in d_negs])
                    else:
                        self.querys.extend([{
                            'qids': qids,
                            'q_cands': q_cands,
                            'q_h': q_h,
                            'history_len': history_len,
                            'd_cand': d,
                            'label': 1,
                            'candidates': q['candidates'],
                            'q_text': query
                        } for d in d_poss])
                        self.querys.extend([{
                            'qids': qids,
                            'q_cands': q_cands,
                            'q_h': q_h,
                            'history_len': history_len,
                            'd_cand': d,
                            'label': 0,
                            'candidates': q['candidates'],
                            'q_text': query
                        } for d in d_negs])
                    q_c = {
                        'qids': qids,
                        'd_pos': d_poss,
                        'd_neg': d_negs
                    }
                    q_cs.append(q_c)
            if args.last_q:
                self.querys = self.last_querys

    def __len__(self):
        return len(self.querys)

    def __getitem__(self, idx):
        query = self.querys[idx]
        q_h = []
        d_hc = []

        for ind, q in enumerate(query['q_h']):
            q_h.append(q['qids'])
            d_hc.append([d for d in q['d_pos']])

        return query['qids'], query['history_len'], q_h, d_hc, query['d_cand'], query['label'], query['q_cands']


def sen2qid(args, sen, vocab):
    idx = [vocab.get_id(token) for token in sen.split()]
    idx = [vocab.get_id(vocab.bos_token)] + idx + [vocab.get_id(vocab.eos_token)]
    idx = idx[:args.max_query_len]
    padding = [vocab.get_id(vocab.pad_token)] * (args.max_query_len - len(idx))
    idx = padding + idx
    return idx

def sen2did(args, sen, vocab):
    idx = [vocab.get_id(token) for token in sen.split()]
    idx = [vocab.get_id(vocab.bos_token)] + idx + [vocab.get_id(vocab.eos_token)]
    idx = idx[:args.max_doc_len]
    padding = [vocab.get_id(vocab.pad_token)] * (args.max_doc_len - len(idx))
    #idx = idx + padding
    idx = padding + idx
    return idx

def pad_score_history_head_tg(args, history, vocab):
    pad_id = vocab.get_id(vocab.pad_token)
    bos_id = vocab.get_id(vocab.bos_token)
    eos_id = vocab.get_id(vocab.eos_token)
    history = history[-args.history_num:]
    padded_history = copy.deepcopy(history)
    for his in padded_history:
        his['d_pos'] = his['d_pos'][-10:]
        his['d_neg'] = his['d_neg'][-10:]
        while len(his['d_pos']) < 10:
            his['d_pos'].append([pad_id] * (args.max_doc_len - 2) + [bos_id] + [eos_id])
        while len(his['d_neg']) < 10:
            his['d_neg'].append([pad_id] * (args.max_doc_len - 2) + [bos_id] + [eos_id])
    
    while len(padded_history) < args.history_num:
        padded_history.insert(0, {
            'qids': [pad_id] * (args.max_query_len - 2) + [bos_id] + [eos_id],
            'd_pos': [[pad_id] * (args.max_doc_len - 2) + [bos_id] + [eos_id]] * 10,
            'd_neg': [[pad_id] * (args.max_doc_len - 2) + [bos_id] + [eos_id]] * 10
        })
    return padded_history

def pad_train_history_head_tg(args, history, vocab):
    pad_id = vocab.get_id(vocab.pad_token)
    bos_id = vocab.get_id(vocab.bos_token)
    eos_id = vocab.get_id(vocab.eos_token)
    history = history[-args.history_num:]
    padded_history = copy.deepcopy(history)
    for his in padded_history:
        his['d_pos'] = his['d_pos'][-10:]
        his['d_neg'] = his['d_neg'][-10:]
        while len(his['d_pos']) < 10:
            his['d_pos'].append([pad_id] * (args.max_doc_len - 2) + [bos_id] + [eos_id])
        while len(his['d_neg']) < 10:
            his['d_neg'].append([pad_id] * (args.max_doc_len - 2) + [bos_id] + [eos_id])
    
    while len(padded_history) < 10:
        padded_history.insert(0, {
            'qids': [pad_id] * (args.max_query_len - 2) + [bos_id] + [eos_id],
            'd_pos': [[pad_id] * (args.max_doc_len - 2) + [bos_id] + [eos_id]] * 10,
            'd_neg': [[pad_id] * (args.max_doc_len - 2) + [bos_id] + [eos_id]] * 10
        })
    return padded_history

def pad_score_history_head_aol(args, history, vocab):
    pad_id = vocab.get_id(vocab.pad_token)
    bos_id = vocab.get_id(vocab.bos_token)
    eos_id = vocab.get_id(vocab.eos_token)
    history = history[-args.history_num:]
    padded_history = copy.deepcopy(history)
    for his in padded_history:
        his['d_pos'] = his['d_pos'][-1:]
        his['d_neg'] = his['d_neg'][-49:]
        while len(his['d_neg']) < 49:
            his['d_neg'].append([pad_id] * (args.max_doc_len - 2) + [bos_id] + [eos_id])
    
    while len(padded_history) < args.history_num:
        padded_history.insert(0, {
            'qids': [pad_id] * (args.max_query_len - 2) + [bos_id] + [eos_id],
            'd_pos': [[pad_id] * (args.max_doc_len - 2) + [bos_id] + [eos_id]],
            'd_neg': [[pad_id] * (args.max_doc_len - 2) + [bos_id] + [eos_id]] * 49
        })
    return padded_history

def pad_train_history_head_aol(args, history, vocab):
    pad_id = vocab.get_id(vocab.pad_token)
    bos_id = vocab.get_id(vocab.bos_token)
    eos_id = vocab.get_id(vocab.eos_token)
    history = history[-args.history_num:]
    padded_history = copy.deepcopy(history)
    for his in padded_history:
        his['d_pos'] = his['d_pos'][-5:]
        his['d_neg'] = his['d_neg'][-5:]
        while len(his['d_pos']) < 5:
            his['d_pos'].append([pad_id] * (args.max_doc_len - 2) + [bos_id] + [eos_id])
        while len(his['d_neg']) < 5:
            his['d_neg'].append([pad_id] * (args.max_doc_len - 2) + [bos_id] + [eos_id])
    
    while len(padded_history) < args.history_num:
        padded_history.insert(0, {
            'qids': [pad_id] * (args.max_query_len - 2) + [bos_id] + [eos_id],
            'd_pos': [[pad_id] * (args.max_doc_len - 2) + [bos_id] + [eos_id]] * 5,
            'd_neg': [[pad_id] * (args.max_doc_len - 2) + [bos_id] + [eos_id]] * 5
        })
    return padded_history
