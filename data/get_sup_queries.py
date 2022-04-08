import json
from tqdm import tqdm
from operator import itemgetter

from utils.misc import count_file_lines, AverageMeter
from utils.vocab import Vocab
from difflib import SequenceMatcher 

def similarity(a, b):    
    return SequenceMatcher(None, a, b).ratio() 

class Dataset_train():
    def __init__(self, vocab):
        train_file = './aol/train.json'
        data = []

        self.querys = set()
        self.vocab = vocab

        with open(train_file) as f:
            exm_num = count_file_lines(train_file)
            for line in tqdm(f, total=exm_num):
                data.append(json.loads(line))


        for session in tqdm(data):
            for qindex, q in enumerate(session['query']):
                query = q['text']
                if(len(query)>=0):
                    self.querys.add(query)

    def count_percent(self):
        total_percent = AverageMeter()
        d_pos_at_least_one_word_percent = AverageMeter()
        session_percent = AverageMeter()
        last_session_id = -1
        session_cnt = 0
        
        for query in self.querys:
            if(query["history_len"] == 0):
                continue
            session_id = query['session_id']
            
            d_pos = query["d_pos"]
            #print(query['history_len'], len(query['q_h']))
            sing_percent = AverageMeter()
            for pos_word in d_pos.split():
                cnt = 0
                for q in query['q_h']:
                    sentences = [q['qids']] + q['d_pos'] + q['d_neg']
                    for sen in sentences:
                        print(d_pos)
                        print(sen+'\n')
                        if pos_word in sen.split():
                            cnt = 1
                            session_cnt = 1
                            break
                    if cnt == 1:
                        break
                sing_percent.update(cnt)
            d_pos_at_least_one_word_percent.update(1 if sing_percent.avg>0 else 0)
            total_percent.update(sing_percent.avg)
            if session_id != last_session_id:
                session_percent.update(session_cnt)
                last_session_id = session_id
                session_cnt = 0

        return total_percent.avg, d_pos_at_least_one_word_percent.avg, session_percent.avg
    
    def pool_candidate_query(self, num_cand):
        train_file = './aol/train.json'
        new_train_file = './aol/train_candidate.json'
        data = []
        cands = {}

        with open(train_file) as f:
            exm_num = count_file_lines(train_file)
            for line in tqdm(f, total=exm_num):
                data.append(json.loads(line))
                if(len(data)>=exm_num):
                    break
        
        for session in tqdm(data, total=len(data)):
            for qindex, q in enumerate(session['query']):
                query = q['text']
                if(query in cands.keys()):
                    q['candidates'] = cands[query]
                    continue
                if('www' in query or 'com' in query):
                    q['candidates'] = [query] * 10
                    continue

                query_with_sim = {}
                for cand in (self.querys):
                    query_with_sim[cand] = self.cand_sim(query, cand)

                candidates = list(dict(sorted(query_with_sim.items(), key = itemgetter(1), reverse = True)[:num_cand]).keys())[:num_cand]
                
                cands[query] = candidates
                q['candidates'] = candidates
        
        with open(new_train_file, 'a') as output:
            for exm in tqdm(data, total=len(data)):
                jsonObj = json.dumps(exm, ensure_ascii=False)
                output.write(jsonObj + '\n')

    def cand_sim(self, query, cand):
        score = 0
        qw = query.split()
        cw = cand.split()
        if(set(qw) < set(cw)):
            score += (len(cand) - len(query)) / float(len(cand))
        qw = qw[:7]
        cw = cw[:7]
        
        score += similarity(qw, cw)
        
        return score

def main():

    emb_file = './aol/word2vec.txt'
    src_vocab = Vocab(emb_file)
    src_vocab.init_pretrained_embeddings(100, emb_file)

    dataset = Dataset_train(src_vocab)
    dataset.pool_candidate_query(10)
    #per = dataset.count_percent()
    #print(per)

if __name__ == '__main__':
    main()