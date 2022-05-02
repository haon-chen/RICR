import argparse
import time
import logging
import os
import glob
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler)
from torch.autograd import Variable
from utils.Trec_Metrics import Metrics

from utils.Constants import PAD, UNK_WORD
from utils.vocab import Vocab
from model import RICR
from datasets import Dataset_train, Dataset_score, collate_fn_train, collate_fn_score

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO,
                    filename='./ricr.log')
logger = logging.getLogger(__name__)

def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def set_seed(seed = 0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def train(args, train_loader, score_loader, model, optimizer, epoch):
    model.train()
    
    best_result_trec = {
        'map': 0.0,
        'mrr': 0.0,
        'ndcg@1': 0.0,
        'ndcg@3': 0.0,
        'ndcg@5': 0.0,
        'ndcg@10': 0.0,
    }
    for batch_idx, (q_c, history_len, q_h, d_hc, d_pos, d_neg, q_cands) in enumerate(train_loader):
        optimizer.zero_grad()
        if args.use_cuda:
            q_c = q_c.cuda(args.cuda_id)
            history_len = history_len.cuda(args.cuda_id)
            q_h = q_h.cuda(args.cuda_id)
            d_hc = d_hc.cuda(args.cuda_id)
            d_pos = d_pos.cuda(args.cuda_id)
            d_neg = d_neg.cuda(args.cuda_id)
            q_cands = q_cands.cuda(args.cuda_id)
        
        pos_score = model(q_c, history_len, q_h, d_hc, d_pos, q_cands)
        neg_score = model(q_c, history_len, q_h, d_hc, d_neg, q_cands)

        label = torch.ones(pos_score.size()).cuda()
        crit = nn.MarginRankingLoss(margin=1, size_average=True).cuda()
        loss = crit(pos_score, neg_score, Variable(label, requires_grad=False))
        
        if(batch_idx % (len(train_loader) // 5)) == 0:
            re = open(args.result_file_path,'a')
            re.write('Epoch {}\n'.format(epoch+1))
            re.close()
            best_result_trec = evaluate(args, score_loader, model, best_result_trec)
            model.train()
        
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm, norm_type=2)
        optimizer.step()


def evaluate(args, score_loader, model, best_result_trec):
    model.eval()
    model.apply(set_bn_eval)
    re = open(args.result_file_path,'a')
    score_file = open(args.score_file_path, 'w')
    for batch_idx, (q_c, history_len, q_h, d_hc, d_cand, label, q_cands) in enumerate(score_loader):
         
        if args.use_cuda:
            q_c = q_c.cuda(args.cuda_id)
            history_len = history_len.cuda(args.cuda_id)
            q_h = q_h.cuda(args.cuda_id)
            d_hc = d_hc.cuda(args.cuda_id)
            d_cand = d_cand.cuda(args.cuda_id)
            label = label.cuda(args.cuda_id)
            q_cands = q_cands.cuda(args.cuda_id)
            
        with torch.no_grad():
            score = model(q_c, history_len, q_h, d_hc, d_cand, q_cands)
            
            gt = label
            
            for score, label in zip(score.cpu().detach().numpy(), gt.cpu().detach().numpy()):
                score_file.write(str(score) + '\t' + str(label) + '\n')
    score_file.close()
    logger.info('*'*100)
    segment=50
    if(args.dataset == 'tg'):
        segment=10
    metric = Metrics(args.score_file_path, segment=segment)
    result_trec = metric.evaluate_all_metrics()

    if result_trec['map'] + result_trec['mrr'] + result_trec['ndcg@1'] + result_trec['ndcg@3'] + result_trec['ndcg@5'] + result_trec['ndcg@10'] > best_result_trec['map'] + best_result_trec['mrr'] + best_result_trec['ndcg@1'] + best_result_trec['ndcg@3'] + best_result_trec['ndcg@5'] + best_result_trec['ndcg@10']:
        best_result_trec = {
            'map': result_trec['map'],
            'mrr': result_trec['mrr'],
            'ndcg@1': result_trec['ndcg@1'],
            'ndcg@3': result_trec['ndcg@3'],
            'ndcg@5': result_trec['ndcg@5'],
            'ndcg@10': result_trec['ndcg@10'],
        }

    re.write('*'*100+"\n")
    re.write('Model {}\n'.format(args.model))
    re.write('params {}\n'.format(args))
    re.write('MAP {}\n'.format(best_result_trec['map']))
    re.write('MRR {}\n'.format(best_result_trec['mrr']))
    re.write('NDCG@1 {}\n'.format(best_result_trec['ndcg@1']))
    re.write('NDCG@3 {}\n'.format(best_result_trec['ndcg@3']))
    re.write('NDCG@5 {}\n'.format(best_result_trec['ndcg@5']))
    re.write('NDCG@10 {}\n'.format(best_result_trec['ndcg@10']))
    re.write('*'*100+"\n")
    re.close()


    return best_result_trec


def main(args):
    # initialize
    logger.info("Initializing gpu...")
    
    set_seed(2021)
    # device_ids = [0, 1]
    epoch_num = args.epoch_num
    if(args.dataset == 'aol'):
        args.emb_file = args.emb_file + '/aol/word2vec.txt'
    elif(args.dataset == 'tg'):
        args.emb_file = args.emb_file + '/tiangong/word2vec.model'
        
    print(args.emb_file)
    logger.info("Building Dictionary...")
    src_vocab = Vocab(args.emb_file)
    src_vocab.init_pretrained_embeddings(args.d_word_vec, args.emb_file)
    logger.info("Dictionary Built.")

    model = None
    if args.model == 'ricr':
        model = RICR(args, src_vocab)
    else:
        raise ("Not implement!")
    assert model != None

    if args.use_cuda:
        model = model.cuda(args.cuda_id)

    optimizer = torch.optim.AdamW(model.parameters(), 
        lr=args.lr,
    )

    # DataLoaders
    if args.do_train:
        train_dataset = Dataset_train(args, src_vocab)
        train_sampler = RandomSampler(train_dataset)
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=args.train_batch_size,
            sampler=train_sampler,
            collate_fn=collate_fn_train)
    if args.do_eval:
        score_dataset = Dataset_score(args, src_vocab)
        score_sampler = SequentialSampler(score_dataset)
        score_loader = DataLoader(
            dataset=score_dataset,
            batch_size=args.score_batch_size,
            sampler=score_sampler,
            collate_fn=collate_fn_score)
    
    if args.do_train:
        for epoch in range(epoch_num):
            logger.info("Training for epoch %s..." %(epoch+1))
            begin = time.time()
            train(args, train_loader, score_loader, model, optimizer, epoch)
            end = time.time()
            logger.info("Training time is %s" %(end-begin))

            if args.save_epochs > 0 and epoch % args.save_epochs == 0:
                # Save model checkpoint if it outperforms previous models
                # Only evaluate when single GPU otherwise metrics may not average well
                output_dir = os.path.join(args.checkpoint_dir, 'checkpoint-{}'.format(epoch+1))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                logger.info("Saving model checkpoint to %s", output_dir)
                model_name = os.path.join(output_dir, args.model)
                torch.save(model.state_dict(), model_name)
    
    if args.do_eval and not args.do_train:
        logger.info("Eval on all checkpoints with dev set")
        checkpoints = [args.checkpoint_dir]
        checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.checkpoint_dir + '/**/' + args.model, recursive=True)))
        logger.info("Evaluate the following checkpoints: {}".format(checkpoints))
        for checkpoint in checkpoints:
            epoch = checkpoint.split('-')[-1]
            logger.info('epoch {}'.format(epoch))
            # model = model_class.from_pretrained(checkpoint)

            state_dict = torch.load(os.path.join(checkpoint, args.model), 
                map_location=lambda storage, loc: storage.cuda(args.cuda_id) if args.use_cuda else storage)
            model.load_state_dict(state_dict)
            if args.use_cuda:
                model = model.cuda(args.cuda_id)
            logger.info("Scoring for epoch {}...".format(epoch))
            begin = time.time()

            best_result_trec = {
                'map': 0.0,
                'mrr': 0.0,
                'ndcg@1': 0.0,
                'ndcg@3': 0.0,
                'ndcg@5': 0.0,
                'ndcg@10': 0.0,
            }

            re = open(args.result_file_path,'a')
            re.write('Epoch {}\n'.format(epoch))
            re.close()
            best_result_trec = evaluate(args, score_loader, model, best_result_trec)
            
            end = time.time()
            logger.info("Scoring time is {}".format(end-begin))
    
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', required=True)
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--emb_file', required=True)
    parser.add_argument('--checkpoint_dir', required=True)
    parser.add_argument('--score_file_path', required=True)
    parser.add_argument('--result_file_path', required=True)
    parser.add_argument('--dataset', required=True)

    parser.add_argument('--use_cuda', type=str2bool, default=True)
    parser.add_argument('--last_q', type=str2bool, default=False, help="whether to use Tiangong-ST-relevace")
    parser.add_argument('--cuda_id', type=int, default=0)
    parser.add_argument('--save_epochs', type=int, default=-1)
    parser.add_argument('--do_eval', type=str2bool, default=True)
    parser.add_argument('--do_train', type=str2bool, default=True)
    

    parser.add_argument('--epoch_num', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--d_word_vec', type=int, default=100)
    parser.add_argument('--train_batch_size', type=int, default=100)
    parser.add_argument('--score_batch_size', type=int, default=100)
    
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument("--max_grad_norm", default=5.0, type=float)
    parser.add_argument('--n_kernels', type=int, default=11)
    parser.add_argument('--num_heads', type=int, default=1)
    parser.add_argument('--max_doc_len', type=int, default=15)
    parser.add_argument('--max_query_len', type=int, default=7)
    parser.add_argument('--history_num', type=int, default=5)
    parser.add_argument('--d_hid_qat', type=int, default=256)
    parser.add_argument('--d_hid_rnn', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--layer_num', type=int, default=1)
    parser.add_argument('--rnn_type', default='GRU')
    

    args = parser.parse_args()
    args.cuda = args.use_cuda
    logger.info(args)
    
    main(args)