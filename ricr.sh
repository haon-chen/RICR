DATA_DIR=./data
CHECKPOINT_DIR=./checkpoints
SCORE_FILE_DIR=./output

CUDA_VISIBLE_DEVICES=0 python -W ignore ./main.py \
--model ricr \
--data_path ${DATA_DIR} \
--dataset tg \
--score_file_path ${SCORE_FILE_DIR}/score_file.txt \
--result_file_path ${SCORE_FILE_DIR}/results.txt \
--max_doc_len 17 \
--max_query_len 9 \
--d_word_vec 100 \
--train_batch_size 512 \
--score_batch_size 200 \
--num_workers 0 \
--epoch_num 10 \
--lr 1e-3 \
--n_kernels 21 \
--num_heads 1 \
--emb_file ${DATA_DIR} \
--history_num 7 \
--d_hid_qat 100 \
--d_hid_rnn 256 \
--layer_num 1 \
--rnn_type 'GRU' \
--save_epochs 1 \
--checkpoint_dir ${CHECKPOINT_DIR} \
--do_eval True \
--do_train True \
--use_cuda True \
--last_q False