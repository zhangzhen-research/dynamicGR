# environment
transformers==4.18.0
nanopq
pip install SentencePiece
datasets==1.18.4

# command

# PQ

## prepare data

gen nq data

get_corpus_vectors

get_docids

query_generation

gen t5 train data

get docids pre

## training


#!/bin/bash

python runT5.py \
--epoch 10 \
--per_gpu_batch_size  100 \
--learning_rate 1e-3 \
--save_path ../outputs/nq/stage1 \
--log_path ../logs/pretrain.msmarco.log \
--doc_file_path ../dataset/msmarco-data/msmarco-docs-sents-all.json \
--pretrain_model_path ../../../../huggingface/t5-base \
--docid_path ../dataset/encoded_docid/t5_pq_all.txt \
--train_file_path ../dataset/msmarco-data/train_data_pq/pretrain_data.json \
--test_file_path ../dataset/msmarco-data/train_data_pq/eval_query2docid.json \
--dataset_script_dir ../data_scripts \
--dataset_cache_dir ../../negs_tutorial_cache \
--add_doc_num 6144 \
--max_seq_length 128 \
--max_docid_length 24 \
--use_origin_head False \
--output_every_n_step 5000 \
--save_every_n_epoch 2 \
--operation training

python runT5.py \
--epoch 20 \
--per_gpu_batch_size  100 \
--learning_rate 1e-3 \
--save_path ../outputs/msmarco/stage2 \
--log_path ../logs/searchtrain.msmarco.log \
--doc_file_path ../dataset/msmarco-data/msmarco-docs-sents-all.json \
--pretrain_model_path ../../../../huggingface/t5-base \
--docid_path ../dataset/encoded_docid/t5_pq_all.txt \
--train_file_path ../dataset/msmarco-data/train_data_pq/pseudo2docid.json \
--test_file_path ../dataset/msmarco-data/train_data_pq/eval_query2docid.json \
--dataset_script_dir ../data_scripts \
--dataset_cache_dir ../../negs_tutorial_cache \
--add_doc_num 6144 \
--max_seq_length 128 \
--max_docid_length 24 \
--use_origin_head 0 \
--load_ckpt True \
--load_ckpt_path ../outputs/msmarco/stage1/model_9.pkl \
--output_every_n_step 5000 \
--save_every_n_epoch 4 \
--operation training

python runT5.py \
--epoch 10 \
--per_gpu_batch_size  100 \
--learning_rate 1e-3 \
--save_path ../outputs/msmarco/stage3 \
--log_path ../logs/finetune.msmarco.log \
--doc_file_path ../dataset/msmarco-data/msmarco-docs-sents-all.json \
--pretrain_model_path ../../../../huggingface/t5-base \
--docid_path ../dataset/encoded_docid/t5_pq_all.txt \
--train_file_path ../dataset/msmarco-data/train_data_pq/query2docid.json \
--test_file_path ../dataset/msmarco-data/train_data_pq/eval_query2docid.json \
--dataset_script_dir ../data_scripts \
--dataset_cache_dir ../../negs_tutorial_cache \
--add_doc_num 6144 \
--max_seq_length 64 \
--max_docid_length 24 \
--use_origin_head 0 \
--load_ckpt True \
--load_ckpt_path ../outputs/msmarco/stage2/model_19.pkl \
--output_every_n_step 5000 \
--save_every_n_epoch 2 \
--operation training


python runT5.py \
--epoch 10 \
--per_gpu_batch_size 4 \
--learning_rate 1e-3 \
--save_path ../outputs/nq/stage3/model_9.pkl \
--log_path ../logs/result.log \
--doc_file_path ../dataset/nq-data/nq-docs-sents-all.json \
--pretrain_model_path ../../../../../huggingface/t5-base \
--docid_path ../dataset/encoded_docid/t5_pq_all.txt \
--train_file_path ../dataset/nq-data/train_data/query2docid.json \
--test_file_path ../dataset/nq-data/train_data/eval_query2docid.json \
--dataset_script_dir ../data_scripts \
--dataset_cache_dir ../../negs_tutorial_cache \
--num_beams 100 \
--add_doc_num 6144 \
--max_seq_length 64 \
--max_docid_length 24 \
--output_every_n_step 1000 \
--save_every_n_epoch 2 \
--operation testing \
--use_docid_rank True \
--per_gpu_batch_size 2


#!/bin/bash

for i in {1..5}
do
  python runT5.py \
  --epoch 10 \
  --per_gpu_batch_size 4 \
  --learning_rate 1e-3 \
  --save_path ../outputs/nq/stage3/model_9.pkl \
  --log_path ../logs/result_new_${i}.log \
  --doc_file_path ../dataset/nq-data/nq-docs-sents-all_${i}.json \
  --pretrain_model_path ../../../../../huggingface/t5-base \
  --docid_path ../dataset/encoded_docid/t5_pq_all_${i}.txt \
  --train_file_path ../dataset/nq-data/train_data/query2docid.json \
  --test_file_path ../dataset/nq-data/train_data/eval_query2docid_${i}.json \
  --dataset_script_dir ../data_scripts \
  --dataset_cache_dir ../../negs_tutorial_cache \
  --num_beams 100 \
  --add_doc_num 6144 \
  --max_seq_length 64 \
  --max_docid_length 24 \
  --output_every_n_step 1000 \
  --save_every_n_epoch 2 \
  --operation testing \
  --use_docid_rank True
done

#!/bin/bash

for i in {1..6}
do
  python runT5.py \
  --epoch 10 \
  --per_gpu_batch_size 4 \
  --learning_rate 1e-3 \
  --save_path ../outputs/nq/stage3/model_9.pkl \
  --log_path ../logs/result_${i}.log \
  --doc_file_path ../dataset/nq-data/nq-docs-sents-all_${i}.json \
  --pretrain_model_path ../../../../../huggingface/t5-base \
  --docid_path ../dataset/encoded_docid/t5_pq_all_${i}.txt \
  --train_file_path ../dataset/nq-data/train_data/query2docid.json \
  --test_file_path ../dataset/nq-data/train_data/eval_query2docid.json \
  --dataset_script_dir ../data_scripts \
  --dataset_cache_dir ../../negs_tutorial_cache \
  --num_beams 100 \
  --add_doc_num 6144 \
  --max_seq_length 64 \
  --max_docid_length 24 \
  --output_every_n_step 1000 \
  --save_every_n_epoch 2 \
  --operation testing \
  --use_docid_rank True
done


# URL

## prepare data


gen url data

get_docids


gen t5 train data

get docids pre

## training




#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3

python runT5.py \
--epoch 10 \
--per_gpu_batch_size  100 \
--learning_rate 1e-3 \
--save_path ../outputs/msmarco_url/stage1 \
--log_path ../logs/pretrain_url.msmarco.log \
--doc_file_path ../dataset/msmarco-data/msmarco-docs-sents-all.json \
--pretrain_model_path ../../../../huggingface/t5-base \
--docid_path ../dataset/encoded_docid/t5_url_all.txt \
--train_file_path ../dataset/msmarco-data/train_data_url/pretrain_data.json \
--test_file_path ../dataset/msmarco-data/train_data_url/eval_query2docid.json \
--dataset_script_dir ../data_scripts \
--dataset_cache_dir ../../negs_tutorial_cache \
--add_doc_num 6144 \
--max_seq_length 128 \
--max_docid_length 24 \
--use_origin_head False \
--output_every_n_step 5000 \
--save_every_n_epoch 2 \
--operation training

python runT5.py \
--epoch 20 \
--per_gpu_batch_size  100 \
--learning_rate 1e-3 \
--save_path ../outputs/msmarco_url/stage2 \
--log_path ../logs/searchtrain_url.msmarco.log \
--doc_file_path ../dataset/msmarco-data/msmarco-docs-sents-all.json \
--pretrain_model_path ../../../../huggingface/t5-base \
--docid_path ../dataset/encoded_docid/t5_url_all.txt \
--train_file_path ../dataset/msmarco-data/train_data_url/pseudo2docid.json \
--test_file_path ../dataset/msmarco-data/train_data_url/eval_query2docid.json \
--dataset_script_dir ../data_scripts \
--dataset_cache_dir ../../negs_tutorial_cache \
--add_doc_num 6144 \
--max_seq_length 128 \
--max_docid_length 24 \
--use_origin_head 0 \
--load_ckpt True \
--load_ckpt_path ../outputs/msmarco_url/stage1/model_9.pkl \
--output_every_n_step 5000 \
--save_every_n_epoch 4 \
--operation training

python runT5.py \
--epoch 10 \
--per_gpu_batch_size  100 \
--learning_rate 1e-3 \
--save_path ../outputs/msmarco_url/stage3 \
--log_path ../logs/finetune_url.msmarco.log \
--doc_file_path ../dataset/msmarco-data/msmarco-docs-sents-all.json \
--pretrain_model_path ../../../../huggingface/t5-base \
--docid_path ../dataset/encoded_docid/t5_url_all.txt \
--train_file_path ../dataset/msmarco-data/train_data_url/query2docid.json \
--test_file_path ../dataset/msmarco-data/train_data_url/eval_query2docid.json \
--dataset_script_dir ../data_scripts \
--dataset_cache_dir ../../negs_tutorial_cache \
--add_doc_num 6144 \
--max_seq_length 64 \
--max_docid_length 24 \
--use_origin_head 0 \
--load_ckpt True \
--load_ckpt_path ../outputs/msmarco_url/stage2/model_19.pkl \
--output_every_n_step 5000 \
--save_every_n_epoch 2 \
--operation training



#!/bin/bash
# test

python runT5.py \
--epoch 10 \
--per_gpu_batch_size 4 \
--learning_rate 1e-3 \
--save_path ../outputs/msmarco_url/stage3/model_9.pkl \
--log_path ../logs/result_url.log \
--doc_file_path ../dataset/msmarco-data/msmarco-docs-sents-all.json \
--pretrain_model_path ../../../../../huggingface/t5-base \
--docid_path ../dataset/encoded_docid/t5_url_all.txt \
--train_file_path ../dataset/msmarco-data/train_data_url/query2docid.json \
--test_file_path ../dataset/msmarco-data/train_data_url/eval_query2docid.json \
--dataset_script_dir ../data_scripts \
--dataset_cache_dir ../../negs_tutorial_cache \
--num_beams 100 \
--add_doc_num 6144 \
--max_seq_length 64 \
--max_docid_length 24 \
--output_every_n_step 1000 \
--save_every_n_epoch 2 \
--operation testing \
--use_docid_rank True \
--per_gpu_batch_size 2

#!/bin/bash

for i in {1..6}
do
  python runT5.py \
  --epoch 10 \
  --per_gpu_batch_size 4 \
  --learning_rate 1e-3 \
  --save_path ../outputs/msmarco_url/stage3/model_9.pkl \
  --log_path ../logs/result__url${i}.log \
  --doc_file_path ../dataset/msmarco-data/msmarco-docs-sents-all_${i}.json \
  --pretrain_model_path ../../../../../huggingface/t5-base \
  --docid_path ../dataset/encoded_docid/t5_url_all_${i}.txt \
  --train_file_path ../dataset/msmarco-data/train_data_url/query2docid.json \
  --test_file_path ../dataset/msmarco-data/train_data_url/eval_query2docid.json \
  --dataset_script_dir ../data_scripts \
  --dataset_cache_dir ../../negs_tutorial_cache \
  --num_beams 100 \
  --add_doc_num 6144 \
  --max_seq_length 64 \
  --max_docid_length 24 \
  --output_every_n_step 1000 \
  --save_every_n_epoch 2 \
  --operation testing \
  --use_docid_rank True \
  --per_gpu_batch_size 2
done