TOKENIZERS_PARALLELISM=false  python seal/MINDER_learn_to_rank_v1.4.py \
  --checkpoint data/ckpts/checkpoint_best.pt \
  --fm_index data/NQ/fm_index/corpus/corpus \
  --train_file LTRGR/data/results/MINDER_NQ_train_top100.json \
  --do_fm_index True \
  --per_gpu_train_batch_size 8 \
  --output_dir ./release_test \
  --rescore_batch_size 70 \
  --num_train_epochs 3 \
  --factor_of_generation_loss 1000 \
  --rank_margin 300 \
  --shuffle_positives True \
  --shuffle_negatives True \
  --decode_query stable \
  --pid2query data/pseudo_queries/pid2query_nq.pkl