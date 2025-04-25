#TOKENIZERS_PARALLELISM=false python seal/search.py \
#  --topics_format dpr_qas_train --topics data/NQ/dataset/biencoder-nq-train.json \
#  --output_format dpr --output LTRGR/data/results/MINDER_NQ_train_top200.json \
#  --checkpoint data/ckpts/checkpoint_best.pt \
#  --jobs 10 --progress --device cuda:0 --batch_size 10 \
#  --beam 15 \
#  --decode_query stable \
#  --fm_index data/NQ/fm_index/corpus/corpus \
#  --include_keys \
#  --hits 100



for i in 0 1 2 3 4
do
  TOKENIZERS_PARALLELISM=false python seal/search.py \
    --topics_format dpr_qas_train --topics data/NQ/dataset/biencoder-nq-train_${i}.json \
    --output_format dpr --output LTRGR/data/results/MINDER_NQ_train_top100_${i}.json \
    --checkpoint data/ckpts/checkpoint_best.pt \
    --jobs 10 --progress --device cuda:0 --batch_size 10 \
    --beam 15 \
    --decode_query stable \
    --fm_index data/NQ/fm_index/corpus/corpus \
    --include_keys \
    --hits 100
done