TOKENIZERS_PARALLELISM=false python -m seal.search \
    --topics_format dpr --topics dataset/msmarco/stage1/test_data.json \
    --output_format dpr --output dataset/msmarco/output/output_0_1.json \
    --checkpoint scripts/training/checkpoints/checkpoint_best.pt \
    --fm_index dataset/msmarco/fm_index/fm_index/corpus.fm_index \
    --jobs 20 --progress --device cuda:0 --batch_size 20 \
    --beam 15


for i in {1..5}; do
    TOKENIZERS_PARALLELISM=false python -m seal.search \
        --topics_format dpr --topics dataset/msmarco/stage1/test_data.json \
        --output_format dpr --output dataset/msmarco/output/output_0_${i}.json \
        --checkpoint scripts/training/checkpoints/checkpoint_best.pt \
        --fm_index dataset/msmarco/fm_index/fm_index_${i}/corpus.fm_index \
        --jobs 20 --progress --device cuda:0 --batch_size 20 \
        --beam 15
done

for i in {1..5}; do
    TOKENIZERS_PARALLELISM=false python -m seal.search \
        --topics_format dpr --topics dataset/msmarco/stage1/test_data_${i}.json \
        --output_format dpr --output dataset/msmarco/output/output_${i}_${i}.json \
        --checkpoint scripts/training/checkpoints/checkpoint_best.pt \
        --fm_index dataset/msmarco/fm_index/fm_index_${i}/corpus.fm_index \
        --jobs 20 --progress --device cuda:0 --batch_size 20 \
        --beam 15
done