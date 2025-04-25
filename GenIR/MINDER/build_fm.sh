FILE_I=data/NQ/corpus.tsv
FILE_O=data/NQ/fm_index/corpus

python scripts/build_fm_index.py \
    $FILE_I $FILE_O \
    --hf_model ../../../huggingface/bart-large  \
    --jobs 40 --include_title


for i in {1..5}
do
    FILE_I=data/NQ/dataset/corpus_${i}.tsv
    FILE_O=data/NQ/fm_index/corpus_${i}
    python scripts/build_fm_index.py \
        $FILE_I $FILE_O \
        --hf_model ../../../huggingface/bart-large  \
        --jobs 40 --include_title
done







