#FILE_I=dataset/msmarco/stage1/corpus.tsv
#FILE_O=dataset/msmarco/fm_index/fm_index/corpus.fm_index
#
#python scripts/build_fm_index.py \
#    $FILE_I $FILE_O \
#    --hf_model ../../../huggingface/bart-large  \
#    --jobs 40 --include_title


for i in 1 2 3 4 5
do
    FILE_I=dataset/msmarco/stage1/corpus_$i.tsv
    FILE_O=dataset/msmarco/fm_index/fm_index_$1/corpus.fm_index
    python scripts/build_fm_index.py \
        $FILE_I $FILE_O \
        --hf_model ../../../huggingface/bart-large  \
        --jobs 40 --include_title
done