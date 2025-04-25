TOKENIZERS_PARALLELISM=false python seal/search.py \
--topics_format dpr_qas --topics data/NQ/test/nq-test.qa.csv  \
--output_format dpr --output data/NQ/output/test.json \
--checkpoint data/ckpts/checkpoint_best.pt \
--jobs 10 --progress --device cuda:0 --batch_size 20 \
--beam 15 \
--decode_query stable \
--fm_index data/NQ/fm_index/corpus/corpus


for i in {1..5}
do
    TOKENIZERS_PARALLELISM=false python seal/search.py \
    --topics_format dpr_qas --topics data/NQ/test/nq-test-${i}.qa.csv  \
    --output_format dpr --output data/NQ/output/new_test_${i}.json \
    --checkpoint data/ckpts/checkpoint_best.pt \
    --jobs 10 --progress --device cuda:0 --batch_size 20 \
    --beam 15 \
    --decode_query stable \
    --fm_index data/NQ/fm_index/corpus_${i}/corpus_${i}
done


for i in {1..5}
do
    TOKENIZERS_PARALLELISM=false python seal/search.py \
    --topics_format dpr_qas --topics data/NQ/test/nq-test.qa.csv  \
    --output_format dpr --output data/NQ/output/test_${i}.json \
    --checkpoint data/ckpts/checkpoint_best.pt \
    --jobs 10 --progress --device cuda:0 --batch_size 20 \
    --beam 15 \
    --decode_query stable \
    --fm_index data/NQ/fm_index/corpus_${i}/corpus_${i}
done