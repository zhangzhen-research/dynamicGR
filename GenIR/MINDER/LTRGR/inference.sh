#TOKENIZERS_PARALLELISM=false python -m seal.search \
#  --topics_format dpr_qas --topics data/NQ/test/nq-test.qa.csv \
#  --output_format dpr --output data/NQ/LTRGR/output.json \
#  --checkpoint release_test/checkpoint-3751model.pt \
#  --jobs 5 --progress --device cuda:0 --batch_size 10 --beam 15 \
#  --decode_query stable --fm_index data/NQ/fm_index/corpus/corpus --dont_fairseq_checkpoint


for i in 1 2 3 4 5
do
  TOKENIZERS_PARALLELISM=false python -m seal.search \
    --topics_format dpr_qas --topics data/NQ/test/nq-test-${i}.qa.csv \
    --output_format dpr --output data/NQ/LTRGR/new-output-${i}.json \
    --checkpoint release_test/checkpoint-3751model.pt \
    --jobs 5 --progress --device cuda:0 --batch_size 10 --beam 15 \
    --decode_query stable --fm_index data/NQ/fm_index/corpus_${i}/corpus_${i} --dont_fairseq_checkpoint
done

for i in 1 2 3 4 5
do
  TOKENIZERS_PARALLELISM=false python -m seal.search \
    --topics_format dpr_qas --topics data/NQ/test/nq-test.qa.csv \
    --output_format dpr --output data/NQ/LTRGR/output-${i}.json \
    --checkpoint release_test/checkpoint-3751model.pt \
    --jobs 5 --progress --device cuda:0 --batch_size 10 --beam 15 \
    --decode_query stable --fm_index data/NQ/fm_index/corpus_${i}/corpus_${i} --dont_fairseq_checkpoint
done




