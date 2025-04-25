# environment
pip install tensorboardX
https://ku.baidu-int.com/knowledge/HFVrC7hq1Q/pKzJfZczuc/f_BqiC72p7/oQebYTrn4KwUBp
apt install openjdk-11-jdk
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export PATH=$JAVA_HOME/bin:$PATH

https://ku.baidu-int.com/knowledge/HFVrC7hq1Q/pKzJfZczuc/YiW0s4TS7p/HwdhvkJnboj_5Q

# command



## prepare data
get train test.py

!bin/bash

DATASET=dataset/msmarco/stage1
for FILE in train_data test_data ; do

    python scripts/training/make_supervised_dpr_dataset.py \
$DATASET/$FILE.json $DATASET/$FILE \
--target title \
--mark_target \
--mark_silver \
--n_samples 3 \
--mode a
    
    python scripts/training/make_supervised_dpr_dataset.py \
$DATASET/$FILE.json $DATASET/$FILE \
--target span \
--mark_target \
--mark_silver \
--n_samples 10 \
--mode a
done


sh preprocess_fairseq.sh ../../dataset/msmarco/stage1 ../../../../../huggingface/bart.large

sh training_fairseq.sh ../../dataset/msmarco/stage1 ../../../../../huggingface/bart.large

