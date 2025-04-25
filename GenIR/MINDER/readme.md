apt install openjdk-11-jdk
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export PATH=$JAVA_HOME/bin:$PATH



python scripts/training/make_supervised_dpr_dataset.py \
    data/NQ/biencoder-nq-dev.json data/training_data/NQ_title_body_query_generated/dev \
    --target title \
    --mark_target \
    --mark_silver \
    --n_samples 3 \
    --mode a \
    --min_score 0.0 \
    --min_score_gold 0.0

python scripts/training/make_supervised_dpr_dataset.py \
    data/NQ/biencoder-nq-dev.json data/training_data/NQ_title_body_query_generated/dev \
    --target span \
    --mark_target \
    --mark_silver \
    --n_samples 10 \
    --mode a \
    --min_score 0.0 \
    --min_score_gold 0.0

python scripts/training/make_supervised_dpr_dataset.py \
    data/NQ/biencoder-nq-dev.json data/training_data/NQ_title_body_query_generated/dev \
    --target query \
    --pid2query data/pseudo_queries/pid2query_nq.pkl \
    --mark_target \
    --mark_silver \
    --n_samples 5 \
    --mode a \
    --min_score 0.0 \
    --min_score_gold 0.0

# train
python scripts/training/make_supervised_dpr_dataset.py \
    data/NQ/biencoder-nq-train.json data/training_data/NQ_title_body_query_generated/train \
    --target title \
    --mark_target \
    --mark_silver \
    --n_samples 3 \
    --mode a \
    --min_score 0.0 \
    --min_score_gold 0.0

python scripts/training/make_supervised_dpr_dataset.py \
    data/NQ/biencoder-nq-train.json data/training_data/NQ_title_body_query_generated/train \
    --target span \
    --mark_target \
    --mark_silver \
    --n_samples 10 \
    --mode a \
    --min_score 0.0 \
    --min_score_gold 0.0

python scripts/training/make_supervised_dpr_dataset.py \
    data/NQ/biencoder-nq-train.json data/training_data/NQ_title_body_query_generated/train \
    --target query \
    --pid2query data/pseudo_queries/pid2query_nq.pkl \
    --mark_target \
    --mark_silver \
    --n_samples 5 \
    --mode a \
    --min_score 0.0 \
    --min_score_gold 0.0



build fm index
retrieval
get score






