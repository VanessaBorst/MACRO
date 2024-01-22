#!/bin/bash

source venv/bin/activate

REL_PATH="savedVM/models/BaselineWithMultiHeadAttention_ParamStudy/0117_145626_ml_bs64_attention_type_v2_withFC"

# Before the bug fix in Jan 24, the valid set varied from tune run to tune run
for dir in $(find $REL_PATH -mindepth 1 -maxdepth 1 -type d )
do
    echo "Evaluating $dir ..."
    python test.py --resume "$dir/model_best.pth" --test_dir "data/CinC_CPSC/train/preprocessed/4ms/eq_len_60s/valid" --tune
done

for dir in $(find $REL_PATH -mindepth 1 -maxdepth 1 -type d )
do
    echo "Evaluating $dir ..."
    python test.py --resume "$dir/model_best.pth" --test_dir "data/CinC_CPSC/test/preprocessed/4ms/eq_len_60s" --tune
done

