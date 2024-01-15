#!/bin/bash

source venv/bin/activate

REL_PATH="savedVM/models/BaselineWithMultiHeadAttention_ParamStudy/0108_114757_ml_bs64_attention_type_v1_reduced_dim_entmax15"

# THE VALID SET VARIES FROM TUNE RUN TO TUNE RUN!!!!
#for dir in $(find $REL_PATH -mindepth 1 -maxdepth 1 -type d )
#do
#    echo "Evaluating $dir ..."
#    python test.py --resume "$dir/model_best.pth" --test_dir "data/CinC_CPSC/train/preprocessed/4ms/eq_len_60s/valid" --tune
#done

for dir in $(find $REL_PATH -mindepth 1 -maxdepth 1 -type d )
do
    echo "Evaluating $dir ..."
    python test.py --resume "$dir/model_best.pth" --test_dir "data/CinC_CPSC/test/preprocessed/4ms/eq_len_60s" --tune
done

