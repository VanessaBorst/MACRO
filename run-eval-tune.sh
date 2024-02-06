#!/bin/bash

source venv/bin/activate

REL_PATH="savedVM/models/Multibranch_MACRO_ParamStudy/0130_135327_ml_bs64_noConvRedBlock"

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

