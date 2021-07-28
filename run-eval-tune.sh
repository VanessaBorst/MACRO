#!/bin/bash

source venv/bin/activate

REL_PATH="savedVM_v2/models/FinalModel/0727_142315_ml_bs64_experiment_3"


for dir in $(find $REL_PATH -mindepth 1 -maxdepth 1 -type d )
do
    echo "Evaluating $dir ..."
    python3.8 test.py --resume "$dir/model_best.pth" --test_dir "data/CinC_CPSC/train/preprocessed/no_sampling/eq_len_72000/valid" --tune
done

for dir in $(find $REL_PATH -mindepth 1 -maxdepth 1 -type d )
do
    echo "Evaluating $dir ..."
    python3.8 test.py --resume "$dir/model_best.pth" --test_dir "data/CinC_CPSC/test/preprocessed/no_sampling/eq_len_72000" --tune
done

