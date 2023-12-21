#!/bin/bash

source venv/bin/activate

# REL_PATH="savedVM_v2/models/FinalModel/manual_runs/"
REL_PATH="savedVM/models/BaselineModel_SingleRun/"

for dir in $(find $REL_PATH -mindepth 1 -maxdepth 1 -type d )
do
    echo "Evaluating $dir ..."
    python test.py --resume "$dir/model_best.pth" --test_dir "data/CinC_CPSC/train/preprocessed/4ms/eq_len_60s/valid"
done

for dir in $(find $REL_PATH -mindepth 1 -maxdepth 1 -type d )
do
    echo "Evaluating $dir ..."
    python test.py --resume "$dir/model_best.pth" --test_dir "data/CinC_CPSC/test/preprocessed/4ms/eq_len_60s"
done