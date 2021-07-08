#!/bin/bash

source venv/bin/activate

for dir in $(find savedVM/models/CPSC_BaselineModel/lr_10e-4 -mindepth 1 -maxdepth 1 -type d )
do
    echo "Evaluating $dir ..."
    python3.8 test.py --resume "$dir/model_best.pth" --test_dir "data/CinC_CPSC/train/preprocessed/no_sampling/eq_len_72000/valid"
done

for dir in $(find savedVM/models/CPSC_BaselineModel/lr_10e-3 -mindepth 1 -maxdepth 1 -type d )
do
    echo "Evaluating $dir ..."
    python3.8 test.py --resume "$dir/model_best.pth" --test_dir "data/CinC_CPSC/train/preprocessed/no_sampling/eq_len_72000/valid"
done

for dir in $(find savedVM/models/CPSC_BaselineWithSkips -mindepth 1 -maxdepth 1 -type d )
do
    echo "Evaluating $dir ..."
    python3.8 test.py --resume "$dir/model_best.pth" --test_dir "data/CinC_CPSC/train/preprocessed/no_sampling/eq_len_72000/valid"
done