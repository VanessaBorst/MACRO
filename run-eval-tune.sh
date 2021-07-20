#!/bin/bash

source venv/bin/activate

#for dir in $(find savedVM/models/CPSC_BaselineWithMultiHeadAttention/param_study_1 -mindepth 1 -maxdepth 1 -type d )
#do
#    echo "Evaluating $dir ..."
#    python3.8 test.py --resume "$dir/model_best.pth" --test_dir "data/CinC_CPSC/train/preprocessed/no_sampling/eq_len_72000/valid" --tune
#done
#
#for dir in $(find savedVM/models/CPSC_BaselineWithMultiHeadAttention/param_study_1 -mindepth 1 -maxdepth 1 -type d )
#do
#    echo "Evaluating $dir ..."
#    python3.8 test.py --resume "$dir/model_best.pth" --test_dir "data/CinC_CPSC/test/preprocessed/no_sampling/eq_len_72000" --tune
#done


#for dir in $(find savedVM/models/CPSC_BaselineWithSkips/tune_random_search -mindepth 1 -maxdepth 1 -type d )
#do
#    echo "Evaluating $dir ..."
#    python3.8 test.py --resume "$dir/model_best.pth" --test_dir "data/CinC_CPSC/train/preprocessed/no_sampling/eq_len_72000/valid" --tune
#done
#
#for dir in $(find savedVM/models/CPSC_BaselineWithSkips/tune_random_search -mindepth 1 -maxdepth 1 -type d )
#do
#    echo "Evaluating $dir ..."
#    python3.8 test.py --resume "$dir/model_best.pth" --test_dir "data/CinC_CPSC/test/preprocessed/no_sampling/eq_len_72000" --tune
#done

for dir in $(find savedVM/models/CPSC_BaselineWithMultiHeadAttention_uBCE_F1/tune_run_1 -mindepth 1 -maxdepth 1 -type d )
do
    echo "Evaluating $dir ..."
    python3.8 test.py --resume "$dir/model_best.pth" --test_dir "data/CinC_CPSC/train/preprocessed/no_sampling/eq_len_72000/valid" --tune
done

for dir in $(find savedVM/models/CPSC_BaselineWithMultiHeadAttention_uBCE_F1/tune_run_1 -mindepth 1 -maxdepth 1 -type d )
do
    echo "Evaluating $dir ..."
    python3.8 test.py --resume "$dir/model_best.pth" --test_dir "data/CinC_CPSC/test/preprocessed/no_sampling/eq_len_72000" --tune
done