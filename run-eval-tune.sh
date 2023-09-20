#!/bin/bash

source venv/bin/activate

REL_PATH="savedVM/models/FinalModel_MACRO_MultiBranch_ParamStudy/0905_153928_ml_bs16_varyChannelsLighter_False"

for dir in $(find $REL_PATH -mindepth 1 -maxdepth 1 -type d )
do
    echo "Evaluating $dir ..."
    python test.py --resume "$dir/model_best.pth" --test_dir "data/CinC_CPSC/train/preprocessed/no_sampling/eq_len_72000/valid" --tune
done

for dir in $(find $REL_PATH -mindepth 1 -maxdepth 1 -type d )
do
    echo "Evaluating $dir ..."
    python test.py --resume "$dir/model_best.pth" --test_dir "data/CinC_CPSC/test/preprocessed/no_sampling/eq_len_72000" --tune
done

