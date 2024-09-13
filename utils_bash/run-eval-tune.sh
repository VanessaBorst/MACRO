#!/bin/bash

source venv/bin/activate

# # Adapt this path to the location of the saved models (trained with --tune flag and a fixed train-valid-test split)
# Call example (from the project source folder):  . utils_bash/run-eval-tune.sh
REL_PATH="savedVM/models/FinalModel_MACRO_ParamStudy_PTB_XL_Superdiag/0906_170745_ml_bs64_12gru-entmax15"

for dir in $(find $REL_PATH -mindepth 1 -maxdepth 1 -type d )
do
    echo "Evaluating $dir ..."
    python "test.py" --resume "$dir/model_best.pth" --test_dir "data/PTB_XL/superdiag_100/test" --tune
done

