#!/bin/bash

source venv/bin/activate

# Adapt this path to the location of the saved models (trained as single run with a fixed train-valid-test split)
REL_PATH="../savedVM/models/BaselineModel_SingleRun/"

for dir in $(find $REL_PATH -mindepth 1 -maxdepth 1 -type d )
do
    echo "Evaluating $dir ..."
    python "../test.py" --resume "$dir/model_best.pth" --test_dir "../data/CinC_CPSC/test/preprocessed/250Hz/60s"
done