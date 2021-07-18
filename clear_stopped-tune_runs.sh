#!/bin/bash

source venv/bin/activate

REL_PATH="savedVM/models/CPSC_BaselineWithSkips/0714_222352_ml_bs64_weightedBCE"
REL_PATH_LOGS="savedVM/log/CPSC_BaselineWithSkips/0714_222352_ml_bs64_weightedBCE"
CHECKPOINT_NAME="checkpoint_000005"

#REL_PATH="savedVM/models/CPSC_BaselineWithSkips/tune_random_search"
#REL_PATH_LOGS="savedVM/log/CPSC_BaselineWithSkips/tune_random_search"
#CHECKPOINT_NAME="checkpoint-epoch5.pth"

COUNTER_MODELS_DELETE=0
COUNTER_MODELS_KEEP=0
keep_models_folders=()
# Delete all models that were not trained for at least 5 epochs
for dir in $(find $REL_PATH -mindepth 1 -maxdepth 1 -type d )
do
    if [ -e $dir/$CHECKPOINT_NAME ]; then
        echo "Fifth checkpoint exists in $dir"
        keep_models_folders+=($dir)
        COUNTER_MODELS_KEEP=$(( COUNTER_MODELS_KEEP + 1 ))
    else
        echo "Fifth checkpoint does NOT exists in $dir"
        #rm -r "$dir"
        echo "Directory $dir deleted"
        COUNTER_MODELS_DELETE=$(( COUNTER_MODELS_DELETE + 1 ))
    fi
done

keep_logs_folder=()
for i in "${keep_models_folders[@]}"
do
   printf 'Keeping %s\n' "$i"
   keep_logs_folder+=(${i//models/log})
done

for i in "${keep_logs_folder[@]}"
do
   printf 'Keeping %s\n' "$i"
done

COUNTER_MODELS_DELETE_LOG=0
COUNTER_MODELS_KEEP_LOG=0

for dir in $(find $REL_PATH_LOGS -mindepth 1 -maxdepth 1 -type d )
do
    if [[ " ${keep_logs_folder[*]} " =~ $dir ]]; then
      # whatever you want to do when array contains value
      echo "Keep $dir"
      COUNTER_MODELS_KEEP_LOG=$(( COUNTER_MODELS_KEEP_LOG + 1 ))
    else
      echo "Delete $dir"
      # rm -r "$dir"
      COUNTER_MODELS_DELETE_LOG=$(( COUNTER_MODELS_DELETE_LOG + 1 ))
    fi
done
echo "Deleted $COUNTER_MODELS_DELETE model folders"
echo "Deleted $COUNTER_MODELS_DELETE_LOG log folders"
echo "Kept $COUNTER_MODELS_KEEP model folders"
echo "Kept $COUNTER_MODELS_KEEP_LOG log folders"
