# 2020 MA Vanessa Borst

## Usage of tensorboard:
1. ssh -L 16006:127.0.0.1:6006 SE-GPUs
2. Start tensorboard in terminal: `tensorboard --logdir="savedVM/log/folder_to_tf_events/"`
3. On your local machine, go to http://127.0.0.1:16006 


## Early Stopping
The metric can e.g. be set to `val_macro_sk_f1` or `val_weighted_sk_f1`



## Copy files from train/test to cross_valid 
250 Hz

cp -a train/preprocessed/4ms/eq_len_10s/* cross_valid/250Hz/10s/
cp -a train/preprocessed/4ms/eq_len_15s/* cross_valid/250Hz/15s/
cp -a train/preprocessed/4ms/eq_len_30s/* cross_valid/250Hz/30s/
cp -a train/preprocessed/4ms/eq_len_60s/* cross_valid/250Hz/60s/
cp -a test/preprocessed/4ms/eq_len_10s/* cross_valid/250Hz/10s/
cp -a test/preprocessed/4ms/eq_len_15s/* cross_valid/250Hz/15s/
cp -a test/preprocessed/4ms/eq_len_30s/* cross_valid/250Hz/30s/
cp -a test/preprocessed/4ms/eq_len_60s/* cross_valid/250Hz/60s/


500 Hz

cp -a train/preprocessed/no_sampling/eq_len_10s/* cross_valid/500Hz/10s/
cp -a train/preprocessed/no_sampling/eq_len_15s/* cross_valid/500Hz/15s/
cp -a train/preprocessed/no_sampling/eq_len_30s/* cross_valid/500Hz/30s/
cp -a train/preprocessed/no_sampling/eq_len_60s/* cross_valid/500Hz/60s/
cp -a test/preprocessed/no_sampling/eq_len_10s/* cross_valid/500Hz/10s/
cp -a test/preprocessed/no_sampling/eq_len_15s/* cross_valid/500Hz/15s/
cp -a test/preprocessed/no_sampling/eq_len_30s/* cross_valid/500Hz/30s/
cp -a test/preprocessed/no_sampling/eq_len_60s/* cross_valid/500Hz/60s/


## Change to project and run python script
cd projects/2023-macro-paper-3.10/
source venv/bin/activate
python train_with_cv.py -c config_baseline_crossValid.json