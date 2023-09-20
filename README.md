# 2020 MA Vanessa Borst

## Usage of tensorboard:
1. ssh -L 16006:127.0.0.1:6006 SE-GPUs
2. Start tensorboard in terminal: `tensorboard --logdir="savedVM/log/folder_to_tf_events/"`
3. On your local machine, go to http://127.0.0.1:16006 


## Early Stopping
The metric can e.g. be set to `val_macro_sk_f1` or `val_weighted_sk_f1`