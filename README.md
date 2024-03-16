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


## Stuck GPU usage
`ssh -L 16006:127.0.0.1:6006 SE-GPUs-admin`
Dann dort folgende Befehle ausführen
````
 1851  ps -u vab30xh
 1852  pkill -9 -u vab30xh python
 1853  ps -u vab30xh
 1854  nvidia-smi
````

## Find out users on GPU server
(echo "GPU_ID PID MEM% UTIL% UID APP" ; for GPU in 0 1 2 3 ; do for PID in $( nvidia-smi -q --id=${GPU} --display=PIDS | awk '/Process ID/{print $NF}') ; do echo -n "${GPU} ${PID} " ; nvidia-smi -q --id=${GPU} --display=UTILIZATION | grep -A4 -E '^[[:space:]]*Utilization' | awk 'NR=0{gut=0 ;mut=0} $1=="Gpu"{gut=$3} $1=="Memory"{mut=$3} END{printf "%s %s ",mut,gut}' ; ps -up ${PID} | gawk 'NR-1 {print $1,$NF}' ; done ; done) | column -t



## TODOs Refactor
- Check scripts in utils_bash (relative paths)
- Check scripts in visualization (relative paths)
- Unneeded params in config such as stratified k-fold
- Folgende Params wurden entfernt, auch aus Kommentaren und co (zb Train/Paramstudy und summarize_tune_runs) löschen 
  - discard_FC_before_MH (_noFC, _withFC)
- Train, Test und Summarize Tune Runs aufräumen und an neue Configs anpassen