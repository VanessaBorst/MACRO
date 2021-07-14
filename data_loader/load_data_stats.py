import pickle
import os

import pandas as pd

path = "data_loader/JuliaVM"

df_summary = pd.DataFrame()
# ML
with open(os.path.join(path, "class_freq_ml_train.p"), 'rb') as file:
    df_ml_train = pickle.load(file)
    df_summary = df_summary.append(pd.Series(df_ml_train["Class_freq"], name="ml_train"))
with open(os.path.join(path, "class_freq_ml_valid.p"), 'rb') as file:
    df_ml_valid = pickle.load(file)
    df_summary = df_summary.append(pd.Series(df_ml_valid["Class_freq"], name="ml_valid"))
with open(os.path.join(path, "class_freq_ml_test.p"), 'rb') as file:
    df_ml_test = pickle.load(file)
    df_summary = df_summary.append(pd.Series(df_ml_test["Class_freq"], name="ml_test"))
# SL
with open(os.path.join(path, "class_freq_sl_train.p"), 'rb') as file:
    df_sl_train = pickle.load(file)
    df_summary = df_summary.append(pd.Series(df_sl_train["Class_freq"], name="sl_train"))
with open(os.path.join(path, "class_freq_sl_valid.p"), 'rb') as file:
    df_sl_valid = pickle.load(file)
    df_summary = df_summary.append(pd.Series(df_sl_valid["Class_freq"], name="sl_valid"))
with open(os.path.join(path, "class_freq_sl_test.p"), 'rb') as file:
    df_sl_test = pickle.load(file)
    df_summary = df_summary.append(pd.Series(df_sl_test["Class_freq"], name="sl_test"))

df_summary['Total'] = df_summary.sum(axis=1)
df_summary=df_summary.astype(int)

with open(os.path.join(path, 'label_stats.tex'), 'w') as file:
    df_summary.to_latex(buf=file, index=True, bold_rows=True)
print(df_summary.to_latex())
print("Finished")
