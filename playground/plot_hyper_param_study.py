import copy
import json
import os
import random

import numpy as np
import pandas as pd
import plotly
from matplotlib import pyplot as plt

from utils import get_project_root

# 0810_215444_ml_bs64_rerun_100821_withFC
# 0810_215735_ml_bs64_rerun_100821_noFC
# path = "savedVM_v2/models/CPSC_BaselineWithMultiHeadAttention_uBCE_F1/0810_215735_ml_bs64_rerun_100821_noFC"
# select_TOP_x = 5 #6
# lower_constraint_range_f1 = 0.8445  # rounded 0.845
# diff = "0.5"
# file_name="experiment_2_2_rerun_without_FC"

# tune_run_2_additional_FC_discarded
# tune_run_1
path = "savedVM_v2/models/CPSC_BaselineWithMultiHeadAttention_uBCE_F1/tune_run_2_additional_FC_discarded"
select_TOP_x = 5
lower_constraint_range_f1 = 0.8475  # rounded to next higher number with 3 decimals
diff = "1.8"
file_name="experiment_2_2_without_FC_sqrtT"

df = pd.DataFrame(columns=["Dropout", "GRU Units", "Heads", "Validation F1"])
for run in os.listdir(path):
    path2 = os.path.join(path, run)
    if os.path.isdir(path2):
        with open(os.path.join(path2, "params.json"), 'r') as file:
            params = json.loads(file.read())
        try:
            progress = pd.read_csv(os.path.join(path2, "progress.csv"))
            df = df.append({
                "Dropout": params["dropout_attention"],
                "Heads": params["heads"],
                "GRU Units": params["gru_units"],
                "Validation F1": progress.val_weighted_sk_f1.max()  # progress.iloc[-1]["val_weighted_sk_f1"]
            }, ignore_index=True)
        except:
            pass

df = df.sort_values("Validation F1", ascending=False)
if file_name == "experiment_3_2_no_FC":
    df.drop(df.tail(1).index,inplace=True)
print(df.shape)
print(df)

# Modify the values by a random number to generate more than one line in the plot
for col in df.columns[:-1]:     # Do not transform the final score!
    df[col] = df[col].apply(lambda x: x + round(random.uniform(-(df[col].max() - df[col].min()) / 100,
                                   (df[col].max() - df[col].min()) / 100), 3))
import plotly.express as px
import plotly.graph_objects as go


# Uncomment for tune_rune_1 (With FC)
fig = px.parallel_coordinates(df, color="Validation F1", dimensions=df.columns,
                              color_continuous_scale=px.colors.sequential.matter)
fig.update_layout(
    font_family="serif",
)

dimensions_TOP5 = []
# Add Dimensions
dimensions_TOP5.append(dict(
    label="Dropout",
    values=df["Dropout"].values,
    tickvals=[0.2, 0.3, 0.4],
    range=[df["Dropout"].min(), df["Dropout"].max()]
))
dimensions_TOP5.append(dict(
    label="GRU Units",
    values=df["GRU Units"].values,
    tickvals=[12, 24, 32],
    range=[df["GRU Units"].min(), df["GRU Units"].max()]
))
dimensions_1pp = copy.deepcopy(dimensions_TOP5)

dimensions_TOP5.append(dict(
    label="Validation F1",
    values=df["Validation F1"].values,
    range=[df["Validation F1"].min(), df["Validation F1"].max()],
    # tickvals=[0.82, 0.83, 0.84, 0.85],
    constraintrange=[df["Validation F1"].sort_values(ascending=False).values[select_TOP_x-1], df["Validation F1"].max()]
))
dimensions_1pp.append(dict(
    label="Validation F1",
    values=df["Validation F1"].values,
    range=[df["Validation F1"].min(), df["Validation F1"].max()],
    # tickvals=[0.82, 0.83, 0.84, 0.85],
    constraintrange=[lower_constraint_range_f1, df["Validation F1"].max()]
))

dimensions_TOP5.append(dict(
    label="Heads",
    values=df["Heads"].values,
    tickvals=[3, 5, 8, 16, 32],
    range=[df["Heads"].min(), df["Heads"].max()]
))
dimensions_1pp.append(dict(
    label="Heads",
    values=df["Heads"].values,
    tickvals=[3, 5, 8, 16, 32],
    range=[df["Heads"].min(), df["Heads"].max()]
))


fig_top5 = go.Figure(data=
    go.Parcoords(
        line=dict(color=df['Validation F1'],
                  colorscale='Oryel',
                  showscale=False,
                  cmin=df["Validation F1"].min(), cmax=df["Validation F1"].max()),
        dimensions=dimensions_TOP5
    ),

)
fig_top5.update_layout(
    font_family="serif",
    font_size=18,
    #title="Hyperparameter Study"
    autosize=True
)

fig_1pp = go.Figure(data=
    go.Parcoords(
        line=dict(color=df['Validation F1'],
                  colorscale='Oryel',
                  showscale=False,
                  cmin=df["Validation F1"].min(), cmax=df["Validation F1"].max()),
        dimensions=dimensions_1pp
    ),

)
fig_1pp.update_layout(
    font_family="serif",
    font_size=18,
    #title="Hyperparameter Study"
    autosize=True
)

dst_path = os.path.join(path, "param_study_" + file_name + "_top5.pdf")
fig_top5.write_image(dst_path, format='pdf')

dst_path = os.path.join(path, "param_study_" + file_name + "_" + diff + "pp.pdf")
fig_1pp.write_image(dst_path, format='pdf')

print("Done")
