import json
import os
import random

import numpy as np
import pandas as pd
import plotly
from matplotlib import pyplot as plt

from utils import get_project_root

path = "savedVM_v2/models/CPSC_BaselineWithMultiHeadAttention_uBCE_F1/tune_run_1"

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

df = df.sort_values("Validation F1")
print(df.shape)
print(df)

# Modify the values by a random number to generate more than one line in the plot
for col in df.columns:
    df[col] = df[col].apply(lambda x: x + round(random.uniform(-(df[col].max() - df[col].min()) / 40,
                                   (df[col].max() - df[col].min()) / 40), 3))
import plotly.express as px
import plotly.graph_objects as go

fig = px.parallel_coordinates(df, color="Validation F1", dimensions=df.columns,
                              color_continuous_scale=px.colors.sequential.matter)
fig.update_layout(
    font_family="serif",
)

df[["Dropout", "GRU Units", "Heads"]] = df[["Dropout", "GRU Units", "Heads"]].applymap(
    lambda x: x + np.random.rand() / 10000.0)

dimensions = [dict(label=col, values=df[col]) for col in df.columns]
dimensions[-1]["range"] = [0.82, 0.87]
dimensions[-1]["constraintrange"] = [0.85, 0.87]
fig2 = go.Figure(data=
go.Parcoords(
    line=dict(color=df['Validation F1'],
              colorscale='Matter',
              showscale=True,
              cmin=0, cmax=1),
    dimensions=dimensions
)
)
fig2.update_layout(
    font_family="serif",
    title="Hyperparameter Sturdy"
)

dst_path = os.path.join(get_project_root(), "playground", "plots", "mh_attention_with_FC_hyperparams.pdf")
plotly.io.write_image(fig, dst_path, format='pdf')
print("Done")
