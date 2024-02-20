
import os
import pickle as pk

import pandas as pd
import wfdb
from matplotlib import pyplot as plt
from scipy.io import loadmat

from utils import get_project_root

# src_path = os.path.join(get_project_root(), "data/CinC_CPSC/cross_valid/500Hz/10s")
#
# for file in os.listdir(src_path):
#     if file.endswith(".pk"):
#         p = os.path.join(src_path, file)
#         df, meta = pk.load(open(p, "rb"))
#         df.loc[:, "I"].plot()
#         plt.show()


src_path = os.path.join(get_project_root(), "data/CinC_CPSC/train/raw")

# Problematic files in train set:
# A0718: Lead 12 is missing
# A6837: Lead 12 is missing
# A3263:
# A3736
# A5556
# A5421
# A2758
# A3545 (!)
# A5936
# A4591
# A4181
# A3049 (!)
# A1798
# A4151
# A2905
################## Begin Test ##################
# A3329
# A6316
# A4680



values = loadmat(os.path.join(src_path,  "A3545.mat"))
# Reads the values into a dataframe, transposes the matrix (one column per lead) and sorts the dataframe
# by the column labels
df = pd.DataFrame(values["val"]).T.sort_index(axis=1)

# Sets the column labels to the lead names retrieved from the header file
header = wfdb.rdheader(os.path.join(src_path, "A3736"))
df.columns = header.sig_name

# Reads the meta data contained in the header into a dictionary; example (A1655):
# {'age': '46', 'sex': 'Male', 'dx': '426783006', 'rx': 'Unknown', 'hx': 'Unknown', 'sx': 'Unknown'}
meta = {key.lower(): val for key, val in (h.split(": ") for h in header.comments)}
