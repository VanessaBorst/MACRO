
import os
import pickle as pk

from matplotlib import pyplot as plt


from utils import get_project_root

src_path = os.path.join(get_project_root(), "data/CinC_CPSC/cross_valid/500Hz/10s")

for file in os.listdir(src_path):
    if file.endswith(".pk"):
        p = os.path.join(src_path, file)
        df, meta = pk.load(open(p, "rb"))
        df.loc[:, "I"].plot()
        plt.show()
