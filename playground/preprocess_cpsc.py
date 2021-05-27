import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import wfdb
import os
import uuid
import numpy as np
from datetime import timedelta
from scipy.io import loadmat
import pickle as pk
from multiprocessing import Pool


def _compute(sampling, file, path):
    values = loadmat(os.path.join(path, file))
    # Reads the values into a dataframe, transposes the matrix (one column per lead) and sorts the dataframe
    # by the column labels
    df = pd.DataFrame(values["val"]).T.sort_index(axis=1)

    # Sets the column labels to the lead names retrieved from the header file
    header = wfdb.rdheader(os.path.join(path, file))
    df.columns = header.sig_name

    # Reads the meta data contained in the header into a dictionary; example (A1655):
    # {'age': '46', 'sex': 'Male', 'dx': '426783006', 'rx': 'Unknown', 'hx': 'Unknown', 'sx': 'Unknown'}
    meta = {key.lower(): val for key, val in (h.split(": ") for h in header.comments)}

    # A timedelta object represents a duration, the difference between two dates or times
    # ==> An list of timedeltas is passed to the TimedeltaIndex method which uses this to construct the index with
    # After this step, the row labels are the TimedeltaIndices
    df.index = pd.TimedeltaIndex([timedelta(seconds=i / 500) for i in df.index], unit="ms")

    # Downsampling
    # resample() is a time-based groupby, followed by a reduction method on each of its groups.
    # Groups together the values contained in time-spans of "sampling" (offset string), e.g. within "20ms" intervals
    # and takes the mean to aggregate them to a single value
    # Example:
    # 500 Hz -> 6000 Samples -> 12 sec record
    # delta_t between two timestamps: 2ms for 500 Hz
    # For sampling="20ms" + 500Hz:  20/2=10 values are merged   -> 6000/10= 600 Samples for 12 sec   -> 50 Hz
    # For sampling="4ms" + 500Hz:   4/2=2 values are merged     -> 600/2= 3000 Samples for 12 sec    -> 250 Hz
    df = df.resample(sampling).mean()
    df.index = np.arange(0, df.shape[0])    # return to numbers as row index (0-#samples)

    # Normalize with mean normalization
    # df.mean() and df.std() operate on all columns, i.e. leads, separately
    # df - df.mean(): For each col, the mean of the col is subtracted from each element in the respective col
    # df = (df - df.mean()) / df.std()

    # Vanessa: Min-Max-Normalization it would be the following (Pandas automatically applies colomn-wise function):
    # df=(df-df.min())/(df.max()-df.min())

    pk.dump((df, meta), open(f"data_dir/cpsc/{sampling}/{file}.pk", "wb"))


def read(sampling, path):  # 4ms = 250Hz
    # for file in [f.split(".")[0] for f in os.listdir(path) if f.endswith("mat")]:
    #     _compute(sampling, file, path)

    with Pool(6) as pool:
        for file in [f.split(".")[0] for f in os.listdir(path) if f.endswith("mat")]:
            pool.apply_async(_compute, (sampling, file, path), error_callback=lambda x: print(x))
        pool.close()
        pool.join()


def run(sampling="20ms", path="data_dir/CPSC_dataset"):
    if not os.path.exists(f"data_dir/cpsc/{sampling}"):
        os.makedirs(f"data_dir/cpsc/{sampling}")

    read(sampling, path)


def min_max_scaling(path):
    scaler = MinMaxScaler()

    g_max = pd.DataFrame()

    invalid_files = []

    for file in os.listdir(path):
        if os.path.isdir(os.path.join(path, file)):
            continue

        df, meta = pk.load(open(os.path.join(path, file), "rb"))

        g_max = g_max.append(df.max(), ignore_index=True)

    g_max_threshold = g_max.median() + (g_max.std() * 2.5)

    file_list = []

    for file in os.listdir(path):
        if os.path.isdir(os.path.join(path, file)):
            continue
        df, meta = pk.load(open(os.path.join(path, file), "rb"))

        valid = True
        for col in df.columns:
            if df.loc[:, col].max() > g_max_threshold.loc[col] \
                    or abs(df.loc[:, col].min()) > g_max_threshold.loc[col]:
                valid = False
                invalid_files.append(file)

                # df.loc[:, col].plot()
                # plt.show()

                break

        if valid:
            file_list.append(file)

            scaler = scaler.partial_fit(df)

    if not os.path.exists(os.path.join(path, "minmax")):
        os.makedirs(os.path.join(path, "minmax"))

    for file in file_list:
        df, meta = pk.load(open(os.path.join(path, file), "rb"))

        df.loc[:, :] = scaler.transform(df)
        pk.dump((df, meta), open(os.path.join(path, "minmax", file), "wb"))

    print(f"Valid: {len(file_list)} Invalid: {len(invalid_files)}")


def normalize(path):
    if not os.path.exists(os.path.join(path, "normalized")):
        os.makedirs(os.path.join(path, "normalized"))

    for file in os.listdir(path):
        if ".pk" not in file:
            continue
        df, meta = pk.load(open(os.path.join(path, file), "rb"))

        df = (df - df.mean()) / df.std()
        pk.dump((df, meta), open(os.path.join(path, "normalized", file), "wb"))


def show(path):
    for file in os.listdir(path):
        if file.endswith(".pk"):
            df, meta = pk.load(open(os.path.join(path, file), "rb"))
            df.loc[:, "I"].plot()
            plt.show()
            pass


def clean_meta(path):
    classes = pd.read_csv("data/dx_classes.csv").set_index("SNOMED CT Code")

    metas = pd.DataFrame(columns=classes.index)

    for file in os.listdir(path):
        if file.endswith(".pk"):
            p = os.path.join(path, file)
            df, meta = pk.load(open(p, "rb"))

            dx = meta["dx"].split(",")
            for d in dx:
                metas.loc[p, d] = 1

    metas = metas.dropna(axis=1, how="all")
    metas.columns = list(range(len(metas.columns)))

    for p, classes in metas.iterrows():
        df, meta = pk.load(open(str(p), "rb"))
        meta["classes_encoded"] = classes.replace(np.nan, 0)
        meta["classes"] = classes.dropna().keys().to_list()
        pk.dump((df, meta), open(str(p), "wb"))


if __name__ == "__main__":
    path = "data_dir/CPSC_dataset"
    run("20ms",path)

#  path = "data_dir/cpsc/50ms/"
#
# # show(path + "minmax")
#  min_max_scaling(path=path)
#  clean_meta(path + "minmax")
#  #normalize(path+"50ms/")
