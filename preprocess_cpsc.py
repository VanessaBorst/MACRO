import csv
import shutil

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import wfdb
import os
import uuid
import numpy as np
from datetime import timedelta
from scipy.io import loadmat
import pickle as pk
from multiprocessing import Pool
from bidict import bidict


def _parse_and_downsample_record(src_path, file, target_path, sampling):
    """
    The method is called for each record separately
    It preprocess the record and stores the result as pickle dump
    @param sampling: DateOffset, Timedelta or str -> offset string or object representing target conversion.
    """
    values = loadmat(os.path.join(src_path, file))
    # Reads the values into a dataframe, transposes the matrix (one column per lead) and sorts the dataframe
    # by the column labels
    df = pd.DataFrame(values["val"]).T.sort_index(axis=1)

    # Sets the column labels to the lead names retrieved from the header file
    header = wfdb.rdheader(os.path.join(src_path, file))
    df.columns = header.sig_name

    # Reads the meta data contained in the header into a dictionary; example (A1655):
    # {'age': '46', 'sex': 'Male', 'dx': '426783006', 'rx': 'Unknown', 'hx': 'Unknown', 'sx': 'Unknown'}
    meta = {key.lower(): val for key, val in (h.split(": ") for h in header.comments)}

    if sampling is not None:
        """Downsampling"""
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
        df.index = np.arange(0, df.shape[0])  # return to numbers as row index (0-#samples)

    pk.dump((df, meta), open(f"{target_path}/{file}.pk", "wb"))


def _read_records(src_path, target_path, sampling):  # 4ms = 250Hz
    with Pool(6) as pool:
        for file in [f.split(".")[0] for f in os.listdir(src_path) if f.endswith("mat")]:
            pool.apply_async(_parse_and_downsample_record, (src_path, file, target_path, sampling),
                             error_callback=lambda x: print(x))
        pool.close()
        pool.join()
    print("Basic preprocessing finished for " + src_path + ".")


def split_train_test(src_path, dest_path, test_ratio=0.2):
    """
    Splits the given data into train and test with the given ratio
    Before, multi-labeled
    """
    file_names = []
    for file in [f.split(".")[0] for f in sorted(os.listdir(src_path)) if f.endswith("mat")]:
        # header = wfdb.rdheader(os.path.join(src_path, file))
        # meta = {key.lower(): val for key, val in (h.split(": ") for h in header.comments)}
        # labels = meta['dx'].split(',')
        file_names.append(file)

    train_files, test_files = train_test_split(file_names, test_size=test_ratio, random_state=42, shuffle=True)

    # Copy the train and validation files into a dedicated folder
    dest_path_train = os.path.join(dest_path, "train/")
    if not os.path.exists(dest_path_train):
        os.makedirs(dest_path_train)
    for file_name in train_files:
        for file_ext in [".mat", ".hea"]:
            full_file_name = os.path.join(src_path, file_name + file_ext)
            if os.path.isfile(full_file_name):
                shutil.copy(full_file_name, dest_path_train)

    # Copy the test files into a dedicated folder
    dest_path_test = os.path.join(dest_path, "test/")
    if not os.path.exists(dest_path_test):
        os.makedirs(dest_path_test)
    for file_name in test_files:
        for file_ext in [".mat", ".hea"]:
            full_file_name = os.path.join(src_path, file_name + file_ext)
            if os.path.isfile(full_file_name):
                shutil.copy(full_file_name, dest_path_test)

    print("Split finished.")


def run_basic_preprocessing(src_path, target_path, sampling=None):
    full_target_path = f"{target_path}{sampling}" if sampling is not None else f"{target_path}no_sampling"
    if not os.path.exists(full_target_path):
        os.makedirs(full_target_path)

    _read_records(src_path, full_target_path, sampling)


def clean_meta(path):
    """
        This method appends the meta information for each .pk record under the given path
        To this end, two additional entries are added to the meta dictionary of the record
        1) meta["classes_one_hot"]: Pandas Series containing a 1 if the class applies to the record and a 0 otherwise
        2) meta["classes_encoded"]: List of integers encoding the classes that apply to the record
                            (the integers are in the range between 0 and N-1, with N being the number of classes
                            existing amongst the record under the given path)

        However, before that, the order of multi-labeled records is changed to match the one of the orig CPSC dataset
        For this purpose, the already existing meta["dx"] information is updated if needed to adapt the order

        ==> The method only operates on the meta information and keeps the actual time series data unchanged
    """
    # Read in the REFERENCES.csv provided by the official CPSC
    cpsc_labels = pd.read_csv("info/csv/REFERENCE_cpsc.csv").set_index("Recording")

    # Get the mapping between the classes and push them in a dict
    mapping_df= pd.read_csv("info/csv/mapping_cpsc_CinC.csv").drop(["type","abbreviation_cpsc","abbreviation_wfdb"], axis=1)
    mapping_dict = mapping_df.set_index('id_cpsc')['id_wfdb'].to_dict()

    # Reads in the encoding csv provided by CinC and converts the snomed CT code column to the row index
    cinc_classes = pd.read_csv("info/csv/dx_classes_CinC.csv").set_index("SNOMED CT Code")
    cinc_classes.index = cinc_classes.index.map(str)

    # Creates an empty dataframe with one column per class/CT code
    metas = pd.DataFrame(columns=cinc_classes.index)

    for file in os.listdir(path):
        if file.endswith(".pk"):
            p = os.path.join(path, file)
            df, meta = pk.load(open(p, "rb"))

            # Read the meta information of the record and store the codes as list
            dx = meta["dx"].split(",")

            # If it is multi-label record, ensure the order matches the one of the original CPSC dataset
            if len(dx)>1:
                new_dx = []
                cspc_order = cpsc_labels.loc[os.path.splitext(file)[0]].dropna().astype('int64').to_list()
                for label in cspc_order:
                    assert str(mapping_dict[label]) in dx
                    new_dx.append(str(mapping_dict[label]))
                # dx now contains the same information as before, but potentially in a changed order
                if not dx == new_dx:
                    dx = new_dx
                    # Save the changed order also to the meta information of the pickle object!
                    meta["dx"] = ','.join(dx)  # convert list to comma separated string
                    pk.dump((df, meta), open(str(p), "wb"))

            for d in dx:
                # Appends a row in the new dataframe with the record path as row index
                # Sets a 1 in each CT code column that is class of the record
                metas.loc[p, d] = 1

    # Deletes all columns, i.e. CT codes, that are not present in none of the records
    # Hence, only the codes, which are the class for at least one record, are maintained
    metas = metas.dropna(axis=1, how="all")
    # Instead of the long codes, the classes are just numbered from 0 to N-1 but the mapping is stored
    keys = list(range(len(metas.columns)))
    values = metas.columns.to_list()
    own_wfdb_encoding = bidict(dict(zip(keys, values)))
    pd.DataFrame.from_dict(data=own_wfdb_encoding, orient='index', columns=['label']).reset_index(level=0).\
        to_csv('info/csv/own_encoding_CinC.csv', index=False)
    metas.columns = list(range(len(metas.columns)))

    # Iterate through the records (one row in metas per record) and update its meta information
    for p, classes in metas.iterrows():
        df, meta = pk.load(open(str(p), "rb"))
        # Appends to additional entries to the dict
        # The first is a Series containing a 1 if the class applies to the record and a 0 otherwise
        # The second is list of integers encoding the classes that apply to the record
        meta["classes_one_hot"] = classes.replace(np.nan, 0)
        meta["classes_encoded"] = [own_wfdb_encoding.inverse[label] for label in meta["dx"].split(",")]
        # The following potentially changes the order but can be used for integrity check
        assert set(meta["classes_encoded"]) == set(classes.dropna().keys().to_list()), \
            "Integrity check failed - check meta data cleaning during preprocessing"
        pk.dump((df, meta), open(str(p), "wb"))

    print("Finished meta data cleaning for " + src_path)


def min_max_scaling(path):
    """
        The method applies min-max-scaling to each record under the given path
        It uses the global min and max for the normalization, not the local ones per record
    """
    # Vanessa:
    # For Min-Max-Normalization with local min() and max() per record,
    # it would be the following (Pandas automatically applies column-wise function):
    # df=(df-df.min())/(df.max()-df.min())

    # The scaler internally maintains some attributes, which can be iteratively trained when using partial_fit
    # Attributes:
    # min_,: ndarray of shape (n_features,) -> per feature adjustment for minimum.
    # scale_,: ndarray of shape (n_features,) -> per feature relative scaling of the data
    # data_min_: ndarray of shape (n_features,) -> per feature minimum seen in the data
    # data_max_: ndarray of shape (n_features,) -> per feature maximum seen in the data
    # data_range_ ndarray of shape (n_features,) -> per feature range (data_max_ - data_min_) seen in the data
    # n_samples_seen_:  The number of samples processed by the estimator.
    #                   It will be reset on new calls to fit, but increments across partial_fit calls.
    scaler = MinMaxScaler()

    g_max = pd.DataFrame()

    invalid_files = []

    for file in os.listdir(path):
        if os.path.isdir(os.path.join(path, file)):
            continue

        df, meta = pk.load(open(os.path.join(path, file), "rb"))

        # Adds the max value for each lead of the record to a new row of the dataframe (the columns are again the leads)
        g_max = g_max.append(df.max(), ignore_index=True)

    # Sets a threshold for the max value per lead that valid records can contain
    # If this threshold is exceeded, the record is considered invalid
    # Again, the operations are applied column-wise, i.e. for each lead separately
    g_max_threshold = g_max.median() + (g_max.std() * 2.5)

    file_list = []

    for file in os.listdir(path):
        if os.path.isdir(os.path.join(path, file)):
            continue
        df, meta = pk.load(open(os.path.join(path, file), "rb"))

        valid = True
        for col in df.columns:
            # df.loc[:, col] selects all values contained in the given column
            # In this case, it returns all values of the current lead for the given record
            if df.loc[:, col].max() > g_max_threshold.loc[col] \
                    or abs(df.loc[:, col].min()) > g_max_threshold.loc[col]:
                valid = False
                invalid_files.append(file)

                df.loc[:, col].plot()
                plt.show()

                break

        if valid:
            file_list.append(file)

            # Partial_fit ==> Online computation of min and max on X for later scaling.
            # Needed for training the scaler iteratively; with the fit() method the previous training would be discarded
            scaler = scaler.partial_fit(df)

    if not os.path.exists(os.path.join(path, "minmax")):
        os.makedirs(os.path.join(path, "minmax"))

    for file in file_list:
        df, meta = pk.load(open(os.path.join(path, file), "rb"))

        # Scales the features of the dataframe according to the desired feature_range, which is (0,1) by default
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

        # Normalize with mean normalization
        # df.mean() and df.std() operate on all columns, i.e. leads, separately
        # df - df.mean(): For each col, the mean of the col is subtracted from each element in the respective col
        df = (df - df.mean()) / df.std()
        pk.dump((df, meta), open(os.path.join(path, "normalized", file), "wb"))


def show(path):
    """
    The method creates a plot for each .pk file stored under the respective path
    -> Lead I is plotted for each record
    """
    for file in os.listdir(path):
        if file.endswith(".pk"):
            df, meta = pk.load(open(os.path.join(path, file), "rb"))
            df.loc[:, "I"].plot()
            plt.show()
            pass


if __name__ == "__main__":
    # src_path = "data/CinC_CPSC/raw/"
    # dest_path = "data/CinC_CPSC/"
    # split_train_test(src_path,dest_path, test_ratio=0.2)

    # # Uncomment for applying basic preprocessing
    # # Reads the .mat files, possibly downsamples the data, extracts meta data and writes everything to pickle dumps
    # src_path = "data/CinC_CPSC/train"
    # target_path = "data/CinC_CPSC/train/preprocessed/"
    # run_basic_preprocessing(src_path, target_path, sampling=None)
    # src_path = "data/CinC_CPSC/test"
    # target_path = "data/CinC_CPSC/test/preprocessed/"
    # run_basic_preprocessing(src_path, target_path, sampling=None)

    # Uncomment to extend the meta information by encoded classes
    # More importantly, deal with multi-label-case to fix the order of labels to match the one of the original CPSC
    src_path = "data/CinC_CPSC/train/preprocessed/no_sampling/"
    clean_meta(src_path)
    src_path = "data/CinC_CPSC/test/preprocessed/no_sampling/"
    clean_meta(src_path)

    # Uncomment for applying further preprocessing like normalization or padding (padding not yet implemented)
    # src_path = "data/CinC_CPSC/train/preprocessed/without_sampling/"
    # normalize(src_path)
    # show(src_path + "normalized")
    # min_max_scaling(path=src_path)
    # show(src_path + "minmax")
    # src_path = "data/CinC_CPSC/test/preprocessed/without_sampling/"
    # normalize(src_path)
    # show(src_path + "normalized")
    # min_max_scaling(path=src_path)
    # show(src_path + "minmax")
