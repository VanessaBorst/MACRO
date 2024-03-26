import math
import os
import pickle as pk
import shutil
from datetime import timedelta
from multiprocessing import Pool

import numpy as np
import pandas as pd
import wfdb
from bidict import bidict
from matplotlib import pyplot as plt
from scipy.io import loadmat
from sklearn.model_selection import train_test_split

from utils import plot_record_from_df


def _get_seq_len(hz,desired_seconds):
    """
    Returns the number of samples that are needed to represent the given number of seconds
    @param hz: Sampling rate in Hz
    @param desired_seconds: Desired length of the record in seconds
    @return: Number of samples needed to represent the given number of seconds
    """
    return hz * desired_seconds


def _parse_and_downsample_record(src_path, file, target_path, sampling, orig_Hz=500):
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
        ms_per_timedelta = 1000 / orig_Hz
        df.index = pd.TimedeltaIndex([timedelta(milliseconds=i * ms_per_timedelta) for i in df.index], unit="ms")
        # Alternative: df.index = pd.TimedeltaIndex([timedelta(seconds=i / orig_Hz) for i in df.index], unit="ms")

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
    with Pool(12) as pool:
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
    cpsc_labels = pd.read_csv("preprocessing/info/REFERENCE_cpsc.csv").set_index("Recording")

    # Get the mapping between the classes and push them in a dict
    mapping_df= pd.read_csv("preprocessing/info/mapping_cpsc_CinC.csv").drop(["type", "abbreviation_cpsc", "abbreviation_wfdb"], axis=1)
    mapping_dict = mapping_df.set_index('id_cpsc')['id_wfdb'].to_dict()

    # Reads in the encoding csv provided by CinC and converts the snomed CT code column to the row index
    cinc_classes = pd.read_csv("preprocessing/info/dx_classes_CinC.csv").set_index("SNOMED CT Code")
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
        to_csv('preprocessing/info/own_encoding_CinC.csv', index=False)
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



def pad_or_truncate(path, seq_len, seconds=None, pad_halfs=False):
    """
        The method applies zero padding/truncation to each record under the given path:
        - pads records with a smaller amount of samples with zeros
        - cuts records that exceed seq_len from both sides and only uses values in the middle
    """

    folder_name = f"eq_len_{seq_len}" if seconds is None else f"eq_len_{seconds}s"
    folder_name = folder_name + "_pad_halfs" if pad_halfs else folder_name
    if not os.path.exists(os.path.join(path, folder_name)):
        os.makedirs(os.path.join(path, folder_name))

    for file in os.listdir(path):
        if ".pk" not in file:
            continue
        df_record, meta = pk.load(open(os.path.join(path, file), "rb"))

        # Do transformation
        record_len = len(df_record.index)
        diff = seq_len - record_len

        # Plot record to verify the padding visually
        # plot_record_from_df(record_name=str(file), df_record=df_record, preprocesed=False)

        if diff > 0:
            if not pad_halfs:
                # Pad the record to the maximum length of the batch
                df_zeros = pd.DataFrame([[0] * df_record.shape[1]] * diff, columns=df_record.columns)
                df_record = pd.concat([df_zeros, df_record], axis=0, ignore_index=True)
            else:
                # Pad the record to the maximum length of the batch but append half of the values to the beginning
                # and half to the end
                df_zeros = pd.DataFrame([[0] * df_record.shape[1]] * math.ceil(diff / 2), columns=df_record.columns)
                # If the diff is uneven, omit the last column in df_zeros  before concatenation
                if diff % 2 == 0:
                    df_record = pd.concat([df_zeros, df_record, df_zeros], axis=0, ignore_index=True)
                else:
                    df_record = pd.concat([df_zeros, df_record, df_zeros.iloc[:-1]], axis=0, ignore_index=True)
        elif diff < 0:
            # Cut the record to have length seq_len (if possible, cut the equal amount of values from both sides)
            # If the diff is not even, cut one value more from the beginning
            df_record = df_record.iloc[math.ceil(-diff / 2):record_len - math.floor(-diff / 2)]

        # Plot record to verify the padding visually
        # plot_record_from_df(record_name=str(file), df_record=df_record, preprocesed=True)

        # Dump updated, ready-to-use df to new pk file
        pk.dump((df_record, meta), open(os.path.join(path, folder_name, file), "wb"))


def show(path):
    """
    The method creates a plot for each .pk file stored under the respective path
    -> Lead I is plotted for each record
    """
    for file in os.listdir(path):
        if file.endswith(".pk"):
            df, meta = pk.load(open(os.path.join(path, file), "rb"))
            df.loc[:, "I"].plot()
            # Plot certain range of lead I
            # df.loc[6250:8750, "I"].plot()
            plt.show()
            pass


if __name__ == "__main__":
    src_path = "data/CinC_CPSC/raw/"
    dest_path = "data/CinC_CPSC/"
    split_train_test(src_path,dest_path, test_ratio=0.2)

    # Uncomment for applying basic preprocessing
    # Reads the .mat files, possibly downsamples the data, extracts meta data and writes everything to pickle dumps
    src_path = "data/CinC_CPSC/train/raw"
    target_path = "data/CinC_CPSC/train/preprocessed/"
    run_basic_preprocessing(src_path, target_path, sampling="4ms")
    src_path = "data/CinC_CPSC/test/raw"
    target_path = "data/CinC_CPSC/test/preprocessed/"
    run_basic_preprocessing(src_path, target_path, sampling="4ms")

    # Uncomment to extend the meta information by encoded classes
    # More importantly, deal with multi-label-case to fix the order of labels to match the one of the original CPSC
    src_path = "data/CinC_CPSC/train/preprocessed/4ms/"
    clean_meta(src_path)
    src_path = "data/CinC_CPSC/test/preprocessed/4ms/"
    clean_meta(src_path)

    # Uncomment for applying further preprocessing like padding
    # 4ms = 250Hz
    src_path = "data/CinC_CPSC/train/preprocessed/4ms/"
    for desired_len_in_seconds in [60]:     # [10,15,30,60]:
        seq_len = _get_seq_len(hz=250, desired_seconds=desired_len_in_seconds)
        pad_or_truncate(path=src_path, seq_len=seq_len, seconds=desired_len_in_seconds, pad_halfs=False)

    src_path = "data/CinC_CPSC/test/preprocessed/4ms/"
    for desired_len_in_seconds in [60]:     # [10,15,30,60]:
        seq_len = _get_seq_len(hz=250, desired_seconds=desired_len_in_seconds)
        pad_or_truncate(path=src_path, seq_len=seq_len, seconds=desired_len_in_seconds, pad_halfs=False)

    # show("data/CinC_CPSC/test/preprocessed/4ms/eq_len_60s")

    # Copy all files to another folder used for k-fold cross-validation
    for mode in ["train", "test"]:
        src_path = f"data/CinC_CPSC/{mode}/preprocessed/4ms/eq_len_60s"
        dest_path = f"data/CinC_CPSC/cross_valid/250Hz/60s"
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)
        for file in os.listdir(src_path):
            shutil.copy(os.path.join(src_path, file), dest_path)

    print("Finished")
