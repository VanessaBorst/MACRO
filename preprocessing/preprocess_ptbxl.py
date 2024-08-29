import pandas as pd
import numpy as np
import wfdb
import ast
from datetime import timedelta
import pickle
from tqdm.auto import tqdm

import os
import joblib
import shutil

from sklearn.preprocessing import MultiLabelBinarizer


def select_data(XX,YY, ctype, min_samples, outputfolder):
    # convert multilabel to multi-hot
    mlb = MultiLabelBinarizer()

    if ctype == 'superdiag':
        # Concat all label annotations of all records to count their number
        counts = pd.Series(np.concatenate(YY.superdiagnostic.values)).value_counts()
        # Counts is a series with class names as index and counts as values
        # Select only those classes with the required minimum number of samples
        counts = counts[counts > min_samples]
        YY.superdiagnostic = YY.superdiagnostic.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        YY['superdiagnostic_len'] = YY.superdiagnostic.apply(lambda x: len(x))
        # Only consider records with at least one class label with a boolean series-based selection
        X = XX[YY.superdiagnostic_len > 0]
        Y = YY[YY.superdiagnostic_len > 0]
        mlb.fit(Y.superdiagnostic.values)
        # One-hot encoding in the following order: mlb.classes_    -> ['CD', 'HYP', 'MI', 'NORM', 'STTC']
        y = mlb.transform(Y.superdiagnostic.values)
    else:
        raise ValueError("Other variants not yet tested")

    # if ctype == 'diagnostic':
    #     X = XX[YY.diagnostic_len > 0]
    #     Y = YY[YY.diagnostic_len > 0]
    #     mlb.fit(Y.diagnostic.values)
    #     y = mlb.transform(Y.diagnostic.values)
    # elif ctype == 'subdiagnostic':
    #     counts = pd.Series(np.concatenate(YY.subdiagnostic.values)).value_counts()
    #     counts = counts[counts > min_samples]
    #     YY.subdiagnostic = YY.subdiagnostic.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
    #     YY['subdiagnostic_len'] = YY.subdiagnostic.apply(lambda x: len(x))
    #     X = XX[YY.subdiagnostic_len > 0]
    #     Y = YY[YY.subdiagnostic_len > 0]
    #     mlb.fit(Y.subdiagnostic.values)
    #     y = mlb.transform(Y.subdiagnostic.values)
    # elif ctype == 'superdiag':
    #     # See above
    #     pass
    # elif ctype == 'form':
    #     # filter
    #     counts = pd.Series(np.concatenate(YY.form.values)).value_counts()
    #     counts = counts[counts > min_samples]
    #     YY.form = YY.form.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
    #     YY['form_len'] = YY.form.apply(lambda x: len(x))
    #     # select
    #     X = XX[YY.form_len > 0]
    #     Y = YY[YY.form_len > 0]
    #     mlb.fit(Y.form.values)
    #     y = mlb.transform(Y.form.values)
    # elif ctype == 'rhythm':
    #     # filter
    #     counts = pd.Series(np.concatenate(YY.rhythm.values)).value_counts()
    #     counts = counts[counts > min_samples]
    #     YY.rhythm = YY.rhythm.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
    #     YY['rhythm_len'] = YY.rhythm.apply(lambda x: len(x))
    #     # select
    #     X = XX[YY.rhythm_len > 0]
    #     Y = YY[YY.rhythm_len > 0]
    #     mlb.fit(Y.rhythm.values)
    #     y = mlb.transform(Y.rhythm.values)
    # elif ctype == 'all':
    #     # filter
    #     counts = pd.Series(np.concatenate(YY.all_scp.values)).value_counts()
    #     counts = counts[counts > min_samples]
    #     YY.all_scp = YY.all_scp.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
    #     YY['all_scp_len'] = YY.all_scp.apply(lambda x: len(x))
    #     # select
    #     X = XX[YY.all_scp_len > 0]
    #     Y = YY[YY.all_scp_len > 0]
    #     mlb.fit(Y.all_scp.values)
    #     y = mlb.transform(Y.all_scp.values)
    # else:
    #     pass

    # save LabelBinarizer
    with open(os.path.join(outputfolder, 'mlb.pkl'), 'wb') as tokenizer:
        pickle.dump(mlb, tokenizer)

    return X, Y, y, mlb

def load_raw_data_ptbxl(df, sampling_rate, path):

    if sampling_rate == 100:
        if os.path.exists(path + "Serialized_" + 'raw100.npy'):
            data = np.load(path + "Serialized_" + 'raw100.npy', allow_pickle=True)
        else:
            data = [wfdb.rdsamp(path + f) for f in tqdm(df.filename_lr)]
            data = np.array([signal for signal, meta in data])
            pickle.dump(data, open(path + "Serialized_" + 'raw100.npy', 'wb'), protocol=4)

    elif sampling_rate == 500:
        if os.path.exists(path + "Serialized_" + 'raw500.npy'):
            data = np.load(path + "Serialized_" + 'raw500.npy', allow_pickle=True)
        else:
            data = [wfdb.rdsamp(path + f) for f in tqdm(df.filename_hr)]
            data = np.array([signal for signal, meta in data])
            pickle.dump(data, open(path + "Serialized_" + 'raw500.npy', 'wb'), protocol=4)

    elif sampling_rate ==  250:
        if os.path.exists(path + "Serialized_" + 'raw250.npy'):
            data = np.load(path + "Serialized_" + 'raw250.npy', allow_pickle=True)
        else:
            data = []
            for f in tqdm(df.filename_hr):
                # Get data with 500Hz
                single_record_data = wfdb.rdsamp(path + f)

                raw_signal = single_record_data[0]
                raw_signal_df = pd.DataFrame(raw_signal)

                meta = single_record_data[1]
                orig_Hz = meta['fs']
                # meta.update({'fs': 250, 'sig_len': 2500})

                # Apply the downsampling
                ms_per_timedelta = 1000 / orig_Hz
                raw_signal_df.index = pd.TimedeltaIndex([timedelta(milliseconds=i * ms_per_timedelta)
                                                         for i in raw_signal_df.index], unit="ms")
                sampling = "4ms"
                raw_signal_df = raw_signal_df.resample(sampling).mean()
                raw_signal_df.index = np.arange(0,
                                                raw_signal_df.shape[0])  # return to numbers as row index (0-#samples)

                # data_tuple = (raw_signal_df.to_numpy(), meta)
                # data.append(data_tuple)
                data.append(raw_signal_df.to_numpy())

            data = np.array([signal for signal in data])
            pickle.dump(data, open(path + "Serialized" + 'raw250.npy', 'wb'), protocol=4)
    else:
        raise ValueError('The given sampling rate is not supported')

    return data


def load_dataset(path, sampling_rate):
    # load and convert annotation data
    Y = pd.read_csv(path + 'ptbxl_database.csv', index_col='ecg_id')
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
    # Load raw signal data
    X = load_raw_data_ptbxl(Y, sampling_rate, path)

    return X, Y



def compute_label_aggregations(labels_df, aggregation_df, ctype):

    labels_df['scp_codes_len'] = labels_df.scp_codes.apply(lambda x: len(x))

    if ctype in ['diagnostic', 'subdiagnostic', 'superdiag']:

        def aggregate_diagnostic(y_dic, aggregation_level="superdiag"):
            tmp = []
            for key in y_dic.keys():
                if key in diag_agg_df.index:
                    match aggregation_level:
                        case "all_diagnostic":
                            tmp.append(key)
                        case "superdiag":
                            c = diag_agg_df.loc[key].diagnostic_class
                            if str(c) != 'nan':
                                tmp.append(c)
                            # tmp.append(diag_agg_df.loc[key].diagnostic_class)
                        case "subdiagnostic":
                            c = diag_agg_df.loc[key].diagnostic_subclass
                            if str(c) != 'nan':
                                tmp.append(c)
                            # tmp.append(diag_agg_df.loc[key].diagnostic_subclass)
                        case _:
                            raise ValueError("The provided aggregation level is not supported.")
            return list(set(tmp))

        diag_agg_df = aggregation_df[aggregation_df.diagnostic == 1]
        if ctype == 'diagnostic':
            labels_df['diagnostic'] = labels_df.scp_codes.apply(aggregate_diagnostic,
                                                                aggregation_level="all_diagnostic")
            labels_df['diagnostic_len'] = labels_df.diagnostic.apply(lambda x: len(x))
        elif ctype == 'subdiagnostic':
            labels_df['subdiagnostic'] = labels_df.scp_codes.apply(aggregate_diagnostic,
                                                                   aggregation_level="subdiagnostic")
            labels_df['subdiagnostic_len'] = labels_df.subdiagnostic.apply(lambda x: len(x))
        elif ctype == 'superdiag':
            labels_df['superdiagnostic'] = labels_df.scp_codes.apply(aggregate_diagnostic,
                                                                   aggregation_level="superdiag")
            labels_df['superdiagnostic_len'] = labels_df.superdiagnostic.apply(lambda x: len(x))
    elif ctype == 'form':

        form_agg_df = aggregation_df[aggregation_df.form == 1.0]
        def aggregate_form(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in form_agg_df.index:
                    c = key
                    if str(c) != 'nan':
                        tmp.append(c)
            return list(set(tmp))

        labels_df['form'] = labels_df.scp_codes.apply(aggregate_form)
        labels_df['form_len'] = labels_df.form.apply(lambda x: len(x))
    elif ctype == 'rhythm':
        rhythm_agg_df = aggregation_df[aggregation_df.rhythm == 1.0]

        def aggregate_rhythm(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in rhythm_agg_df.index:
                    c = key
                    if str(c) != 'nan':
                        tmp.append(c)
            return list(set(tmp))

        labels_df['rhythm'] = labels_df.scp_codes.apply(aggregate_rhythm)
        labels_df['rhythm_len'] = labels_df.rhythm.apply(lambda x: len(x))
    elif ctype == 'all':
        labels_df['all_scp'] = labels_df.scp_codes.apply(lambda x: list(set(x.keys())))
    else:
        raise ValueError(f'The given ctype {ctype} is not supported')

    return labels_df.reset_index(drop=True)




def finalize_preprocessing(dest_path, X, y, file_names):
    """
        This method appends the required information for each record
        To this end, two additional entries are added to a meta dictionary of the record
        1) meta["classes_one_hot"]: Pandas Series containing a 1 if the class applies to the record and a 0 otherwise
        2) meta["classes_encoded"]: List of integers encoding the classes that apply to the record
                            (the integers are in the range between 0 and N-1, with N being the number of classes
                            existing amongst the record under the given path)

        ==> The method only operates on the meta information and keeps the actual time series data unchanged
    """

    # path = "../data/CinC_CPSC/train/preprocessed/250Hz/"
    # file = os.listdir(path)[0]
    # df_SOLL, meta_SOLL = pickle.load(open(os.path.join(path, file), "rb"))

    if not os.path.exists(dest_path):
        os.makedirs(dest_path)

    for idx, file in enumerate(file_names):
        # Retrieve ECG data
        df_data = pd.DataFrame(X[idx])

        # Create required meta data
        meta = {'classes_one_hot': pd.Series(y[idx]), 'classes_encoded': np.flatnonzero(y[idx]==1).tolist()}

        # Dump the results to pickle
        pickle.dump((df_data, meta), open(f"{dest_path}/{file}.pk", "wb"))

    print("Finished data finalizing for " + dest_path)


if __name__ == "__main__":
    src_path = '../data/PTB_XL/raw/'
    ctype = "superdiag"

    for sampling_rate in [250]:

        dest_path = f'../data/PTB_XL/{ctype}_{sampling_rate}'
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)

        X, raw_labels = load_dataset(src_path,sampling_rate)

        # Load scp_statements.csv for diagnostic aggregation
        agg_df = pd.read_csv(src_path + 'scp_statements.csv', index_col=0)

        ####### OWN

        # agg_df = agg_df[agg_df.diagnostic == 1]

        # superclasses = agg_df['diagnostic_class'].unique()
        # subclasses = agg_df['diagnostic_subclass'].unique()
        #
        # # Apply diagnostic superclass
        # Y['superdiagnostic'] = Y.scp_codes.apply(aggregate_diagnostic, agg_df=agg_df, aggregation_level="superdiagnostic")
        # # Apply diagnostic subclass
        # Y['subdiagnostic'] = Y.scp_codes.apply(aggregate_diagnostic, agg_df=agg_df, aggregation_level="subdiagnostic")
        #
        #
        #
        #
        # filtration_no_diag_super = Y["superdiagnostic"].apply(lambda x: 1 if len(x) ==0 else 0)
        # print(f"Patient that have 'NORM' in their superdiagnostic is: {sum(filtration_no_diag_super == 1)}")
        #
        # filtration_no_diag_sub = Y["subdiagnostic"].apply(lambda x: 1 if len(x) == 0 else 0)
        # print(f"Patient that have 'NORM' in their subdiagnostic is: {sum(filtration_no_diag_sub == 1)}")


        labels = compute_label_aggregations(labels_df=raw_labels, aggregation_df=agg_df, ctype=ctype)

        if ctype=="superdiag":
            superclasses = agg_df[agg_df.diagnostic == 1]['diagnostic_class'].unique()
            for superclass in superclasses:
                filtration = labels["superdiagnostic"].apply(lambda x: 1 if superclass in x else 0)
                print(f"Patient that have {superclass} in their diagnostic is: {sum(filtration == 1)}")
            filtration_no_diag_super = labels["superdiagnostic"].apply(lambda x: 1 if len(x) == 0 else 0)
            print(f"Patient that have 'NORM' in their superdiagnostic is: {sum(filtration_no_diag_super == 1)}")

        # Select relevant data and convert to one-hot
        X, labels, Y, _ = select_data(X, labels, ctype, min_samples=0, outputfolder=dest_path)


        # Split data into train, valid and test set
        # 1-8 for training
        X_train = X[labels.strat_fold < 9]
        y_train = Y[labels.strat_fold < 9]
        names_train = labels[labels.strat_fold < 9].filename_lr.str[-8:-3].values
        # 9 for validation
        X_valid = X[labels.strat_fold == 9]
        y_valid = Y[labels.strat_fold == 9]
        names_valid = labels[labels.strat_fold == 9].filename_lr.str[-8:-3].values
        # 10 for test
        X_test = X[labels.strat_fold == 10]
        y_test = Y[labels.strat_fold == 10]
        names_test = labels[labels.strat_fold == 10].filename_lr.str[-8:-3].values

        # Write them to disk in the required format
        finalize_preprocessing(os.path.join(dest_path, "train"), X_train, y_train, names_train)
        finalize_preprocessing(os.path.join(dest_path, "valid"), X_valid, y_valid, names_valid)
        finalize_preprocessing(os.path.join(dest_path, "test"), X_test, y_test, names_test)
        ########################################


        print(f"Finished preprocessing for sampling frequency {sampling_rate} and aggregation level {ctype}")
