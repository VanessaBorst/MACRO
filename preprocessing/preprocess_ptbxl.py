import pandas as pd
import numpy as np
import wfdb
import ast
from datetime import timedelta
import pickle
from tqdm.auto import tqdm

import os

from sklearn.preprocessing import MultiLabelBinarizer


def select_data(XX, YY, ctype, min_samples, outputfolder):
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
        # One-hot encoding in the following order: mlb.classes_
        # -> ['CD', 'HYP', 'MI', 'NORM', 'STTC']
        y = mlb.transform(Y.superdiagnostic.values)
    elif ctype == 'subdiag':
        counts = pd.Series(np.concatenate(YY.subdiagnostic.values)).value_counts()
        counts = counts[counts > min_samples]
        YY.subdiagnostic = YY.subdiagnostic.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        YY['subdiagnostic_len'] = YY.subdiagnostic.apply(lambda x: len(x))
        X = XX[YY.subdiagnostic_len > 0]
        Y = YY[YY.subdiagnostic_len > 0]
        mlb.fit(Y.subdiagnostic.values)
        # One-hot encoding in the following order: mlb.classes_  
        # -> ['AMI', 'CLBBB', 'CRBBB', 'ILBBB', 'IMI', 'IRBBB', 'ISCA', 'ISCI', 'ISC_', 'IVCD',, 'LAFB/LPFB', 'LAO/LAE',
        # 'LMI', 'LVH', 'NORM', 'NST_', 'PMI', 'RAO/RAE', 'RVH', 'SEHYP', 'STTC', 'WPW', '_AVB']
        y = mlb.transform(Y.subdiagnostic.values)
    elif ctype == 'diag':
        X = XX[YY.diagnostic_len > 0]
        Y = YY[YY.diagnostic_len > 0]
        mlb.fit(Y.diagnostic.values)
        # One-hot encoding in the following order: mlb.classes_
        # ['1AVB', '2AVB', '3AVB', 'ALMI', 'AMI', 'ANEUR', 'ASMI', 'CLBBB',
        #        'CRBBB', 'DIG', 'EL', 'ILBBB', 'ILMI', 'IMI', 'INJAL', 'INJAS',
        #        'INJIL', 'INJIN', 'INJLA', 'IPLMI', 'IPMI', 'IRBBB', 'ISCAL',
        #        'ISCAN', 'ISCAS', 'ISCIL', 'ISCIN', 'ISCLA', 'ISC_', 'IVCD',
        #        'LAFB', 'LAO/LAE', 'LMI', 'LNGQT', 'LPFB', 'LVH', 'NDT', 'NORM',
        #        'NST_', 'PMI', 'RAO/RAE', 'RVH', 'SEHYP', 'WPW']
        y = mlb.transform(Y.diagnostic.values)
    elif ctype == 'form':
        # filter
        counts = pd.Series(np.concatenate(YY.form.values)).value_counts()
        counts = counts[counts > min_samples]
        YY.form = YY.form.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        YY['form_len'] = YY.form.apply(lambda x: len(x))
        # select
        X = XX[YY.form_len > 0]
        Y = YY[YY.form_len > 0]
        mlb.fit(Y.form.values)
        # One-hot encoding in the following order: mlb.classes_
        # -> ['ABQRS', 'DIG', 'HVOLT', 'INVT', 'LNGQT', 'LOWT', 'LPR', 'LVOLT', 'NDT', 'NST_', 'NT_', 'PAC',
        #       'PRC(S)', 'PVC', 'QWAVE', 'STD_', 'STE_', 'TAB_', 'VCLVH']
        y = mlb.transform(Y.form.values)
    elif ctype == 'rhythm':
        # filter
        counts = pd.Series(np.concatenate(YY.rhythm.values)).value_counts()
        counts = counts[counts > min_samples]
        YY.rhythm = YY.rhythm.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        YY['rhythm_len'] = YY.rhythm.apply(lambda x: len(x))
        # select
        X = XX[YY.rhythm_len > 0]
        Y = YY[YY.rhythm_len > 0]
        mlb.fit(Y.rhythm.values)
        # One-hot encoding in the following order: mlb.classes_
        # ['AFIB', 'AFLT', 'BIGU', 'PACE', 'PSVT', 'SARRH', 'SBRAD', 'SR', 'STACH', 'SVARR', 'SVTAC', 'TRIGU']
        y = mlb.transform(Y.rhythm.values)
    elif ctype == 'all':
        # filter
        counts = pd.Series(np.concatenate(YY.all_diags.values)).value_counts()
        counts = counts[counts > min_samples]
        YY.all_diags = YY.all_diags.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        YY['all_diags_len'] = YY.all_diags.apply(lambda x: len(x))
        # select
        X = XX[YY.all_diags_len > 0]
        Y = YY[YY.all_diags_len > 0]
        mlb.fit(Y.all_diags.values)
        # One-hot encoding in the following order: mlb.classes_c
        # ['1AVB', '2AVB', '3AVB', 'ABQRS', 'AFIB', 'AFLT', 'ALMI', 'AMI',
        #        'ANEUR', 'ASMI', 'BIGU', 'CLBBB', 'CRBBB', 'DIG', 'EL', 'HVOLT',
        #        'ILBBB', 'ILMI', 'IMI', 'INJAL', 'INJAS', 'INJIL', 'INJIN',
        #        'INJLA', 'INVT', 'IPLMI', 'IPMI', 'IRBBB', 'ISCAL', 'ISCAN',
        #        'ISCAS', 'ISCIL', 'ISCIN', 'ISCLA', 'ISC_', 'IVCD', 'LAFB',
        #        'LAO/LAE', 'LMI', 'LNGQT', 'LOWT', 'LPFB', 'LPR', 'LVH', 'LVOLT',
        #        'NDT', 'NORM', 'NST_', 'NT_', 'PAC', 'PACE', 'PMI', 'PRC(S)',
        #        'PSVT', 'PVC', 'QWAVE', 'RAO/RAE', 'RVH', 'SARRH', 'SBRAD',
        #        'SEHYP', 'SR', 'STACH', 'STD_', 'STE_', 'SVARR', 'SVTAC', 'TAB_',
        #        'TRIGU', 'VCLVH', 'WPW']
        y = mlb.transform(Y.all_diags.values)
    else:
        raise ValueError("Other variants not yet tested")

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

    elif sampling_rate == 250:
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

    if ctype in ['diag', 'subdiag', 'superdiag']:

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
                        case "subdiag":
                            c = diag_agg_df.loc[key].diagnostic_subclass
                            if str(c) != 'nan':
                                tmp.append(c)
                            # tmp.append(diag_agg_df.loc[key].diagnostic_subclass)
                        case _:
                            raise ValueError("The provided aggregation level is not supported.")
            return list(set(tmp))

        diag_agg_df = aggregation_df[aggregation_df.diagnostic == 1]
        if ctype == 'diag':
            labels_df['diagnostic'] = labels_df.scp_codes.apply(aggregate_diagnostic,
                                                                aggregation_level="all_diagnostic")
            labels_df['diagnostic_len'] = labels_df.diagnostic.apply(lambda x: len(x))
        elif ctype == 'subdiag':
            labels_df['subdiagnostic'] = labels_df.scp_codes.apply(aggregate_diagnostic,
                                                                   aggregation_level="subdiag")

            # # Function to check if the special key exists in the dictionary
            # def contains_key(d, key):
            #     return key in d
            #
            # # Function to count keys from the scp_codes dictionary that are in filter_strings
            # def count_matching_keys(d):
            #     # Count how many keys from the dictionary are present in filter_strings
            #     total_sum = sum(key in filter_strings for key in d.keys())
            #     if total_sum > 2:
            #         print("Record with more than 2 diags of the same sub-diag found")
            #     return total_sum
            #
            # # Do some data checking, e.g. for subclass STTC
            # labels_df_STTC = labels_df[labels_df.subdiagnostic.apply(lambda x: 'STTC' in x)]
            # labels_df_NDT = labels_df[labels_df['scp_codes'].apply(contains_key, key='EL')]
            # filter_strings = ['NDT', 'DIG', 'LNGQT', 'ANEUR', 'EL']
            #
            # print(f"Num Records with class NDT: {labels_df['scp_codes'].apply(contains_key, key='NDT').sum()}")
            # # IMPORTANT: Discrepancy comes from wrong annotations;
            # # For four records, NDT is only mentioned in the report column but not within SCP_CODES!!!!
            # print(f"Num Records with class DIG: {labels_df['scp_codes'].apply(contains_key, key='DIG').sum()}")
            # print(f"Num Records with class LNGQT: {labels_df['scp_codes'].apply(contains_key, key='LNGQT').sum()}")
            # print(f"Num Records with class ANEUR: {labels_df['scp_codes'].apply(contains_key, key='ANEUR').sum()}")
            # print(f"Num Records with class EL: {labels_df['scp_codes'].apply(contains_key, key='EL').sum()}")
            #
            #
            # # Apply the function and filter rows where more than one key matches
            # # (records that have different labels but only result in ONE sub-class label, i.e., STTC)
            # filtered_df = labels_df[labels_df['scp_codes'].apply(count_matching_keys) > 1]
            #
            # print(f"Records with subclass STTC: {len(labels_df_STTC)}")
            # print(f"Number if counting records with different labels several times: "
            #       f"{len(labels_df_STTC) + len(filtered_df)}")
            # print(f"Total number should be {1829 + 181 + 118 + 104 + 97}")

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
        labels_df['all_diags'] = labels_df.scp_codes.apply(lambda x: list(set(x.keys())))
        labels_df['all_diags_len'] = labels_df.all_diags.apply(lambda x: len(x))
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
        meta = {'classes_one_hot': pd.Series(y[idx]), 'classes_encoded': np.flatnonzero(y[idx] == 1).tolist()}

        # Dump the results to pickle
        pickle.dump((df_data, meta), open(f"{dest_path}/{file}.pk", "wb"))

    print("Finished data finalizing for " + dest_path)


def print_class_details(ctype, labels, agg_df):
    if ctype == "superdiag":
        available_classes = agg_df[agg_df.diagnostic == 1]['diagnostic_class'].unique()
        col_name = "superdiagnostic"
    elif ctype == "subdiag":
        available_classes = agg_df[agg_df.diagnostic == 1]['diagnostic_subclass'].unique()
        col_name="subdiagnostic"
    elif ctype =="diag":
        available_classes = agg_df[agg_df.diagnostic == 1].index.values
        col_name="diagnostic"
    elif ctype =="form":
        available_classes = agg_df[agg_df.form == 1.0].index.values
        col_name="form"
    elif ctype =="rhythm":
        available_classes = agg_df[agg_df.rhythm == 1.0].index.values
        col_name = "rhythm"
    elif ctype =="all":
        available_classes = agg_df.index.values
        col_name = "all_diags"
    else:
        raise ValueError("ctype not supported")

    for class_name in available_classes:
        filtration = labels[col_name].apply(lambda x: 1 if class_name in x else 0)
        print(f"Number of patients who have {class_name} in their {col_name} is: {sum(filtration == 1)}")
    filtration_no_diag= labels[col_name].apply(lambda x: 1 if len(x) == 0 else 0)
    print(f"Number of patients who have no {col_name} is: {sum(filtration_no_diag == 1)}")



if __name__ == "__main__":
    src_path = '../data/PTB_XL/raw/'
    ctypes = ['diag', 'subdiag', 'superdiag', 'form', 'rhythm', 'all']

    for ctype in ctypes:

        for sampling_rate in [100]:   #, 250]:

            dest_path = f'../data/PTB_XL/{ctype}_{sampling_rate}'
            if not os.path.exists(dest_path):
                os.makedirs(dest_path)

            X, raw_labels = load_dataset(src_path, sampling_rate)

            # Load scp_statements.csv for diagnostic aggregation
            agg_df = pd.read_csv(src_path + 'scp_statements.csv', index_col=0)

            aggregated_labels = compute_label_aggregations(labels_df=raw_labels, aggregation_df=agg_df, ctype=ctype)

            print_class_details(ctype, aggregated_labels, agg_df)

            # Select relevant data and convert to one-hot
            X, labels, Y, _ = select_data(X, aggregated_labels, ctype, min_samples=0, outputfolder=dest_path)

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
