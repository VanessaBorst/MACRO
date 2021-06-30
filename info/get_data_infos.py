import os
import pickle as pk

import pandas as pd
import wfdb


def _retrieve_data_statistics_from_wfdb(src_path, dest_path='info/'):
    # Reads in the csv and converts the snomed CT code column to row index
    classes = pd.read_csv("data/CinC_CPSC/dx_classes.csv").set_index("SNOMED CT Code")

    # There is on dataframe per variant (empty dataframes with the named columns and one column per class/CT code)
    # With vs. without additional data
    # All labels vs first label counts
    ct_codes = [str(x) for x in classes.index.values]
    df_reduced_all_labels = pd.DataFrame(columns=['record', 'class_count'] + ct_codes).set_index('record')
    df_reduced_first_labels = pd.DataFrame(columns=['record', 'class_count'] + ct_codes).set_index('record')
    df_complete_all_labels = pd.DataFrame(columns=['record', 'class_count'] + ct_codes).set_index('record')
    df_complete_first_labels = pd.DataFrame(columns=['record', 'class_count'] + ct_codes).set_index('record')

    all_dfs = [df_reduced_all_labels, df_reduced_first_labels, df_complete_all_labels, df_complete_first_labels]

    for file in [f.split(".")[0] for f in sorted(os.listdir(src_path)) if f.endswith("hea")]:
        header = wfdb.rdheader(os.path.join(src_path, file))
        meta = {key.lower(): val for key, val in (h.split(": ") for h in header.comments)}
        dx = meta["dx"].split(",")
        if file.startswith('Q'):
            # Files with 'Q' are not considered in the reduced case
            df_complete_all_labels.loc[file, 'class_count'] = len(dx)
            df_complete_first_labels.loc[file, 'class_count'] = len(dx)

            for idx, d in enumerate(dx):
                # Appends a row in the new dataframe with the record path as row index
                # Sets a 1 in each CT code column that is class of the record
                df_complete_all_labels.loc[file, d] = 1
                if idx == 0:
                    df_complete_first_labels.loc[file, d] = 1

        else:
            # Files with 'A' are always considered
            df_reduced_all_labels.loc[file, 'class_count'] = len(dx)
            df_reduced_first_labels.loc[file, 'class_count'] = len(dx)
            df_complete_all_labels.loc[file, 'class_count'] = len(dx)
            df_complete_first_labels.loc[file, 'class_count'] = len(dx)

            for idx, d in enumerate(dx):
                # Appends a row in the new dataframe with the record path as row index
                # Sets a 1 in each CT code column that is class of the record
                df_complete_all_labels.loc[file, d] = 1
                df_reduced_all_labels.loc[file, d] = 1
                if idx == 0:
                    df_complete_first_labels.loc[file, d] = 1
                    df_reduced_first_labels.loc[file, d] = 1

    for df in all_dfs:
        # Instead of the long codes, the classes are named by their abbreviations
        df.columns = ['class_count'] + classes.Abbreviation.to_list()

        # Deletes all columns, i.e. CT codes, that are not present in none of the records
        # Hence, only the codes, which are the class for at least one record, are maintained
        df.dropna(axis=1, how="all", inplace=True)

    # Store the data to an excel file
    path = os.path.join(dest_path, 'data_statistics.xlsx')
    writer = pd.ExcelWriter(path, engine='xlsxwriter')
    df_reduced_all_labels.to_excel(writer, sheet_name='Reduced_AllLabels')
    df_reduced_first_labels.to_excel(writer, sheet_name='Reduced_FirstLabels')
    df_complete_all_labels.to_excel(writer, sheet_name='Complete_AllLabels')
    df_complete_first_labels.to_excel(writer, sheet_name='Complete_FirstLabels')
    writer.close()

    print("Data statistics from wfdb headers written to csv")

    return all_dfs


def _retrieve_data_statistics_from_cleaned_pickle(src_paths, dest_path='info/'):
    # Reads in the csv and converts the encoding code column to row index
    own_encoding = pd.read_csv("info/csv/own_encoding_CinC.csv").set_index("index")

    # Reads in the csv to get a mapping of CinC ct codes to abbreviations
    mapping_df = pd.read_csv("info/csv/mapping_cpsc_CinC.csv").drop(["type", "abbreviation_cpsc", "id_cpsc"],
                                                                    axis=1).set_index("id_wfdb")

    # There is on dataframe per variant (empty dataframes with the named columns and one column per class/CT code)
    # All labels vs first label counts
    all_labels = [str(x) for x in own_encoding.index.values]
    df_all_labels = pd.DataFrame(columns=['record', 'set', 'class_count'] + all_labels).set_index('record')
    df_first_labels = pd.DataFrame(columns=['record', 'set', 'class_count'] + all_labels).set_index('record')

    all_dfs = [df_all_labels, df_first_labels]

    for path in src_paths:
        for file in os.listdir(path):
            if file.endswith(".pk"):
                p = os.path.join(path, file)
                df, meta = pk.load(open(p, "rb"))
                labels = meta["classes_encoded"]

                assert "train" in path or "test" in path, "Path should either contain the train or the test data"
                df_all_labels.loc[file, 'set'] = "train" if "train" in path else "test"
                df_first_labels.loc[file, 'set'] = "train" if "train" in path else "test"

                df_all_labels.loc[file, 'class_count'] = len(labels)
                df_first_labels.loc[file, 'class_count'] = len(labels)

                if len(labels) > 1:
                    print("Record: " + file + ", Encoded Labels: " + str(labels))

                for idx, label in enumerate(labels):
                    # Appends a row in the new dataframe with the record path as row index
                    # Sets a 1 in each CT code column that is class of the record
                    df_all_labels.loc[file, str(label)] = 1
                    if idx == 0:
                        df_first_labels.loc[file, str(label)] = 1

    for df in all_dfs:
        # Instead of the encoding codes, the classes are named by their abbreviations
        label_abbrevs = [mapping_df.loc[wfdb_code]["abbreviation_wfdb"] for wfdb_code in own_encoding.label.to_list()]
        df.columns = ['set', 'class_count'] + label_abbrevs

        # Deletes all columns, i.e. CT codes, that are not present in none of the records
        # Hence, only the codes, which are the class for at least one record, are maintained
        df.dropna(axis=1, how="all", inplace=True)

    # Store the data to an excel file
    path = os.path.join(dest_path, 'data_statistics_cleaned.xlsx')
    writer = pd.ExcelWriter(path, engine='xlsxwriter')
    df_all_labels.to_excel(writer, sheet_name='AllLabels')
    df_first_labels.to_excel(writer, sheet_name='FirstLabels')
    writer.close()

    print("Data statistics from cleaned pickle written to csv")

    return all_dfs


def _verify_cleaned_pickle(src_paths):
    # Read in the REFERENCES.csv provided by the official CPSC
    cpsc_labels = pd.read_csv("info/csv/REFERENCE_cpsc.csv").set_index("Recording")


    # Reads in the csv and converts the encoding code column to row index
    own_encoding = pd.read_csv("info/csv/own_encoding_CinC.csv").set_index("index")

    # Reads in the csv to get a mapping of CinC ct codes to abbreviations
    mapping_df = pd.read_csv("info/csv/mapping_cpsc_CinC.csv").drop(["type", "abbreviation_cpsc"],
                                                                    axis=1).set_index("id_wfdb")

    num_ml_records = 0
    df_ml_records = pd.DataFrame(columns=['record', 'encoded_labels', 'id_wfdb', 'abbrev_label', 'conv_id_cpsc',
                                          'real_id_cpsc']).set_index('record')
    for path in src_paths:
        for file in os.listdir(path):
            if file.endswith(".pk"):
                p = os.path.join(path, file)
                df, meta = pk.load(open(p, "rb"))
                labels = meta["classes_encoded"]

                if len(labels) > 1:
                    file_name = os.path.splitext(file)[0]
                    num_ml_records += 1
                    df_ml_records.loc[file_name, 'encoded_labels'] = labels
                    id_wfdb = [own_encoding.loc[label]['label'] for label in labels]
                    df_ml_records.loc[file_name, 'id_wfdb'] = id_wfdb
                    df_ml_records.loc[file_name, 'abbrev_label'] = \
                        [mapping_df.loc[wfdb_code]["abbreviation_wfdb"] for wfdb_code in id_wfdb]
                    df_ml_records.loc[file_name, 'conv_id_cpsc'] = \
                        [mapping_df.loc[wfdb_code]["id_cpsc"] for wfdb_code in id_wfdb]
                    df_ml_records.loc[file_name, 'real_id_cpsc'] = cpsc_labels.loc[file_name].dropna().astype('int64').to_list()

    assert (df_ml_records.conv_id_cpsc == df_ml_records.real_id_cpsc).value_counts().get(False, 0) == 0
    assert (df_ml_records.conv_id_cpsc == df_ml_records.real_id_cpsc).value_counts().get(True, 0) == num_ml_records

    # Some further tests
    assert cpsc_labels[(cpsc_labels['First_label'] == 6) & (cpsc_labels['Second_label'].notnull())].shape[0] == 42
    assert cpsc_labels[(cpsc_labels['First_label'] == 7) & (cpsc_labels['Second_label'].notnull())].shape[0] == 46
    assert cpsc_labels[(cpsc_labels['First_label'] == 8) & (cpsc_labels['Second_label'].notnull())].shape[0] == 42

    multi_labels_references_df = cpsc_labels[(cpsc_labels['Second_label'].notnull())]
    set_cpsc = set(multi_labels_references_df.index.values.tolist())
    set_wfdb = set(df_ml_records.index.values.tolist())
    print("Additional multi-label record in CPSC: " + str(set_cpsc - set_wfdb) + " (first label= second label = 6)")

    df_PAC_is_first_my_labels = df_ml_records.loc[[True if val[0] == 3 else False for (i, val) in df_ml_records.encoded_labels.iteritems()]]

    real_first_labels = {}
    for record_labels in df_ml_records.conv_id_cpsc.values:
        first_label = record_labels[0]
        real_first_labels[first_label] = real_first_labels.get(first_label, 0) + 1

    my_first_labels = {}
    for record_labels in df_ml_records.encoded_labels.values:
        first_label = record_labels[0]
        my_first_labels[first_label] = my_first_labels.get(first_label, 0) + 1


    print("Number of multi-label subjects: " + str(num_ml_records))


if __name__ == "__main__":
    # src_path = "data/CinC_CPSC/raw/"
    # dest_path = "info/"
    # df_list = _retrieve_data_statistics_from_wfdb(src_path)

    src_paths = ["data/CinC_CPSC/train/preprocessed/no_sampling",
                 "data/CinC_CPSC/test/preprocessed/no_sampling"]
    dest_path = "info/"
    # df_list = _retrieve_data_statistics_from_cleaned_pickle(src_paths, dest_path)
    _verify_cleaned_pickle(src_paths)

    print("")
