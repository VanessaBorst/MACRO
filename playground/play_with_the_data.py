from collections import Counter

import numpy as np, os, sys, joblib
from scipy.io import loadmat
import matplotlib.pyplot as plt
import seaborn as sns

dirname = os.path.dirname(__file__)
input_directory_wfdb = os.path.join(dirname, '../data/CPSC/raw/Training_WFDB')


def load_12ECG_data(input_directory):
    # Load data.
    print('Loading data...')

    header_files = []
    for f in os.listdir(input_directory):
        g = os.path.join(input_directory, f)
        if not f.lower().startswith('.') and f.lower().endswith('hea') and os.path.isfile(g):
            header_files.append(g)

    classes = get_classes(header_files)
    num_classes = len(classes)
    num_files = len(header_files)
    recordings = list()
    headers = list()

    lengths = []

    for i in range(num_files):
        recording, header = _load_challenge_data(header_files[i])
        lengths.append(len(recording[0]))
        recordings.append(recording)
        headers.append(header)

    counter = Counter(lengths)

    # V1
    min_bin = min(counter.keys())
    max_bin = max(counter.keys())
    bins = np.arange(min_bin, max_bin + 1, 100)
    plt.hist(lengths, bins=bins)
    plt.show()

    all_lengths = sorted(counter.items(), key=lambda pair: pair[1], reverse=True)
    print("Number of records per record length (" + str(len(all_lengths))+" entries):")
    print(all_lengths)
    print("Total number of records: " + str(sum(counter.values())))

    print("TOP 10 most common record lengths:")
    print(counter.most_common(10))

    filtered_lengths = dict(filter(lambda elem: elem[1] >= 10, counter.items()))
    print("Only lengths which are represented by at least 10 records (" + str(len(filtered_lengths))+" entries):")
    print(sorted(filtered_lengths.items(), key=lambda pair: pair[1], reverse=True))
    print("Total number of records: " + str(sum(filtered_lengths.values())))


    # Train model.
    # ...


# Load challenge data.
def _load_challenge_data(header_file):
    """
            :param header_file: full path to the header file

            :return recording: ndarray of size (12xnumSamples) containing the record data
            :return header: list with each element representing one line of the .hea file
    """
    with open(header_file, 'r') as f:
        header = f.readlines()
    mat_file = header_file.replace('.hea', '.mat')
    x = loadmat(mat_file)
    recording = np.asarray(x['val'], dtype=np.float64)
    return recording, header


# Find unique classes.
def get_classes(filenames):
    classes = set()
    for filename in filenames:
        with open(filename, 'r') as f:
            for l in f:
                if l.startswith('#Dx'):
                    tmp = l.split(': ')[1].split(',')
                    for c in tmp:
                        classes.add(c.strip())
    return sorted(classes)


def _extract_lead_names_from_header(header_file_name):
    """
        :param file name, e.g. "A0011.hea"

        :return list containing the lead names for the record
    """
    with open(os.path.join(input_directory_wfdb, header_file_name), 'r') as f:
        header = f.readlines()
    # Extract the lead names in the right order
    leads = []
    # Only rows 1 to 12 contain the information
    for i in range(1, 13):
        lead_info = header[i].split(" ")
        leads.append(lead_info[-1].replace('\n', ''))
    return leads


if __name__ == "__main__":
    load_12ECG_data(input_directory=input_directory_wfdb)

