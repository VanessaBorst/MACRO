from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy.io import loadmat

dirname = os.path.dirname(__file__)
input_directory = os.path.join(dirname, '../data/CinC_CPSC/raw/')


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
        recording, header = load_challenge_data(header_files[i])
        lengths.append(len(recording[0]))
        recordings.append(recording)
        headers.append(header)

    counter = Counter(lengths)
    print(counter.most_common())

    # V1
    min_bin = min(counter.keys())
    max_bin = max(counter.keys())
    bins = np.arange(min_bin, max_bin + 1000, 1000)
    plt.xlabel("Record Length (in s)", fontsize=19, labelpad=10)
    plt.ylabel("Frequency", fontsize=19, labelpad=10)
    plt.tick_params(axis='both', which='major', labelsize=18)
    # Ticks are placed at positions 0 to 70000 but the values displayed are in seconds (500HZ -> Value/500)
    plt.xticks(np.arange(0, 80000, 10000), labels=np.arange(0, 160, 20))
    # plt.xaxis.set_ticklabels()
    # plt.figure(figsize=(14, 8))
    plt.hist(lengths, bins=bins)

    # plt.show()

    # location for the zoomed portion
    sub_axes = plt.axes([.45, .45, .45, .45])

    # plot the zoomed portion
    subset_counter = {k: counter[k] for k in counter.keys() if k > 30000}
    min_bin = min(subset_counter.keys())
    max_bin = max(subset_counter.keys())
    bins = np.arange(min_bin, max_bin + 1000, 1000)
    # subset_counter.pop(72000, None)     # Do not plot records exactly having 30s
    sub_axes.set_ylim(0, 4.3)
    sub_axes.set_yticks([1, 2, 3, 4])
    sub_axes.hist(lengths, bins=bins)
    sub_axes.tick_params(axis='both', which='major', labelsize=18)
    # Ticks are placed at positions 30000 to 70000 but the values displayed are in seconds (500HZ -> Value/500)
    sub_axes.set_xticks(np.arange(30000, 80000, 10000))
    sub_axes.set_xticklabels(np.arange(60, 160, 20))
    plt.tight_layout()
    plt.savefig("plots/seq_len_hist.pdf")
    plt.show()

    all_lengths = sorted(counter.items(), key=lambda pair: pair[1], reverse=True)
    print("Number of records per record length (" + str(len(all_lengths)) + " entries):")
    print(all_lengths)
    print("Total number of records: " + str(sum(counter.values())))

    print("TOP 10 most common record lengths:")
    print(counter.most_common(10))

    filtered_lengths = dict(filter(lambda elem: elem[1] >= 10, counter.items()))
    print("Only lengths which are represented by at least 10 records (" + str(len(filtered_lengths)) + " entries):")
    print(sorted(filtered_lengths.items(), key=lambda pair: pair[1], reverse=True))
    print("Total number of records: " + str(sum(filtered_lengths.values())))

    # Train model.
    # ...


# Load challenge data.
def load_challenge_data(header_file):
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


def _extract_leads_from_header(header_file_name):
    with open(os.path.join(input_directory, header_file_name), 'r') as f:
        header = f.readlines()
    # Extract the lead names in the right order
    leads = []
    # Only rows 1 to 12 contain the information
    for i in range(1, 13):
        lead_info = header[i].split(" ")
        leads.append(lead_info[-1].replace('\n', ''))
    return leads


def _extract_multi_labeled_records(input_directory):
    """
    Method can be used to verify that the wfdb data contains a switched order of labels
    The first label of wfdb is not the same as the one given by REFERENCES.csv of the CPSC challenge :(
    """
    header_files = []
    for f in sorted(os.listdir(input_directory)):
        g = os.path.join(input_directory, f)
        if not f.lower().startswith('.') and f.lower().endswith('hea') and os.path.isfile(g):
            header_files.append(g)

    encoding = {
        "426783006": 1,
        "164889003": 2,
        "270492004": 3,
        "164909002": 4,
        "59118001": 5,
        "284470004": 6,
        "164884008": 7,
        "429622005": 8,
        "164931005": 9
    }

    multi_label_records = []
    for file in header_files:
        classes = []
        with open(file, 'r') as f:
            for line in f:
                if line.startswith('#Dx'):
                    tmp = line.split(': ')[1].split(',')
                    for c in tmp:
                        classes.append(encoding[c.strip()])
        if len(classes) > 1:
            multi_label_records.append([file.split('/')[-1], classes])

    df_multi_labels = pd.DataFrame(multi_label_records, columns=['Record', 'Labels']).set_index('Record')
    df_multi_labels.sort_index(inplace=True)
    return df_multi_labels


# load_12ECG_data(input_directory=input_directory)
df = _extract_multi_labeled_records(input_directory)
# load_12ECG_data(input_directory)
print("")
