from collections import Counter

import numpy as np, os, sys, joblib
from scipy.io import loadmat
import matplotlib.pyplot as plt
import seaborn as sns

dirname = os.path.dirname(__file__)
input_directory = os.path.join(dirname, '../data/CPSC/raw/Training_WFDB')


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
    with open(os.path.join(input_directory_wfdb, header_file_name), 'r') as f:
        header = f.readlines()
    # Extract the lead names in the right order
    leads = []
    # Only rows 1 to 12 contain the information
    for i in range(1, 13):
        lead_info = header[i].split(" ")
        leads.append(lead_info[-1].replace('\n', ''))
    return leads


load_12ECG_data(input_directory=input_directory)

# Plot the first lead of the A001 record in the wfdb dataset
input_directory_wfdb = os.path.join(dirname, '../data/CPSC/raw/Training_WFDB')
# x_wfdb = loadmat(os.path.join(input_directory_wfdb, 'A0001.mat'))
# x_wfdb = np.asarray(x_wfdb['val'], dtype=np.float64)
# plt.figure(1)
# plt.plot(x_wfdb[0])
# plt.title("Lead I, A001, wfdb")
#
# # Plot the first lead of the A001 record in the original cpsc dataset
# input_directory_cpsc = os.path.join(dirname, '../data/CPSC/raw/TrainingSet1')
# x_cpsc = loadmat(os.path.join(input_directory_cpsc, 'A0001.mat'))
# x_cpsc= x_cpsc['ECG']['data'][0][0]
# plt.figure(2)
# plt.plot(x_cpsc[0])
# plt.title("Lead I, A001, cpsc")
#
# plt.show()


# Plot all leads of the  A0011 record in the interval [1s,10s], based on the wfdb dataset
x_wfdb = loadmat(os.path.join(input_directory_wfdb, 'A0011.mat'))
x_wfdb = np.asarray(x_wfdb['val'], dtype=np.float64)[:,500:5000]
leads = _extract_leads_from_header('A0011.hea')

# input_directory_cpsc = os.path.join(dirname, '../data/CPSC/raw/TrainingSet1')
# x_cpsc = loadmat(os.path.join(input_directory_cpsc, 'A0011.mat'))
# x_cpsc= x_cpsc['ECG']['data'][0][0]

fig, axs = plt.subplots(6, 2, figsize=(15,15))
fig.suptitle("Record A0011")
axis_0 = 0
axis_1 = 0
for i in range(len(x_wfdb)):
    lead = x_wfdb[i]
    axs[axis_0,axis_1].plot(lead)
    axs[axis_0,axis_1].set_title(leads[i])
    axis_0 = (axis_0+1) % 6
    if axis_0 == 0:
        axis_1 +=1
plt.savefig('Record A0011 - First 10s.pdf')
plt.show()
pass


