import numpy as np, os, sys, joblib
from scipy.io import loadmat
import matplotlib.pyplot as plt

from playground.read_data import _extract_lead_names_from_header

dirname = os.path.dirname(__file__)
input_directory_wfdb = os.path.join(dirname, '../data/CPSC/raw/Training_WFDB')
input_directory_cpsc = os.path.join(dirname, '../data/CPSC/raw/TrainingSet1')

# Plot the first lead of the A001 record in the wfdb dataset
x_wfdb = loadmat(os.path.join(input_directory_wfdb, 'A0001.mat'))
x_wfdb = np.asarray(x_wfdb['val'], dtype=np.float64)
plt.figure(1)
plt.plot(x_wfdb[0])
plt.title("Lead I, A001, wfdb")

# Plot the first lead of the A001 record in the original cpsc dataset
x_cpsc = loadmat(os.path.join(input_directory_cpsc, 'A0001.mat'))
x_cpsc = x_cpsc['ECG']['data'][0][0]
plt.figure(2)
plt.plot(x_cpsc[0])
plt.title("Lead I, A001, cpsc")

plt.show()

# Plot all leads of the  A0011 record in the interval [1s,10s], based on the wfdb dataset
x_wfdb = loadmat(os.path.join(input_directory_wfdb, 'A0011.mat'))
x_wfdb = np.asarray(x_wfdb['val'], dtype=np.float64)[:, 500:5000]
x_cpsc = loadmat(os.path.join(input_directory_cpsc, 'A0011.mat'))
x_cpsc = x_cpsc['ECG']['data'][0][0][:, 500:5000]

leads = _extract_lead_names_from_header('A0011.hea')

for i in range(0, 2):
    data_src = x_wfdb if i == 0 else x_cpsc
    data_src_name = "wfdb" if i == 0 else "cpsc"
    fig, axs = plt.subplots(6, 2, figsize=(15, 15))
    fig.suptitle("Record A0011 (" + data_src_name + ")")
    axis_0 = 0
    axis_1 = 0
    for i in range(len(data_src)):
        lead = data_src[i]
        axs[axis_0, axis_1].plot(lead)
        axs[axis_0, axis_1].set_title(leads[i])
        axis_0 = (axis_0 + 1) % 6
        if axis_0 == 0:
            axis_1 += 1
    plt.savefig('plots/Record A0011 (' + data_src_name + ') - First 10s.pdf')
    plt.show()
