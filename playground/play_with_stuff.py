import os
import scipy.io as sio
import numpy as np

# Play with the data loading from the authors

record_base_path = "data/CPSC/raw/"

for mat_item in os.listdir(record_base_path):
    if mat_item.endswith('.mat') and (not mat_item.startswith('._')):
        print(mat_item)
        X_list = []
        ecg = np.zeros((72000, 12), dtype=np.float32)
        record_path = os.path.join(record_base_path, mat_item)
        # Read input of shape (12 x len), transpose it to get shape (len x 12)
        ecg[-sio.loadmat(record_path)['ECG'][0][0][2].shape[1]:, :] = sio.loadmat(record_path)['ECG'][0][0][2][:,
                                                                      -72000:].T

        # print(ecg.shape)
        X_list.append(ecg)
        X_list = np.asarray(X_list)
        X = X_list[:, :, :]