import os
import pickle
from shutil import copyfile

base_path = "../data/CinC_CPSC/train/preprocessed/no_sampling/eq_len_72000"
with open("savedVM/log/CPSC_BaselineModel/ml_bs8/Record_names_valid.p", "rb") as file:
    record_names = pickle.load(file)

dest_p = os.path.join(base_path, "valid")
if not os.path.exists(dest_p):
    os.makedirs(dest_p)

for file in os.listdir(base_path):
    if file.endswith(".pk") and file in record_names:
        copyfile(os.path.join(base_path, file), os.path.join(dest_p, file))