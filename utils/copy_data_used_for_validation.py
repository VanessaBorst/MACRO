import os
import pickle
from shutil import copyfile

base_path = "../data/CinC_CPSC/train/preprocessed/4ms/eq_len_60s"
with open("../data_loader/valid_record_names_2024-01-10.", "rb") as file:
    record_names = pickle.load(file)

dest_p = os.path.join(base_path, "valid")
if not os.path.exists(dest_p):
    os.makedirs(dest_p)

for file in os.listdir(base_path):
    if file.endswith(".pk") and file in record_names:
        copyfile(os.path.join(base_path, file), os.path.join(dest_p, file))