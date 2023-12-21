import os
import shutil
from shutil import copyfile


BASE_DATA_PATH = "/home/vab30xh/projects/2023-macro-paper-3.10/data/CinC_CPSC/cross_valid/250Hz/60s"
PROBLEM_FOLD_ID = 4
CROSS_FOLD_LOG_PATH = "/home/vab30xh/projects/2023-macro-paper-3.10/cross_fold_log"
BASE_DEST_PATH = "/home/vab30xh/projects/2023-macro-paper-3.10/data/debug"

valid_idx = []
with open(os.path.join(CROSS_FOLD_LOG_PATH, "cross_validation_valid_{}.txt".format(PROBLEM_FOLD_ID)), "rb") as file:
    for line in file:
        valid_idx.append(int(line.strip()))

test_idx = []
with open(os.path.join(CROSS_FOLD_LOG_PATH, "cross_validation_test_{}.txt".format(PROBLEM_FOLD_ID)), "rb") as file:
    for line in file:
        test_idx.append(int(line.strip()))

os.makedirs(BASE_DEST_PATH, exist_ok=True)

# Copy the validation data
if os.path.exists(os.path.join(BASE_DEST_PATH, "valid/")):
    # Delete the folder recursively before copying
    shutil.rmtree(os.path.join(BASE_DEST_PATH, "valid/"))
# Create the folder if it doesn't exist
os.makedirs(os.path.join(BASE_DEST_PATH, "valid/"))
# Copy everything in valid_idx from the base data path as validation data
for idx in valid_idx:
    copyfile(os.path.join(BASE_DATA_PATH, "A{num:04d}.pk".format(num=idx)),
             os.path.join(BASE_DEST_PATH, "valid/", "A{num:04d}.pk".format(num=idx)))


# Copy the test data
if os.path.exists(os.path.join(BASE_DEST_PATH, "test/")):
    # Delete the folder recursively before copying
    shutil.rmtree(os.path.join(BASE_DEST_PATH, "test/"))
# Create the folder if it doesn't exist
os.makedirs(os.path.join(BASE_DEST_PATH, "test/"))
# Copy everything in test_idx from the base data path as test data
for idx in test_idx:
    copyfile(os.path.join(BASE_DATA_PATH, "A{num:04d}.pk".format(num=idx)),
             os.path.join(BASE_DEST_PATH, "test/", "A{num:04d}.pk".format(num=idx)))

# Copy everything not in valid_idx or test_idx as train data
if os.path.exists(os.path.join(BASE_DEST_PATH, "train/")):
    # Delete the folder recursively before copying
    shutil.rmtree(os.path.join(BASE_DEST_PATH, "train/"))
# Create the folder if it doesn't exist
os.makedirs(os.path.join(BASE_DEST_PATH, "train/"))
# Retrieve train_idx as all the files in the base data path that are neither in valid_idx nor test_idx
all_idx = [int(file.replace("A","").replace(".pk","")) for file in os.listdir(BASE_DATA_PATH)]
train_idx = [idx for idx in all_idx if idx not in valid_idx and idx not in test_idx]
for idx in train_idx:
    copyfile(os.path.join(BASE_DATA_PATH, "A{num:04d}.pk".format(num=idx)),
             os.path.join(BASE_DEST_PATH, "train/", "A{num:04d}.pk".format(num=idx)))

