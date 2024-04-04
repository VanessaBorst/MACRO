#!/bin/bash

dir1_files=(~/projects/2023-macro-paper-3.10/data/CinC_CPSC/train/preprocessed/4ms/eq_len_60s/*)
dir2_files=(~/projects/2023-macro-paper-3.10/data/CinC_CPSC/test/preprocessed/4ms/eq_len_60s/*)

for file1 in "${dir1_files[@]}"; do
    file1_name=$(basename "$file1")
    if [[ -e "dir2/$file1_name" ]]; then
        echo "$file1_name exists in both directories."
    else
        : # echo "$file1_name exists only in dir1."
    fi
done

#for file2 in "${dir2_files[@]}"; do
#    file2_name=$(basename "$file2")
#    if [[ ! -e "dir1/$file2_name" ]]; then
#        echo "$file2_name exists only in dir2."
#    fi
#done