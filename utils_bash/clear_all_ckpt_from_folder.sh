#!/bin/bash

source venv/bin/activate

# Specify a list of folders
RELATIVE_PATHS_TO_FOLDERS=("../savedVM/" "../savedVM_old/" "../savedVM_until15Jan/" "../savedVM_veryOld/")

# Loop through each folder
for REL_PATH in "${RELATIVE_PATHS_TO_FOLDERS[@]}"; do
    # Print the disk space used before deletions
    echo "Disk space used before deletions in $REL_PATH:"
    du -sh "$REL_PATH"

    # See https://www.cyberciti.biz/faq/how-to-find-and-delete-directory-recursively-on-linux-or-unix-like-system/
    # -exec rm -rf {} + : Execute rm command.
    # This variant of the -exec action runs the specified command on the selected files, but the command line is built
    # by appending each selected file name at the end; the total number of invocations of the command will be much less than
    # the number of matched files.
    find $REL_PATH -type f -name 'checkpoint*' -exec rm -rf {} +

    # For tune runs
    find $REL_PATH -type d -name 'checkpoint*' -exec rm -rf {} +

    # Print the disk space used after deletions
    echo "Disk space used after deletions in $REL_PATH:"
    du -sh "$REL_PATH"

    echo "Done processing folder: $REL_PATH"
done

echo "Done!"
