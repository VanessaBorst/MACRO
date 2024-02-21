#!/bin/bash

# Number of parallel executions
parallel_executions=12

# Path to your Python script
python_script="train_with_cv.py"

# Path to your config file
config_file="configs/config_baseline_CV.json"

# Values to try for gamma_neg, gamma_pos, and clip
gamma_neg_values=(1 2 3)
gamma_pos_values=(0 1 2)
clip_values=(0.05 0.1 0.2 0.3)

# Iterate through all combinations
for gamma_neg in "${gamma_neg_values[@]}"; do
    for gamma_pos in "${gamma_pos_values[@]}"; do
        # Skip iterations where gamma_neg < gamma_pos
        if ((gamma_neg < gamma_pos)); then
            continue
        fi

        for clip in "${clip_values[@]}"; do
            # Limit parallel executions to the specified number
            while [ $(jobs | wc -l) -ge $parallel_executions ]; do
                sleep 5m
            done

            # Modify config file with current values
            sed -i "s/\"gamma_neg\": [0-9.]*,/\"gamma_neg\": $gamma_neg,/" $config_file
            sed -i "s/\"gamma_pos\": [0-9.]*,/\"gamma_pos\": $gamma_pos,/" $config_file
            sed -i "s/\"clip\": [0-9]*\.[0-9]*/\"clip\": $clip/" $config_file

            # Update "run_details" line
            run_details="_${gamma_neg}_${gamma_pos}_${clip}"
            sed -i "s/\"run_details\": \"_250Hz_60s_asymmetric_loss*[^\"]*\"/\"run_details\": \"_250Hz_60s_asymmetric_loss$run_details\"/" "$config_file"

            # Execute the Python script in the background
            python $python_script -c $config_file &

            echo "  - Execution for gamma_neg=$gamma_neg, gamma_pos=$gamma_pos, clip=$clip started"

            # Wait 5s to avoid starting all processes at the same time (name issues for folder system)
            sleep 5s
        done
    done
done

# Wait for all background processes to finish
wait

echo "All runs completed."
