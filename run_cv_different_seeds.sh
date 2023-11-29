#!/bin/bash

# Total number of runs
total_runs=10

# Number of parallel executions per run
parallel_executions=5

# Path to your Python script
python_script="train_with_cv.py"

# Path to your config file
config_file="config_baseline_MHAttention_CV.json"

# Outer loop for the total number of runs
for ((run=1; run<=$total_runs; run++)); do
    echo "Starting run $run..."

    # Inner loop for parallel executions with different seeds
    for ((i=1; i<=$parallel_executions; i++)); do
        # Generate a random seed for this run
        random_seed=$RANDOM

        # Execute the Python script in the background with the random seed
        python $python_script -c $config_file --seed $random_seed &

        echo "  - Execution $i of run $run started with seed $random_seed"

        # Wait 5s to avoid starting all processes at the same time (name issues for folder system)
        sleep 5s
    done

    # Wait for all background processes to finish before starting the next run
    wait

    echo "Run $run completed."
done

echo "All runs completed."