import os

path = "savedVM/models/CPSC_BaselineWithSkips/tune_random_search"

for trial in os.listdir(path):
    # Read json
    # Find line with max val cpsc f1
    # Write config and value to new dataframe
    print(trial)

# Save dataframe as cs