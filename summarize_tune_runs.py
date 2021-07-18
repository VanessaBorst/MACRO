import json
import os
import pickle as pk
import pandas as pd

# path_to_tune = 'savedVM/models/CPSC_BaselineWithMultiHeadAttention/param_study_1'
# # Attention! The order of the hyper_params must match the one of params.json; it can differ from the order in train.py!
# hyper_params = ['dropout_attention', 'gru_units', 'heads']
# integer_vals = ['gru_units', 'heads']
# single_precision = ['dropout_attention']
# desired_col_order = ['dropout_attention', 'gru_units', 'heads',
#                      'SNR', 'AF', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'VEB', 'STD', 'STE',
#                      'W-AVG_F1', 'W-AVG_ROC', 'W-AVG_Acc',
#                      'CPCS_F1', 'CPCS_Faf', 'CPCS_Fblock', 'CPCS_Fpc', 'CPCS_Fst']

path_to_tune = 'savedVM/models/CPSC_BaselineWithSkips/experiment_1_1'
hyper_params = ["down_sample", "last_kernel_size_first_conv_blocks", "last_kernel_size_second_conv_blocks",
                "mid_kernel_size_first_conv_blocks", "mid_kernel_size_second_conv_blocks"]
integer_vals = ["last_kernel_size_first_conv_blocks", "last_kernel_size_second_conv_blocks",
                "mid_kernel_size_first_conv_blocks", "mid_kernel_size_second_conv_blocks"]
single_precision = []
desired_col_order = ['down_sample', 'mid_kernel_size_first_conv_blocks', 'last_kernel_size_first_conv_blocks',
                     'last_kernel_size_second_conv_blocks',
                     'SNR', 'AF', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'VEB', 'STD', 'STE',
                     'W-AVG_F1', 'W-AVG_ROC', 'W-AVG_Acc',
                     'CPCS_F1', 'CPCS_Faf', 'CPCS_Fblock', 'CPCS_Fpc', 'CPCS_Fst']


df_summary_valid = pd.DataFrame(
    columns=hyper_params + ['IAVB', 'AF', 'LBBB', 'PAC', 'RBBB', 'SNR', 'STD', 'STE', 'VEB', 'W-AVG_F1',
                            'CPCS_F1', 'CPCS_Faf', 'CPCS_Fblock', 'CPCS_Fpc', 'CPCS_Fst', 'W-AVG_ROC', 'W-AVG_Acc'])

df_summary_test = pd.DataFrame(
    columns=hyper_params + ['IAVB', 'AF', 'LBBB', 'PAC', 'RBBB', 'SNR', 'STD', 'STE', 'VEB', 'W-AVG_F1',
                            'CPCS_F1', 'CPCS_Faf', 'CPCS_Fblock', 'CPCS_Fpc', 'CPCS_Fst', 'W-AVG_ROC', 'W-AVG_Acc'])


def _append_to_summary(path, df_summary, tune_dict):
    with open(os.path.join(path, "eval_class_wise.p"), "rb") as file:
        df_class_wise = pk.load(file)
    with open(os.path.join(path, "eval_single_metrics.p"), "rb") as file:
        df_single_metrics = pk.load(file)

    # Update param configuration
    row_values = [tune_dict[k] for k in tune_dict.keys()]
    # Append F1 metrics (class-wise + weighted AVG)
    row_values = row_values + df_class_wise.loc['sk_f1'].values.tolist()
    # Append single metrics
    col_names = ['cpsc_F1', 'cpsc_Faf', 'cpsc_Fblock', 'cpsc_Fpc', 'cpsc_Fst']
    row_values = row_values + df_single_metrics[col_names].values.tolist()[0]
    # Append the weighted AVG for ROC-AUC
    row_values.append(df_class_wise.loc['torch_roc_auc']['weighted avg'])
    # Append the weighted AVG for Acc
    row_values.append(df_class_wise.loc['torch_accuracy']['weighted avg'])

    # Add the row to the summary dataframe
    df_summary.loc[len(df_summary)] = row_values


# Loop through the runs and append the tuning parameters as well as the resulting metrics to the summary df
for tune_run in os.listdir(path_to_tune):
    tune_path = os.path.join(path_to_tune, tune_run)
    if os.path.isdir(tune_path):
        with open(os.path.join(tune_path, "params.json"), "r") as file:
            tune_dict = json.load(file)
        # Validation
        path = os.path.join(tune_path, "valid_output")
        _append_to_summary(path, df_summary_valid, tune_dict)

        # Test
        path = os.path.join(tune_path, "test_output")
        _append_to_summary(path, df_summary_test, tune_dict)


# Parse integer values as ints
for col in integer_vals:
    df_summary_valid[col] = df_summary_valid[col].apply(int)
    df_summary_test[col] = df_summary_test[col].apply(int)

# Format single-precision floats
for col in single_precision:
    df_summary_valid[col] = pd.Series(["{0:.1f}".format(val) for val in df_summary_valid[col]],
                                      index=df_summary_valid.index)
    df_summary_test[col] = pd.Series(["{0:.1f}".format(val) for val in df_summary_test[col]],
                                     index=df_summary_test.index)

# Reorder the columns of the dataframe to match the one used in the thesis
df_summary_valid_reordered = df_summary_valid[desired_col_order]
df_summary_test_reordered = df_summary_test[desired_col_order]

# Sort the rows by the two F1-scores
order_by_cols = ['CPCS_F1', 'W-AVG_F1']  # ['W-AVG_F1', 'CPCS_F1']
df_summary_valid_reordered.sort_values(by=order_by_cols, inplace=True, ascending=False)
df_summary_test_reordered.sort_values(by=order_by_cols, inplace=True, ascending=False)


# Write the results to files
with open(os.path.join(path_to_tune, 'summary_valid.p'), 'wb') as file:
    pk.dump(df_summary_valid_reordered, file)

with open(os.path.join(path_to_tune, 'summary_test.p'), 'wb') as file:
    pk.dump(df_summary_test_reordered, file)

with open(os.path.join(path_to_tune, 'summary_results.tex'), 'w') as file:
    file.write("Validation Summary:\n\n")
    df_summary_valid_reordered.to_latex(buf=file, index=False, float_format="{:0.3f}".format)
    file.write("\n\n\nTest Summary:\n\n")
    df_summary_test_reordered.to_latex(buf=file, index=False, float_format="{:0.3f}".format)


print("Done")
