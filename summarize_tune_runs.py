import json
import os
import pickle as pk
from functools import partial

import pandas as pd


def _bold_formatter(x, value, num_decimals=2):
    """Format a number in bold when (almost) identical to a given value.

    Args:
        x: Input number.

        value: Value to compare x with.

        num_decimals: Number of decimals to use for output format.

    Returns:
        String converted output.

    """
    # Consider values equal, when rounded results are equal
    # otherwise, it may look surprising in the table where they seem identical
    if round(x, num_decimals) == round(value, num_decimals):
        return f"\\textbf{{{x:.{num_decimals}f}}}"
    else:
        return f"{x:.{num_decimals}f}"


# path_to_tune = 'savedVM_v2/models/BaselineModelWithSkipConnectionsV2/experiment_1_1'  # Attention! The order of the hyper_params must match the one of params.json; it can differ from the order in train.py!
# hyper_params = ['down_sample', 'pos_skip', 'vary_channels']
# integer_vals = []
# single_precision = []
# desired_col_order = ['down_sample', 'vary_channels', 'pos_skip',
#                      'SNR', 'AF', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'VEB', 'STD', 'STE',
#                      'W-AVG_F1', 'W-AVG_ROC', 'W-AVG_Acc', 'MR',
#                      'CPCS_F1', 'CPCS_Faf', 'CPCS_Fblock', 'CPCS_Fpc', 'CPCS_Fst', 'Epochs']

# path_to_tune = 'savedVM_v2/models/BaselineModelWithSkipConnectionsAndNormV2/experiment_1_2_all'
# hyper_params = ["down_sample", "norm_pos", "norm_type",
#                 "pos_skip", "vary_channels"]
# integer_vals = []
# single_precision = []
# desired_col_order = ["down_sample", "vary_channels", "pos_skip", "norm_type", "norm_pos",
#                      'SNR', 'AF', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'VEB', 'STD', 'STE',
#                      'W-AVG_F1', 'W-AVG_ROC', 'W-AVG_Acc', 'MR',
#                      'CPCS_F1', 'CPCS_Faf', 'CPCS_Fblock', 'CPCS_Fpc', 'CPCS_Fst', 'Epochs']

# path_to_tune = 'savedVM_v2/models/BaselineModelWithSkipConnectionsAndNormV2PreActivation/experiment_1_3_with_pre_conv'
# hyper_params = ["down_sample", "norm_before_act", "norm_pos", "norm_type",
#                 "pos_skip", "vary_channels"]
# integer_vals = []
# single_precision = []
# desired_col_order = ["down_sample", "pos_skip", "norm_before_act",
#                      'SNR', 'AF', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'VEB', 'STD', 'STE',
#                      'W-AVG_F1', 'W-AVG_ROC', 'W-AVG_Acc', 'MR',
#                      'CPCS_F1', 'CPCS_Faf', 'CPCS_Fblock', 'CPCS_Fpc', 'CPCS_Fst', 'Epochs']


# # Contains best run from experiment_1_3 with different kernel sizes for the pre_conv
# path_to_tune = 'savedVM_v2/models/BaselineModelWithSkipConnectionsAndNormV2PreActivation/experiment_1_3_best_run_with_different_kernels_for_pre_conv'
# hyper_params = ["down_sample", "norm_before_act", "norm_pos", "norm_type",
#                 "pos_skip", "pre_conv_kernel", "use_pre_conv", "vary_channels"]
# integer_vals = []
# single_precision = []
# desired_col_order = ["pre_conv_kernel",
#                      'SNR', 'AF', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'VEB', 'STD', 'STE',
#                      'W-AVG_F1', 'W-AVG_ROC', 'W-AVG_Acc', 'MR',
#                      'CPCS_F1', 'CPCS_Faf', 'CPCS_Fblock', 'CPCS_Fpc', 'CPCS_Fst', 'Epochs']


# path_to_tune = 'savedVM_v2/models/CPSC_BaselineWithMultiHeadAttention_uBCE_F1/tune_run_2_additional_FC_discarded'
# # Attention! The order of the hyper_params must match the one of params.json; it can differ from the order in train.py!
# hyper_params = ['dropout_attention', 'gru_units', 'heads']
# integer_vals = ['gru_units', 'heads', 'Epochs']
# single_precision = ['dropout_attention']
# desired_col_order = ['dropout_attention', 'heads', 'gru_units',
#                      'SNR', 'AF', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'VEB', 'STD', 'STE',
#                      'W-AVG_F1', 'W-AVG_ROC', 'W-AVG_Acc', 'MR',
#                      'CPCS_F1', 'CPCS_Faf', 'CPCS_Fblock', 'CPCS_Fpc', 'CPCS_Fst', 'Epochs']


# path_to_tune = 'savedVM_v2/models/FinalModel/experiment_3_all'
# # Attention! The order of the hyper_params must match the one of params.json; it can differ from the order in train.py!
# hyper_params = ["discard_FC_before_MH", "down_sample", "dropout_attention", "gru_units", "heads",
#                 "norm_before_act", "norm_pos", "norm_type", "pos_skip",  "use_pre_activation_design",  "use_pre_conv",
#                 "vary_channels"]
# integer_vals = ['gru_units', 'heads', 'Epochs']
# single_precision = ['dropout_attention']
# desired_col_order = ['use_pre_activation_design', 'dropout_attention', 'heads', 'gru_units', 'discard_FC_before_MH',
#                      'SNR', 'AF', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'VEB', 'STD', 'STE',
#                      'W-AVG_F1', 'W-AVG_ROC', 'W-AVG_Acc', 'MR',
#                      'CPCS_F1', 'CPCS_Faf', 'CPCS_Fblock', 'CPCS_Fpc', 'CPCS_Fst', 'Epochs']


path_to_tune = 'savedVM_v2/models/FinalModel/experiment_3_with_FC'
hyper_params = ["discard_FC_before_MH", "down_sample", "dropout_attention", "gru_units", "heads",
                "norm_before_act", "norm_pos", "norm_type", "pos_skip",
                "use_pre_activation_design", "use_pre_conv", "vary_channels"]
integer_vals = ['gru_units', 'heads', 'Epochs']
single_precision = ['dropout_attention']
# Discard FC before MH is either True or False for all runs (depending on VM)
desired_col_order = ["use_pre_activation_design", "dropout_attention", "gru_units", "heads",
                     'SNR', 'AF', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'VEB', 'STD', 'STE',
                     'W-AVG_F1', 'W-AVG_ROC', 'W-AVG_Acc', 'MR',
                     'CPCS_F1', 'CPCS_Faf', 'CPCS_Fblock', 'CPCS_Fpc', 'CPCS_Fst', 'Epochs']


# OLD -------------------------
# path_to_tune = 'savedVM/models/CPSC_BaselineWithSkips/experiment_1_1'
# hyper_params = ["down_sample", "last_kernel_size_first_conv_blocks", "last_kernel_size_second_conv_blocks",
#                 "mid_kernel_size_first_conv_blocks", "mid_kernel_size_second_conv_blocks"]
# integer_vals = ["last_kernel_size_first_conv_blocks", "last_kernel_size_second_conv_blocks",
#                 "mid_kernel_size_first_conv_blocks", "mid_kernel_size_second_conv_blocks"]
# single_precision = []
# desired_col_order = ['down_sample', 'mid_kernel_size_first_conv_blocks', 'last_kernel_size_first_conv_blocks',
#                      'last_kernel_size_second_conv_blocks',
#                      'SNR', 'AF', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'VEB', 'STD', 'STE',
#                      'W-AVG_F1', 'W-AVG_RO-------------------------C', 'W-AVG_Acc', 'MR',
#                      'CPCS_F1', 'CPCS_Faf', 'CPCS_Fblock', 'CPCS_Fpc', 'CPCS_Fst']


# Columns to format with maximum condition and 2 floating decimals
max_columns = ['SNR', 'AF', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'VEB', 'STD', 'STE',
                  'W-AVG_F1', 'W-AVG_ROC', 'W-AVG_Acc', 'MR',
                  'CPCS_F1', 'CPCS_Faf', 'CPCS_Fblock', 'CPCS_Fpc', 'CPCS_Fst']

df_summary_valid = pd.DataFrame(
    columns=hyper_params + ['IAVB', 'AF', 'LBBB', 'PAC', 'RBBB', 'SNR', 'STD', 'STE', 'VEB', 'W-AVG_F1',
                            'CPCS_F1', 'CPCS_Faf', 'CPCS_Fblock', 'CPCS_Fpc', 'CPCS_Fst', 'W-AVG_ROC', 'W-AVG_Acc',
                            'MR', 'Epochs'])

df_summary_test = pd.DataFrame(
    columns=hyper_params + ['IAVB', 'AF', 'LBBB', 'PAC', 'RBBB', 'SNR', 'STD', 'STE', 'VEB', 'W-AVG_F1',
                            'CPCS_F1', 'CPCS_Faf', 'CPCS_Fblock', 'CPCS_Fpc', 'CPCS_Fst', 'W-AVG_ROC', 'W-AVG_Acc',
                            'MR'])


def _append_to_summary(path, df_summary, tune_dict, best_epoch=None):
    with open(os.path.join(path, "eval_class_wise.p"), "rb") as file:
        df_class_wise = pk.load(file)
    with open(os.path.join(path, "eval_single_metrics.p"), "rb") as file:
        df_single_metrics = pk.load(file)

    # Update param configuration
    row_values = [tune_dict[k] for k in tune_dict.keys()]
    # Append F1 metrics (class-wise + weighted AVG)
    f1_metrics = df_class_wise.loc['f1-score'][['IAVB', 'AF', 'LBBB', 'PAC', 'RBBB', 'SNR', 'STD', 'STE', 'VEB',
                                                'weighted avg']].values.tolist()
    row_values = row_values + f1_metrics
    # Append single metrics
    col_names = ['cpsc_F1', 'cpsc_Faf', 'cpsc_Fblock', 'cpsc_Fpc', 'cpsc_Fst']
    row_values = row_values + df_single_metrics[col_names].values.tolist()[0]
    # Append the weighted AVG for ROC-AUC
    row_values.append(df_class_wise.loc['torch_roc_auc']['weighted avg'])
    # Append the weighted AVG for Acc
    row_values.append(df_class_wise.loc['torch_accuracy']['weighted avg'])
    # Append the subset acc (=MR)
    row_values = row_values + df_single_metrics['sk_subset_accuracy'].values.tolist()
    # Append the number of epochs
    if best_epoch is not None:
        row_values = row_values + [best_epoch]

    # Add the row to the summary dataframe
    df_summary.loc[len(df_summary)] = row_values


# Loop through the runs and append the tuning parameters as well as the resulting metrics to the summary df
for tune_run in os.listdir(path_to_tune):
    tune_path = os.path.join(path_to_tune, tune_run)
    if os.path.isdir(tune_path):
        with open(os.path.join(tune_path, "progress.csv"), "r") as file:
            # Not improved for 20 epochs -> best was 21 epochs earlier
            # First line is header, so subtract 22
            best_epoch = sum(1 for line in file) - 22

        with open(os.path.join(tune_path, "params.json"), "r") as file:
            tune_dict = json.load(file)
        # Validation
        path = os.path.join(tune_path, "valid_output")
        _append_to_summary(path, df_summary_valid, tune_dict, best_epoch)

        # Test
        path = os.path.join(tune_path, "test_output")
        _append_to_summary(path, df_summary_test, tune_dict)

# Parse integer values as ints
for col in integer_vals:
    df_summary_valid[col] = df_summary_valid[col].apply(int)
    if col!='Epochs':
        df_summary_test[col] = df_summary_test[col].apply(int)

# Format single-precision floats
for col in single_precision:
    df_summary_valid[col] = pd.Series(["{0:.1f}".format(val) for val in df_summary_valid[col]],
                                      index=df_summary_valid.index)
    df_summary_test[col] = pd.Series(["{0:.1f}".format(val) for val in df_summary_test[col]],
                                     index=df_summary_test.index)

# Reorder the columns of the dataframe to match the one used in the thesis
df_summary_valid_reordered = df_summary_valid[desired_col_order]
df_summary_test_reordered = df_summary_test[desired_col_order[:-1]]  # omit epochs

# Sort the rows by the two F1-scores
order_by_cols = ['W-AVG_F1', 'MR', 'W-AVG_ROC']  # ['CPCS_F1', 'W-AVG_F1']
df_summary_valid_reordered = df_summary_valid_reordered.sort_values(by=order_by_cols, inplace=False,
                                                                    ascending=[False for i in range(0, len(order_by_cols))])
df_summary_test_reordered = df_summary_test_reordered.sort_values(by=order_by_cols, inplace=False,
                                                                  ascending=[False for i in range(0, len(order_by_cols))])

# Write the results to files
with open(os.path.join(path_to_tune, 'summary_valid.p'), 'wb') as file:
    pk.dump(df_summary_valid_reordered, file)

with open(os.path.join(path_to_tune, 'summary_test.p'), 'wb') as file:
    pk.dump(df_summary_test_reordered, file)

fmts_max_valid = {column: partial(_bold_formatter, value=df_summary_valid_reordered[column].max(), num_decimals=3)
                  for column in max_columns}
fmts_max_test = {column: partial(_bold_formatter, value=df_summary_test_reordered[column].max(), num_decimals=3) for column in max_columns}
with open(os.path.join(path_to_tune, 'summary_results.tex'), 'w') as file:
    file.write("Validation Summary:\n\n")
    df_summary_valid_reordered.to_latex(buf=file, index=False, float_format="{:0.3f}".format, formatters=fmts_max_valid,
                                        escape=False)
    file.write("\n\n\nTest Summary:\n\n")
    df_summary_test_reordered.to_latex(buf=file, index=False, float_format="{:0.3f}".format, formatters=fmts_max_test,
                                       escape=False)

print("Done")
