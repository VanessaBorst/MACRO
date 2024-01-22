import json
import os
import pickle as pk
from functools import partial

import mmap
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
# desired_col_order = ['down_sample',  'pos_skip', 'vary_channels',
#                      'SNR', 'AF', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'VEB', 'STD', 'STE',
#                      'W-AVG_F1', 'W-AVG_ROC', 'W-AVG_Acc', 'MR', 'Epochs']

# path_to_tune = 'savedVM_v2/models/BaselineModelWithSkipConnectionsAndNormV2/experiment_1_2_all'
# hyper_params = ["down_sample", "norm_pos", "norm_type",
#                 "pos_skip", "vary_channels"]
# integer_vals = []
# single_precision = []
# # "vary_channels" is always true
# desired_col_order = ["down_sample", "pos_skip", "norm_type", "norm_pos",
#                      'SNR', 'AF', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'VEB', 'STD', 'STE',
#                      'W-AVG_F1', 'W-AVG_ROC', 'W-AVG_Acc', 'MR', 'Epochs']

# path_to_tune = 'savedVM_v2/models/BaselineModelWithSkipConnectionsAndNormV2PreActivation/experiment_1_3_no_pre_conv_before_blocks_all'
# hyper_params = ["down_sample", "norm_before_act", "norm_pos", "norm_type",
#                 "pos_skip", "vary_channels"]
# integer_vals = []
# single_precision = []
# desired_col_order = ["down_sample", "pos_skip", "norm_before_act",
#                      'SNR', 'AF', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'VEB', 'STD', 'STE',
#                      'W-AVG_F1', 'W-AVG_ROC', 'W-AVG_Acc', 'MR', 'Epochs']


# # Contains best run from experiment_1_3 with different kernel sizes for the pre_conv
# path_to_tune = 'savedVM_v2/models/BaselineModelWithSkipConnectionsAndNormV2PreActivation/experiment_1_3_best_run_with_different_kernels_for_pre_conv'
# hyper_params = ["down_sample", "norm_before_act", "norm_pos", "norm_type",
#                 "pos_skip", "pre_conv_kernel", "use_pre_conv", "vary_channels"]
# integer_vals = []
# single_precision = []
# desired_col_order = ["pre_conv_kernel",
#                      'SNR', 'AF', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'VEB', 'STD', 'STE',
#                      'W-AVG_F1', 'W-AVG_ROC', 'W-AVG_Acc', 'MR', 'Epochs']

# 'savedVM_v2/models/CPSC_BaselineWithMultiHeadAttention_uBCE_F1/0810_215444_ml_bs64_rerun_100821_withFC'
path_to_tune = 'savedVM/models/BaselineWithMultiHeadAttention_ParamStudy/0117_145626_ml_bs64_attention_type_v2_withFC'
# 'savedVM_v2/models/CPSC_BaselineWithMultiHeadAttention_uBCE_F1/0810_215735_ml_bs64_rerun_100821_noFC'
# Attention! The order of the hyper_params must match the one of params.json; it can differ from the order in train.py!
hyper_params = ['discard_FC_before_MH', 'dropout_attention', 'gru_units', 'heads']
integer_vals = ['gru_units', 'heads', 'Epochs']
single_precision = ['dropout_attention']
desired_col_order = ['discard_FC_before_MH', 'dropout_attention', 'heads', 'gru_units',
                     'SNR', 'AF', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'VEB', 'STD', 'STE',
                     'm-F1', 'm-ROC-AUC', 'm-Acc',
                     'W-AVG_F1', 'W-AVG_ROC', 'W-AVG_Acc',
                     'MR', 'Epochs', 'Params']

# # Old runs (no discard_FC param)
# # 'savedVM_v2/models/CPSC_BaselineWithMultiHeadAttention_uBCE_F1/tune_run_1'
# path_to_tune = 'savedVM_v2/models/CPSC_BaselineWithMultiHeadAttention_uBCE_F1/tune_run_2_additional_FC_discarded'
# # Attention! The order of the hyper_params must match the one of params.json; it can differ from the order in train.py!
# hyper_params = ['dropout_attention', 'gru_units', 'heads']
# integer_vals = ['gru_units', 'heads', 'Epochs']
# single_precision = ['dropout_attention']
# desired_col_order = ['dropout_attention', 'heads', 'gru_units',
#                      'SNR', 'AF', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'VEB', 'STD', 'STE',
#                      'W-AVG_F1', 'W-AVG_ROC', 'W-AVG_Acc', 'MR', 'Epochs']


# 'savedVM_v2/models/FinalModel/experiment_3_rerun_withFC'
# 'savedVM_v2/models/FinalModel/experiment_3_rerun_noFC'
# Old runs:
# 'savedVM_v2/models/FinalModel/experiment_3_with_FC'
# 'savedVM_v2/models/FinalModel/experiment_3_ohne_FC_preAct_all'

# path_to_tune = 'savedVM_v2/models/FinalModel/experiment_3_rerun_noFC'
# hyper_params = ["discard_FC_before_MH", "down_sample", "dropout_attention", "gru_units", "heads",
#                 "norm_before_act", "norm_pos", "norm_type", "pos_skip",
#                 "use_pre_activation_design", "use_pre_conv", "vary_channels"]
# integer_vals = ['gru_units', 'heads', 'Epochs']
# single_precision = ['dropout_attention']
# desired_col_order = None
# #  Discard FC before MH is either True or False for all runs (depending on VM), PreAct always True
# desired_col_order = ["dropout_attention", "heads", "gru_units",
#                      'SNR', 'AF', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'VEB', 'STD', 'STE',
#                      'W-AVG_F1', 'W-AVG_ROC', 'W-AVG_Acc', 'MR', 'Epochs']


# path_to_tune = 'savedVM/models/FinalModel_MACRO_ParamStudy/0829_141413_ml_bs16_macroF1'
# hyper_params = ["discard_FC_before_MH", "down_sample", "dropout_attention", "gru_units", "heads",
#                 "norm_before_act", "norm_pos", "norm_type", "pos_skip", "pre_conv_kernel",
#                 "use_pre_activation_design", "use_pre_conv", "vary_channels"]
# integer_vals = ['gru_units', 'heads', 'Epochs', 'Params']
# single_precision = ['dropout_attention']
# desired_col_order = ["discard_FC_before_MH", "heads", "gru_units",
#                      'SNR', 'AF', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'VEB', 'STD', 'STE', 'm-F1', 'm-ROC-AUC', 'm-Acc',
#                      'W-AVG_F1', 'W-AVG_ROC', 'W-AVG_Acc', 'MR', 'Epochs', 'Params']


# path_to_tune = 'savedVM/models/FinalModel_MACRO_MultiBranch_ParamStudy/0831_110846_ml_bs16'
# hyper_params = [ "branchNet_gru_units", "branchNet_heads", "discard_FC_before_MH", "first_conv_reduction_kernel_size",
#                  "multi_branch_heads", "second_conv_reduction_kernel_size", "third_conv_reduction_kernel_size",
#                  "vary_channels_lighter_version"]
# integer_vals = ['branchNet_gru_units', 'branchNet_heads', "first_conv_reduction_kernel_size", "multi_branch_heads",
#                 "second_conv_reduction_kernel_size", "third_conv_reduction_kernel_size", 'Epochs', 'Params']
# single_precision = []
# desired_col_order = ["multi_branch_heads", "first_conv_reduction_kernel_size", "second_conv_reduction_kernel_size",
#                      "third_conv_reduction_kernel_size",
#                      'SNR', 'AF', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'VEB', 'STD', 'STE', 'm-F1', 'm-ROC-AUC', 'm-Acc',
#                      'W-AVG_F1', 'W-AVG_ROC', 'W-AVG_Acc', 'MR', 'Epochs', 'Params']


include_class_wise_f1 = True
include_weighted_avg = True
include_macro_avg = True
include_CPSC_scores = False
include_details = True  # epochs, params
use_abbrevs = True

cols_class_wise_f1 = ['SNR', 'AF', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'VEB', 'STD', 'STE']
cols_weighted_avg = ['W-AVG_F1', 'W-AVG_ROC', 'W-AVG_Acc']
cols_macro_avg = ['m-F1', 'm-ROC-AUC', 'm-Acc']
cols_CPSC_scores = ['cpsc_F1', 'cpsc_Faf', 'cpsc_Fblock', 'cpsc_Fpc', 'cpsc_Fst']
col_details = ['Epochs', 'Params']

max_columns = []
df_columns = hyper_params
if include_class_wise_f1:
    max_columns = max_columns + cols_class_wise_f1
    df_columns = df_columns + cols_class_wise_f1
if include_weighted_avg:
    df_columns = df_columns + cols_weighted_avg
    max_columns = max_columns + cols_weighted_avg
if include_macro_avg:
    max_columns = max_columns + cols_macro_avg
    df_columns = df_columns + cols_macro_avg
df_columns = df_columns + ['MR']
max_columns = max_columns + ['MR']
if include_CPSC_scores:
    max_columns = max_columns + cols_CPSC_scores
    df_columns = df_columns + cols_CPSC_scores

if include_details:
    df_summary_valid = pd.DataFrame(columns=df_columns + col_details)
else:
    df_summary_valid = pd.DataFrame(columns=df_columns)

# For the test set, we do not need the details since they are identical
df_summary_test = pd.DataFrame(columns=df_columns)


def _rename_param_value(value):
    if value == "max_pool":
        if use_abbrevs:
            return "p"
        else:
            return "pool"
    elif value == "conv":
        if use_abbrevs:
            return "c"
        else:
            return "conv"
    elif value == "not_first":
        if use_abbrevs:
            return "nf"
        else:
            return "not first"
    elif value == "not_last":
        if use_abbrevs:
            return "nl"
        else:
            return "not last"
    elif isinstance(value, bool) and value:
        if use_abbrevs:
            return "t"
        else:
            return "true"
    elif isinstance(value, bool) and not value:
        if use_abbrevs:
            return "f"
        else:
            return "false"
    elif value == "last":
        if use_abbrevs:
            return "last"
        else:
            return "last"
    else:
        return value


def _append_to_summary(path, df_summary, tune_dict, best_epoch=None, num_params=None):
    with open(os.path.join(path, "eval_class_wise.p"), "rb") as file:
        df_class_wise = pk.load(file)
    with open(os.path.join(path, "eval_single_metrics.p"), "rb") as file:
        df_single_metrics = pk.load(file)

    # Update param configuration
    row_values = [_rename_param_value(tune_dict[k]) for k in tune_dict.keys()]

    # Append F1 metrics (class-wise)
    if include_class_wise_f1:
        f1_metrics = df_class_wise.loc['f1-score'][cols_class_wise_f1].values.tolist()
        row_values = row_values + f1_metrics

    if include_weighted_avg:
        # Append the weighted AVG for F1
        row_values.append(df_class_wise.loc['f1-score']['weighted avg'])
        # Append the weighted AVG for ROC-AUC
        row_values.append(df_class_wise.loc['torch_roc_auc']['weighted avg'])
        # Append the weighted AVG for Acc
        row_values.append(df_class_wise.loc['torch_accuracy']['weighted avg'])

    if include_macro_avg:
        # Append the macro AVG for F1
        row_values.append(df_class_wise.loc['f1-score']['macro avg'])
        # Append the macro ROC-AUC
        row_values.append(df_class_wise.loc['torch_roc_auc']['macro avg'])
        # Append the macro for Acc
        row_values.append(df_class_wise.loc['torch_accuracy']['macro avg'])

    # Append the subset acc (=MR)
    row_values = row_values + df_single_metrics['sk_subset_accuracy'].values.tolist()

    if include_CPSC_scores:
        # Append single metrics
        row_values = row_values + df_single_metrics[cols_CPSC_scores].values.tolist()[0]

    if include_details:
        # Append the number of epochs
        if best_epoch is not None:
            row_values = row_values + [best_epoch]
        if num_params is not None:
            row_values = row_values + [num_params]

    # Add the row to the summary dataframe
    df_summary.loc[len(df_summary)] = row_values


# Loop through the runs and append the tuning parameters as well as the resulting metrics to the summary df
def _extract_num_params(tune_path):
    with open(os.path.join(tune_path.replace("/models", "/log"), "debug.log"), "r") as file:
        with mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as mmapped_file:
            search_position = mmapped_file.find(b"Trainable parameters:")

            if search_position != -1:
                # Find the start and end positions of the line containing the search_text
                line_start = mmapped_file.rfind(b'\n', 0, search_position) + 1
                line_end = mmapped_file.find(b'\n', search_position) + 1

                # Read the line from the mmap
                mmapped_file.seek(line_start)
                line = mmapped_file.read(line_end - line_start).decode()
                return line.replace("Trainable parameters: ", "").replace("\n", "")
            else:
                return "N/A"


for tune_run in os.listdir(path_to_tune):
    tune_path = os.path.join(path_to_tune, tune_run)
    if os.path.isdir(tune_path):
        num_params = _extract_num_params(tune_path)

        with open(os.path.join(tune_path, "progress.csv"), "r") as file:
            with open(os.path.join(tune_path,"..","config.json")) as config_file:
                config = json.load(config_file)
                try:
                    early_stop = config.get('trainer').get('early_stop')
                    # Example: Not improved for 20 epochs -> best was 21 epochs earlier
                    # First line is header, so subtract 22
                    best_epoch = sum(1 for line in file) - (early_stop + 2)

                    with open(os.path.join(tune_path, "params.json"), "r") as file:
                        tune_dict = json.load(file)
                    # Validation
                    path = os.path.join(tune_path, "valid_output")
                    _append_to_summary(path, df_summary_valid, tune_dict, best_epoch, num_params)

                    # Test
                    path = os.path.join(tune_path, "test_output")
                    _append_to_summary(path, df_summary_test, tune_dict)

                except AttributeError as e:
                    raise Exception ("The provided config does not contain an early stopping setting."
                          "Re-check the summarize_tune_run.py script to handle this! (cf. line 295)") from e


# Parse integer values as ints
for col in integer_vals:
    df_summary_valid[col] = df_summary_valid[col].apply(int)
    if col != 'Epochs' and col != "Params":
        df_summary_test[col] = df_summary_test[col].apply(int)

# Format single-precision floats
for col in single_precision:
    df_summary_valid[col] = pd.Series(["{0:.1f}".format(val) for val in df_summary_valid[col]],
                                      index=df_summary_valid.index)
    df_summary_test[col] = pd.Series(["{0:.1f}".format(val) for val in df_summary_test[col]],
                                     index=df_summary_test.index)

# Reorder the columns of the dataframe to match the one used in the thesis
if desired_col_order is not None:
    df_summary_valid_reordered = df_summary_valid[desired_col_order]
    # omit epochs and params for test set
    if "Epochs" in desired_col_order:
        desired_col_order.remove("Epochs")
    if "Params" in desired_col_order:
        desired_col_order.remove("Params")
    df_summary_test_reordered = df_summary_test[desired_col_order]
else:
    df_summary_valid_reordered = df_summary_valid
    df_summary_test_reordered = df_summary_test

# Sort the rows by the main metrics
order_by_cols = ['m-F1', 'MR',
                 'W-AVG_F1']  # MA ['W-AVG_F1', 'W-AVG_ROC', 'MR', 'W-AVG_Acc']  # ['m-F1', 'CPCS_F1', 'W-AVG_F1']
# Round before sorting
df_summary_valid_reordered = df_summary_valid_reordered.round(3)
df_summary_test_reordered = df_summary_test_reordered.round(3)
df_summary_valid_reordered = df_summary_valid_reordered.sort_values(by=order_by_cols, inplace=False,
                                                                    ascending=[False for i in
                                                                               range(0, len(order_by_cols))])
df_summary_test_reordered = df_summary_test_reordered.sort_values(by=order_by_cols, inplace=False,
                                                                  ascending=[False for i in
                                                                             range(0, len(order_by_cols))])

# Write the results to files
with open(os.path.join(path_to_tune, 'summary_valid.p'), 'wb') as file:
    pk.dump(df_summary_valid_reordered, file)

with open(os.path.join(path_to_tune, 'summary_test.p'), 'wb') as file:
    pk.dump(df_summary_test_reordered, file)

fmts_max_valid = {column: partial(_bold_formatter, value=df_summary_valid_reordered[column].max(), num_decimals=3)
                  for column in max_columns}
fmts_max_test = {column: partial(_bold_formatter, value=df_summary_test_reordered[column].max(), num_decimals=3) for
                 column in max_columns}
with open(os.path.join(path_to_tune, 'summary_results.tex'), 'w') as file:
    file.write("Validation Summary:\n\n")
    df_summary_valid_reordered.to_latex(buf=file, index=False, float_format="{:0.3f}".format, formatters=fmts_max_valid,
                                        escape=False)
    file.write("\n\n\nTest Summary:\n\n")
    df_summary_test_reordered.to_latex(buf=file, index=False, float_format="{:0.3f}".format, formatters=fmts_max_test,
                                       escape=False)

print("Done")
