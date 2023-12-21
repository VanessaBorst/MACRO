import os
import pickle
import pandas as pd

# 'savedVM_v2/models/FinalModel/cross_vals/cross_validation_rerun_withFC_0.3_32_16'
# 'savedVM_v2/models/FinalModel/cross_vals/cross_validation_rerun_withFC_0.3_32_16_finetune_bayesian'
# 'savedVM_v2/models/FinalModel/cross_vals/cross_validation_rerun_noFC_0.2_24_8'
# 'savedVM_v2/models/FinalModel/cross_vals/cross_validation_rerun_withFC_0.2_32_16_finetune_BO'

# 'savedVM_v2/models/FinalModel/cross_vals_sqrtT/final_cross_validation_sqrtT'
# 'savedVM_v2/models/FinalModel/cross_vals_sqrtT/cross_validation_withFC_0.4_24_32_finetune_BO'
# path = 'savedVM_v2/models/FinalModel/cross_vals/cross_validation_rerun_withFC_0.3_32_16_finetune_bayesian'
path = 'savedVM/models/BaselineWithMultiHeadAttention_CV/1219_143021_ml_bs64_withFC_0.3_12_24_mean_query'
thresholds_active=False
include_weighted_avg=False
include_at_least_weighted_F1 = True

path_class_wise = path + '/test_results_class_wise.p'
path_single_metrics = path + '/test_results_single_metrics.p'
if thresholds_active:
    col_name_acc = 'torch_acc'
else:
    col_name_acc = 'torch_accuracy'

with open(path_class_wise, 'rb') as file:
    df_class_wise = pickle.load(file)

with open(path_single_metrics, 'rb') as file:
    df_single_metrics = pickle.load(file)

df_class_wise = df_class_wise.astype('float64')
df_class_wise = df_class_wise.round(3)
df_single_metrics = df_single_metrics.astype('float64')
df_single_metrics = df_single_metrics.round(3)

# Create a table with class-wise F1 per Fold, the avg metrics, the subset accuracy and the mean values
if include_weighted_avg:
    df_results = pd.DataFrame(columns=['Fold', 'SNR', 'AF', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'VEB', 'STD', 'STE',
                                   'W-AVG_F1', 'W-AVG_ROC', 'W-AVG_Acc', 'm-AVG_F1', 'm-AVG_ROC', 'm-AVG_Acc', 'MR'])
elif include_at_least_weighted_F1:
    df_results = pd.DataFrame(columns=['Fold', 'SNR', 'AF', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'VEB', 'STD', 'STE',
                                       'W-AVG_F1', 'm-AVG_F1', 'm-AVG_ROC', 'm-AVG_Acc', 'MR'])
else:
    df_results = pd.DataFrame(columns=['Fold', 'SNR', 'AF', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'VEB', 'STD', 'STE',
                                       'm-AVG_F1', 'm-AVG_ROC', 'm-AVG_Acc', 'MR'])

for fold in range(1, 11):
    row_values = []

    # Append the fold id
    row_values.append(fold)

    # Append F1 metrics (class-wise)
    f1_metrics =df_class_wise.loc[('fold_' + str(fold), 'f1-score')].iloc[0:9].values.tolist()
    row_values = row_values + f1_metrics

    if include_weighted_avg:
        # Append the weighted AVG for F1
        row_values.append(df_class_wise.loc[('fold_' + str(fold), 'f1-score')]['weighted avg'])
        # Append the weighted AVG for ROC-AUC
        row_values.append(df_class_wise.loc[('fold_' + str(fold), 'torch_roc_auc')]['weighted avg'])
        # Append the weighted AVG for Acc
        row_values.append(df_class_wise.loc[('fold_' + str(fold), col_name_acc)]['weighted avg'])
    elif include_at_least_weighted_F1:
        # Append the weighted AVG for F1
        row_values.append(df_class_wise.loc[('fold_' + str(fold), 'f1-score')]['weighted avg'])

    # Append the macro AVG for F1
    row_values.append(df_class_wise.loc[('fold_' + str(fold), 'f1-score')]['macro avg'])
    # Append the macro ROC-AUC
    row_values.append(df_class_wise.loc[('fold_' + str(fold), 'torch_roc_auc')]['macro avg'])
    # Append the macro for Acc
    row_values.append(df_class_wise.loc[('fold_' + str(fold), col_name_acc)]['macro avg'])
    # Append the subset acc (=MR)
    row_values.append(df_single_metrics.loc['fold_' + str(fold)]['sk_subset_accuracy'])

    # Add the row to the summary dataframe
    df_results.loc[len(df_results)] = row_values

df_results["Fold"] = pd.to_numeric(df_results["Fold"], downcast='integer')
# Append another row containing the mean
df_results.loc['mean'] = df_results.mean()
df_results = df_results.round(3)

# Write result to latex
with open(os.path.join(path, 'cross_valid_runs_summary.tex'), 'w') as file:
    df_results.to_latex(buf=file, index=False, float_format="{:0.3f}".format, escape=False)
    file.write("\n\n\nTest Summary:\n\n")

print("Done")
