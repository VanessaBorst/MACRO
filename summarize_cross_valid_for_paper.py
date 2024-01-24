import os
import pickle
import pandas as pd

base_path = 'savedVM/models/'
model_paths = {'Baseline': 'BaselineModel_CV/0116_145000_ml_bs64_250Hz_60s',
               'Final_Model': 'FinalModel_MACRO_CV/0123_171857_ml_bs64_noFC-0.2-6-12_entmax15',
               'Multibranch': ''}

model_names = list(model_paths.keys())
df_results_paper = pd.DataFrame(columns=['Type',
                                         f'{model_names[0]}_F1', f'{model_names[0]}_AUC', f'{model_names[0]}_Acc',
                                         f'{model_names[1]}_F1', f'{model_names[1]}_AUC', f'{model_names[1]}_Acc',
                                         f'{model_names[2]}_F1', f'{model_names[2]}_AUC', f'{model_names[2]}_Acc'])

df_results_paper['Type'] = ['SNR', 'AF', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'VEB', 'STD', 'STE', 'm-AVG_F1', 'W-AVG_F1']

for model_name, model_path in model_paths.items():
    path_class_wise = os.path.join(base_path, model_path, 'test_results_class_wise.p')
    path_single_metrics = os.path.join(base_path, model_path, 'test_results_single_metrics.p')
    col_name_acc = 'torch_accuracy'
    with open(path_class_wise, 'rb') as file:
        df_class_wise = pickle.load(file)

    df_class_wise = df_class_wise.astype('float64')
    df_class_wise = df_class_wise.round(3)

    # Create tables with class-wise F1, AUC and Acc per Fold as well as the corresponding averages (macro and weighted)
    df_results_F1 = pd.DataFrame(columns=['Fold', 'SNR', 'AF', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'VEB', 'STD', 'STE',
                                          'W-AVG_F1', 'm-AVG_F1'])

    df_results_AUC = pd.DataFrame(columns=['Fold', 'SNR', 'AF', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'VEB', 'STD', 'STE',
                                           'W-AVG_AUC', 'm-AVG_AUC'])

    df_results_acc = pd.DataFrame(columns=['Fold', 'SNR', 'AF', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'VEB', 'STD', 'STE',
                                           'W-AVG_Acc', 'm-AVG_Acc'])

    for fold in range(1, 11):
        row_values_f1 = row_values_AUC = row_values_acc = [fold]

        # Append class-wise metrics (class-wise)
        f1_metrics = df_class_wise.loc[('fold_' + str(fold), 'f1-score')].iloc[0:9].values.tolist()
        row_values_f1 = row_values_f1 + f1_metrics

        AUC_metrics = df_class_wise.loc[('fold_' + str(fold), 'torch_roc_auc')].iloc[0:9].values.tolist()
        row_values_AUC = row_values_AUC + AUC_metrics

        acc_metrics = df_class_wise.loc[('fold_' + str(fold), col_name_acc)].iloc[0:9].values.tolist()
        row_values_acc = row_values_acc + acc_metrics

        # Append the weighted AVGs
        row_values_f1.append(df_class_wise.loc[('fold_' + str(fold), 'f1-score')]['weighted avg'])
        row_values_AUC.append(df_class_wise.loc[('fold_' + str(fold), 'torch_roc_auc')]['weighted avg'])
        row_values_acc.append(df_class_wise.loc[('fold_' + str(fold), col_name_acc)]['weighted avg'])

        # Append the macro AVGs
        row_values_f1.append(df_class_wise.loc[('fold_' + str(fold), 'f1-score')]['macro avg'])
        # Append the macro ROC-AUC
        row_values_AUC.append(df_class_wise.loc[('fold_' + str(fold), 'torch_roc_auc')]['macro avg'])
        # Append the macro for Acc
        row_values_acc.append(df_class_wise.loc[('fold_' + str(fold), col_name_acc)]['macro avg'])

        # Add the row to the summary dataframes
        df_results_F1.loc[len(df_results_F1)] = row_values_f1
        df_results_AUC.loc[len(df_results_AUC)] = row_values_AUC
        df_results_acc.loc[len(df_results_acc)] = row_values_acc

    # Append another row containing the mean
    df_results_F1.loc['mean'] = df_results_F1.mean()
    df_results_F1 = df_results_F1.round(3)
    df_results_AUC.loc['mean'] = df_results_AUC.mean()
    df_results_AUC = df_results_AUC.round(3)
    df_results_acc.loc['mean'] = df_results_acc.mean()
    df_results_acc = df_results_acc.round(3)

    # Add the average metrics across all folds to the final result dataframe
    df_results_paper[f'{model_name}_F1'] = [df_results_F1.loc['mean']['SNR'],
                                            df_results_F1.loc['mean']['AF'],
                                            df_results_F1.loc['mean']['IAVB'],
                                            df_results_F1.loc['mean']['LBBB'],
                                            df_results_F1.loc['mean']['RBBB'],
                                            df_results_F1.loc['mean']['PAC'],
                                            df_results_F1.loc['mean']['VEB'],
                                            df_results_F1.loc['mean']['STD'],
                                            df_results_F1.loc['mean']['STE'],
                                            df_results_F1.loc['mean']['m-AVG_F1'],
                                            df_results_F1.loc['mean']['W-AVG_F1']]

    df_results_paper[f'{model_name}_AUC'] = [df_results_AUC.loc['mean']['SNR'],
                                             df_results_AUC.loc['mean']['AF'],
                                             df_results_AUC.loc['mean']['IAVB'],
                                             df_results_AUC.loc['mean']['LBBB'],
                                             df_results_AUC.loc['mean']['RBBB'],
                                             df_results_AUC.loc['mean']['PAC'],
                                             df_results_AUC.loc['mean']['VEB'],
                                             df_results_AUC.loc['mean']['STD'],
                                             df_results_AUC.loc['mean']['STE'],
                                             df_results_AUC.loc['mean']['m-AVG_AUC'],
                                             df_results_AUC.loc['mean']['W-AVG_AUC']]

    df_results_paper[f'{model_name}_Acc'] = [df_results_acc.loc['mean']['SNR'],
                                             df_results_acc.loc['mean']['AF'],
                                             df_results_acc.loc['mean']['IAVB'],
                                             df_results_acc.loc['mean']['LBBB'],
                                             df_results_acc.loc['mean']['RBBB'],
                                             df_results_acc.loc['mean']['PAC'],
                                             df_results_acc.loc['mean']['VEB'],
                                             df_results_acc.loc['mean']['STD'],
                                             df_results_acc.loc['mean']['STE'],
                                             df_results_acc.loc['mean']['m-AVG_Acc'],
                                             df_results_acc.loc['mean']['W-AVG_Acc']]

# Write result to latex
with open(os.path.join(base_path, 'paper_cross_valid_runs_summary.tex'), 'w') as file:
    df_results_paper.to_latex(buf=file, index=False, float_format="{:0.3f}".format, escape=False)

print("Done")
