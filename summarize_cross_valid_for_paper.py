import os
import pickle
import pandas as pd
from matplotlib import pyplot as plt
import scipy.stats as stats

base_path = 'savedVM/models/'
model_paths = {'Baseline': 'BaselineModel_CV/0116_145000_ml_bs64_250Hz_60s',
               'Final_Model': 'FinalModel_MACRO_CV/0123_171857_ml_bs64_noFC-0.2-6-12_entmax15',
               'Multibranch': 'Multibranch_MACRO_CV/0201_104057_ml_bs64convRedBlock_333_0.2_6_false_0.2_24'}
include_acc = False


def create_and_save_boxplot(df, model_names, metric_name, metric_cols, title, x_label, y_label, filename):
    x_labels = [
        w.replace(f'_{metric_name}', '').replace('Final_Model', 'MACRO').replace('Multibranch', 'Multibranch \n MACRO')
        for w in df.columns[metric_cols].values.tolist()]

    ax = df[[f'{model_names[0]}_{metric_name}',
             f'{model_names[1]}_{metric_name}',
             f'{model_names[2]}_{metric_name}1']].boxplot(meanline=True, showmeans=True)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xticklabels(x_labels, rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join('figures', filename), format="pdf", bbox_inches="tight")
    plt.show()


model_names = list(model_paths.keys())
df_result_columns = \
    ['Type',
     f'{model_names[0]}_F1', f'{model_names[0]}_AUC', f'{model_names[0]}_Acc',
     f'{model_names[1]}_F1', f'{model_names[1]}_AUC', f'{model_names[1]}_Acc',
     f'{model_names[2]}_F1', f'{model_names[2]}_AUC', f'{model_names[2]}_Acc'] if include_acc else \
        ['Type',
         f'{model_names[0]}_F1', f'{model_names[0]}_AUC',
         f'{model_names[1]}_F1', f'{model_names[1]}_AUC',
         f'{model_names[2]}_F1', f'{model_names[2]}_AUC']

df_results_paper = pd.DataFrame(columns=df_result_columns)

df_results_paper['Type'] = ['SNR', 'AF', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'VEB', 'STD', 'STE', 'm-AVG_F1', 'W-AVG_F1']
df_results_macro_averages = pd.DataFrame(columns=[f'{model_names[0]}_F1',
                                                  f'{model_names[1]}_F1',
                                                  f'{model_names[2]}_F1',
                                                  f'{model_names[0]}_AUC',
                                                  f'{model_names[1]}_AUC',
                                                  f'{model_names[2]}_AUC',
                                                  f'{model_names[0]}_Acc',
                                                  f'{model_names[1]}_Acc',
                                                  f'{model_names[2]}_Acc'],
                                         index=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

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

        # Append class-wise metric_cols (class-wise)
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

    # Visualize the results

    # The box extends from the Q1 to Q3 quartile values of the data, with a line at the median (Q2).
    # Q1 = 25% percentile, Q2 = 50% percentile, Q3 = 75% percentile
    # The whiskers extend from the edges of box to show the range of the data.
    # The position of the whiskers is set by default to 1.5*IQR (IQR = Q3 - Q1) from the edges of the box.
    # Outlier points are those past the end of the whiskers.

    # Boxplot for F1
    x_labels = [w.replace('W-AVG_F1', 'w-AVG').replace('m-AVG_F1', 'm-AVG')
                for w in df_results_F1.columns[1:].values.tolist()]
    ax1 = df_results_F1.drop(columns=["Fold"]).boxplot(meanline=True, showmeans=True)  # , figsize=(15,8)
    ax1.set_title(f'{model_name} - F1')
    ax1.set_xlabel('Class name')
    ax1.set_ylabel('F1 score')
    ax1.set_xticklabels(x_labels, rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join('figures', f'{model_name} - F1.pdf'), format="pdf", bbox_inches="tight")
    plt.show()

    # Boxplot for AUC
    x_labels = [w.replace('W-AVG_AUC', 'w-AVG').replace('m-AVG_AUC', 'm-AVG')
                for w in df_results_AUC.columns[1:].values.tolist()]
    ax2 = df_results_AUC.drop(columns=["Fold"]).boxplot(meanline=True, showmeans=True)  # , figsize=(15,8)
    ax2.set_title(f'{model_name} - AUC')
    ax2.set_xlabel('Class name')
    ax2.set_ylabel('AUC score')
    ax2.set_xticklabels(x_labels, rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join('figures', f'{model_name} - AUC.pdf'), format="pdf", bbox_inches="tight")
    plt.show()

    # Boxplot for Acc
    x_labels = [w.replace('W-AVG_Acc', 'w-AVG').replace('m-AVG_Acc', 'm-AVG')
                for w in df_results_acc.columns[1:].values.tolist()]
    ax3 = df_results_acc.drop(columns=["Fold"]).boxplot(meanline=True, showmeans=True)  # , figsize=(15,8)
    ax3.set_title(f'{model_name} - Acc')
    ax3.set_xlabel('Class name')
    ax3.set_ylabel('Accuracy')
    ax3.set_xticklabels(x_labels, rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join('figures', f'{model_name} - Acc.pdf'), format="pdf", bbox_inches="tight")
    plt.show()

    # Put the macro averages of the current model into the dataframe
    df_results_macro_averages[f'{model_name}_F1'] = df_results_F1['m-AVG_F1']
    df_results_macro_averages[f'{model_name}_AUC'] = df_results_AUC['m-AVG_AUC']
    df_results_macro_averages[f'{model_name}_Acc'] = df_results_acc['m-AVG_Acc']

    # Get statistics
    df_F1_statistics = df_results_F1.describe().drop(columns=["Fold"])

    # Append another row containing the mean and std
    df_results_F1 = pd.concat([df_results_F1, df_results_F1.describe().loc[["mean", "std"]]])
    # df_results_F1.loc['mean'] = df_results_F1.mean()
    df_results_F1 = df_results_F1.round(3)
    df_results_AUC = pd.concat([df_results_AUC, df_results_AUC.describe().loc[["mean", "std"]]])
    df_results_AUC = df_results_AUC.round(3)
    df_results_acc = pd.concat([df_results_acc, df_results_acc.describe().loc[["mean", "std"]]])
    df_results_acc = df_results_acc.round(3)

    # Add the average metric_cols across all folds to the final result dataframes
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
    if include_acc:
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

# TODO: Perform statistical tests to check if there is a significant difference in the mean F1 scores
#  among the three architectures -> not yet proved and working! (Feb 2024)
# # Check if there is a significant difference in the mean F1 scores among the three architectures
# df_macro_F1_summary = df_results_macro_averages[[f'{model_names[0]}_F1',
#                                                  f'{model_names[1]}_F1',
#                                                  f'{model_names[2]}_F1']]
# # Perform ANOVA
# anova_result = stats.f_oneway(df_macro_F1_summary[f'{model_names[0]}_F1'],
#                               df_macro_F1_summary[f'{model_names[1]}_F1'],
#                               df_macro_F1_summary[f'{model_names[2]}_F1'])
#
# # Check the p-value
# p_value = anova_result.pvalue
# # Decide on significance level
# alpha = 0.05
#
# # Print the result
# if p_value < alpha:
#     print("There is a significant difference in means among the architectures.")
#     # Perform post-hoc tests (e.g., Tukey's HSD or Bonferroni correction)
# else:
#     print(f"No significant difference in means (F1 scores) among the architectures. The p-value is {p_value}")

# Visualize the macro averages across all models
create_and_save_boxplot(df=df_results_macro_averages,
                        model_names=model_names,
                        metric_name='F1',
                        metric_cols=[0, 1, 2],
                        title='Macro F1 Score',
                        x_label='Architecture',
                        y_label='F1 Score',
                        filename='Macro_F1_Scores.pdf')
create_and_save_boxplot(df=df_results_macro_averages,
                        model_names=model_names,
                        metric_name='AUC',
                        metric_cols=[3, 4, 5],
                        title='Macro AUC Score',
                        x_label='Architecture',
                        y_label='AUC Score',
                        filename='Macro_AUC_Scores.pdf')
create_and_save_boxplot(df=df_results_macro_averages,
                        model_names=model_names,
                        metric_name='Acc',
                        metric_cols=[6, 7, 8],
                        title='Macro Accuracy',
                        x_label='Architecture',
                        y_label='Accuracy',
                        filename='Macro_Acc_Scores.pdf')
# Boxplot for F1
x_labels = [w.replace('_F1', '').replace('Final_Model', 'MACRO').replace('Multibranch', 'Multibranch \n MACRO')
            for w in df_results_macro_averages.columns[0:3].values.tolist()]
ax4 = df_results_macro_averages[[f'{model_names[0]}_F1',
                                 f'{model_names[1]}_F1',
                                 f'{model_names[2]}_F1']].boxplot(meanline=True, showmeans=True)  # , figsize=(15,8)
ax4.set_title(f'Macro F1 Score')
ax4.set_xlabel('Architecture')
ax4.set_ylabel('F1 Score')
ax4.set_xticklabels(x_labels, rotation=30)
plt.tight_layout()
plt.savefig(os.path.join('figures', 'Macro F1 Scores.pdf'), format="pdf", bbox_inches="tight")
plt.show()

# Boxplot for AUC
x_labels = [w.replace('_AUC', '').replace('Final_Model', 'MACRO').replace('Multibranch', 'Multibranch \n MACRO')
            for w in df_results_macro_averages.columns[3:6].values.tolist()]
ax5 = df_results_macro_averages[[f'{model_names[0]}_AUC',
                                 f'{model_names[1]}_AUC',
                                 f'{model_names[2]}_AUC']].boxplot(meanline=True, showmeans=True)  # , figsize=(15,8)
ax5.set_title(f'Macro AUC Score')
ax5.set_xlabel('Architecture')
ax5.set_ylabel('AUC Score')
ax5.set_xticklabels(x_labels, rotation=30)
plt.tight_layout()
plt.savefig(os.path.join('figures', 'Macro AUC Scores.pdf'), format="pdf", bbox_inches="tight")
plt.show()

# Boxplot for Acc
x_labels = [w.replace('_Acc', '').replace('Final_Model', 'MACRO').replace('Multibranch', 'Multibranch \n MACRO')
            for w in df_results_macro_averages.columns[6:9].values.tolist()]
ax6 = df_results_macro_averages[[f'{model_names[0]}_Acc',
                                 f'{model_names[1]}_Acc',
                                 f'{model_names[2]}_Acc']].boxplot(meanline=True, showmeans=True)  # , figsize=(15,8)
ax6.set_title(f'Macro Accuracy')
ax6.set_xlabel('Architecture')
ax6.set_ylabel('Accuracy')
ax6.set_xticklabels(x_labels, rotation=30)
plt.tight_layout()
plt.savefig(os.path.join('figures', 'Macro Acc Scores.pdf'), format="pdf", bbox_inches="tight")
plt.show()

# Write result to latex
with open(os.path.join(base_path, 'paper_cross_valid_runs_summary.tex'), 'w') as file:
    df_results_paper.to_latex(buf=file, index=False, float_format="{:0.3f}".format, escape=False)

print("Done")
