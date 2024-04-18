import os
import pickle

import pandas as pd
from matplotlib import pyplot as plt

import global_config
from utils import ensure_dir
global_config.suppress_warnings()

CONTENT = "gradient_boosting"           # Either gradient_boosting or raw_multi_branch
assert CONTENT in ["gradient_boosting", "raw_multi_branch"], "Invalid CONTENT"
include_acc = False

base_path = 'savedVM/models/'
if CONTENT == "raw_multi_branch":
    model_paths = {'Baseline': 'BaselineModel_CV/0116_145000_ml_bs64_250Hz_60s',
                   'Final_Model': 'FinalModel_MACRO_CV/0123_171857_ml_bs64_noFC-0.2-6-12_entmax15',
                   'Multibranch': 'Multibranch_MACRO_CV/0201_104057_ml_bs64convRedBlock_333_0.2_6_false_0.2_24'}
else:
    model_paths = {# 'MB': 'Multibranch_MACRO_CV/0201_104057_ml_bs64convRedBlock_333_0.2_6_false_0.2_24',
                   'MB_GB_ind_red': 'Multibranch_MACRO_CV/0201_104057_ml_bs64convRedBlock_333_0.2_6_false_0.2_24/ML models/gradient_boosting_individual_features_reduced',
                   'MB_GB_ind': 'Multibranch_MACRO_CV/0201_104057_ml_bs64convRedBlock_333_0.2_6_false_0.2_24/ML models/gradient_boosting_individual_features',
                   'MB_GB_all': 'Multibranch_MACRO_CV/0201_104057_ml_bs64convRedBlock_333_0.2_6_false_0.2_24/ML models/gradient_boosting_BCE_final'
                   }


# The box extends from the Q1 to Q3 quartile values of the data, with a line at the median (Q2).
# Q1 = 25% percentile, Q2 = 50% percentile, Q3 = 75% percentile
# The whiskers extend from the edges of box to show the range of the data.
# The position of the whiskers is set by default to 1.5*IQR (IQR = Q3 - Q1) from the edges of the box.
# Outlier points are those past the end of the whiskers.
def create_and_save_boxplot_for_single_model(df, model_name, metric_name, metric_full_name):
    x_labels = [w.replace(f'W-AVG_{metric_name}', 'w-AVG').replace(f'm-AVG_{metric_name}', 'm-AVG')
                for w in df.columns[1:].values.tolist()]
    ax = df.drop(columns=["Fold"]).boxplot(meanline=True, showmeans=True)  # , figsize=(15,8)
    ax.set_title(f'{model_name} - {metric_name}')
    ax.set_xlabel('Class name')
    ax.set_ylabel(metric_full_name)
    ax.set_xticklabels(x_labels, rotation=30)
    plt.tight_layout()
    save_dir = os.path.join('../figures', 'box_plots', f'{CONTENT}')
    ensure_dir(save_dir)
    plt.savefig(os.path.join(save_dir, f'{model_name} - {metric_name}.pdf'), format="pdf", bbox_inches="tight")
    plt.show()
    # Clear the current axis
    ax.clear()


def create_and_save_boxplot_for_macro_avgs(df, model_names, metric_name, title, x_label, y_label, filename):

    column_names = [f'{name}_{metric_name}' for name in model_names]
    x_labels = [
        w.replace(f'_{metric_name}', '').replace('Final_Model', 'MACRO').replace('Multibranch', 'Multibranch \n MACRO')
        for w in column_names]

    ax = df[column_names].boxplot(meanline=True, showmeans=True)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xticklabels(x_labels, rotation=30)
    plt.tight_layout()
    save_dir = os.path.join('../figures', 'box_plots', f'{CONTENT}')
    ensure_dir(save_dir)
    plt.savefig(os.path.join(save_dir, filename), format="pdf", bbox_inches="tight")
    plt.show()
    # Clear the current axis
    ax.clear()


model_names = list(model_paths.keys())
metrics = ['F1', 'AUC', 'Acc'] if include_acc else ['F1', 'AUC']

# Create column names and then df_results_paper
df_result_columns = ['Type'] + [f'{name}_{metric}' for name in model_names for metric in metrics]
df_results_paper = pd.DataFrame(columns=df_result_columns)
df_results_paper['Type'] = ['SNR', 'AF', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'VEB', 'STD', 'STE', 'm-AVG', 'W-AVG']

# Create column names and then df_results_macro_averages
macro_average_columns = [f'{name}_{metric}' for name in model_names for metric in metrics ]
 # Assuming 10 rows
df_results_macro_averages = pd.DataFrame(columns=macro_average_columns, index=range(10))

for model_name, model_path in model_paths.items():
    path_class_wise = os.path.join(base_path, model_path, 'test_results_class_wise.p')
    path_single_metrics = os.path.join(base_path, model_path, 'test_results_single_metrics.p')

    with open(path_class_wise, 'rb') as file:
        df_class_wise = pickle.load(file)

    df_class_wise = df_class_wise.astype('float64')
    df_class_wise = df_class_wise.round(3)

    # Create tables with class-wise F1, AUC and Acc per Fold as well as the corresponding averages (macro and weighted)
    df_results_F1 = pd.DataFrame(columns=['Fold', 'SNR', 'AF', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'VEB', 'STD', 'STE',
                                          'W-AVG_F1', 'm-AVG_F1'])

    df_results_AUC = pd.DataFrame(columns=['Fold', 'SNR', 'AF', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'VEB', 'STD', 'STE',
                                           'W-AVG_AUC', 'm-AVG_AUC'])

    if include_acc:
        df_results_acc = pd.DataFrame(columns=['Fold', 'SNR', 'AF', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'VEB', 'STD', 'STE',
                                               'W-AVG_Acc', 'm-AVG_Acc'])
        col_name_acc = 'torch_accuracy'

    for fold in range(1, 11):
        row_values_f1 = row_values_AUC = [fold]
        if include_acc:
            row_values_acc = [fold]

        # Append class-wise metrics (class-wise)
        f1_metrics = df_class_wise.loc[('fold_' + str(fold), 'f1-score')].iloc[0:9].values.tolist()
        row_values_f1 = row_values_f1 + f1_metrics

        AUC_metrics = df_class_wise.loc[('fold_' + str(fold), 'torch_roc_auc')].iloc[0:9].values.tolist()
        row_values_AUC = row_values_AUC + AUC_metrics

        # Append the weighted AVGs
        row_values_f1.append(df_class_wise.loc[('fold_' + str(fold), 'f1-score')]['weighted avg'])
        row_values_AUC.append(df_class_wise.loc[('fold_' + str(fold), 'torch_roc_auc')]['weighted avg'])

        # Append the macro AVGs
        row_values_f1.append(df_class_wise.loc[('fold_' + str(fold), 'f1-score')]['macro avg'])
        # Append the macro ROC-AUC
        row_values_AUC.append(df_class_wise.loc[('fold_' + str(fold), 'torch_roc_auc')]['macro avg'])

        # Add the row to the summary dataframes
        df_results_F1.loc[len(df_results_F1)] = row_values_f1
        df_results_AUC.loc[len(df_results_AUC)] = row_values_AUC

        if include_acc:
            acc_metrics = df_class_wise.loc[('fold_' + str(fold), col_name_acc)].iloc[0:9].values.tolist()
            row_values_acc = row_values_acc + acc_metrics
            row_values_acc.append(df_class_wise.loc[('fold_' + str(fold), col_name_acc)]['weighted avg'])
            # Append the macro for Acc
            row_values_acc.append(df_class_wise.loc[('fold_' + str(fold), col_name_acc)]['macro avg'])
            df_results_acc.loc[len(df_results_acc)] = row_values_acc

    # Visualize the results
    # Boxplot for F1
    create_and_save_boxplot_for_single_model(df=df_results_F1, model_name=model_name,
                                             metric_name='F1', metric_full_name='F1 score')
    # Boxplot for AUC
    create_and_save_boxplot_for_single_model(df=df_results_AUC, model_name=model_name,
                                             metric_name='AUC', metric_full_name='AUC score')

    if include_acc:
        # Boxplot for Acc
        create_and_save_boxplot_for_single_model(df=df_results_acc, model_name=model_name,
                                                 metric_name='Acc', metric_full_name='Accuracy')

    # Put the macro averages of the current model into the dataframe
    df_results_macro_averages[f'{model_name}_F1'] = df_results_F1['m-AVG_F1']
    df_results_macro_averages[f'{model_name}_AUC'] = df_results_AUC['m-AVG_AUC']
    if include_acc:
        df_results_macro_averages[f'{model_name}_Acc'] = df_results_acc['m-AVG_Acc']



    # Get statistics
    # df_F1_statistics = df_results_F1.describe().drop(columns=["Fold"])

    # Append another row containing the mean and std
    # Note:
    # np.std(au_roc_scores, ddof=1) returns the sample standard deviation as in pandas dataframes, while
    # np.std(aucs) returns slightly different results for some classes
    # It holds: df.std(ddof=0) == np.std(df) AND df.std() and np.std(ddof=1)
    # Pandas uses the unbiased estimator (N-1 in the denominator), whereas Numpy by default does not.
    # cf. https://stackoverflow.com/questions/24984178/different-std-in-pandas-vs-numpy

    #  = pd.concat([df_results_F1, df_results_F1.describe().loc[["mean", "std"]]])
    series_mean_F1 = df_results_F1.mean()
    series_std_F1 = df_results_F1.std(ddof=0)
    df_results_F1.loc["mean"] = series_mean_F1
    df_results_F1.loc["std"] = series_std_F1
    df_results_F1 = df_results_F1.round(3)

    # df_results_AUC = pd.concat([df_results_AUC, df_results_AUC.describe().loc[["mean", "std"]]])
    series_mean_AUC = df_results_AUC.mean()
    series_std_AUC = df_results_AUC.std(ddof=0)
    df_results_AUC.loc["mean"] = series_mean_AUC
    df_results_AUC.loc["std"] = series_std_AUC
    df_results_AUC = df_results_AUC.round(3)

    if include_acc:
        # df_results_acc = pd.concat([df_results_acc, df_results_acc.describe().loc[["mean", "std"]]])
        series_mean_acc = df_results_acc.mean()
        series_std_acc = df_results_acc.std(ddof=0)
        df_results_acc.loc["mean"] = series_mean_acc
        df_results_acc.loc["std"] = series_std_acc
        df_results_acc = df_results_acc.round(3)

    # Add the average metrics across all folds to the final result dataframes
    df_results_paper[f'{model_name}_F1'] = [f"{df_results_F1.loc['mean'][col]:.3f}±{df_results_F1.loc['std'][col]:.3f}"
                                            for col in ['SNR', 'AF', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'VEB', 'STD', 'STE',
                                                        'm-AVG_F1', 'W-AVG_F1']]

    df_results_paper[f'{model_name}_AUC'] = [f"{df_results_AUC.loc['mean'][col]:.3f}±{df_results_AUC.loc['std'][col]:.3f}"
                                             for col in
                                             ['SNR', 'AF', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'VEB', 'STD', 'STE',
                                              'm-AVG_AUC', 'W-AVG_AUC']]

    if include_acc:
        df_results_paper[f'{model_name}_Acc'] = [f"{df_results_acc.loc['mean'][col]:.3f}±{df_results_acc.loc['std'][col]:.3f}"
                                                 for col in
                                                 ['SNR', 'AF', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'VEB', 'STD', 'STE',
                                                  'm-AVG_Acc', 'W-AVG_Acc']]

# Visualize the macro averages across all models
# Boxplot for F1
create_and_save_boxplot_for_macro_avgs(df=df_results_macro_averages,
                                       model_names=model_names,
                                       metric_name='F1',
                                       title='Macro F1 Score',
                                       x_label='Architecture',
                                       y_label='F1 Score',
                                       filename='Macro_F1_Scores.pdf')
# Boxplot for AUC
create_and_save_boxplot_for_macro_avgs(df=df_results_macro_averages,
                                       model_names=model_names,
                                       metric_name='AUC',
                                       title='Macro AUC Score',
                                       x_label='Architecture',
                                       y_label='AUC Score',
                                       filename='Macro_AUC_Scores.pdf')
# Boxplot for Acc
if include_acc:
    create_and_save_boxplot_for_macro_avgs(df=df_results_macro_averages,
                                           model_names=model_names,
                                           metric_name='Acc',
                                           title='Macro Accuracy',
                                           x_label='Architecture',
                                           y_label='Accuracy',
                                           filename='Macro_Acc_Scores.pdf')

# Write result to latex
save_path = os.path.join(base_path, 'paper', f'{CONTENT}')
ensure_dir(save_path)
with open(os.path.join(save_path, 'paper_cross_valid_runs_summary.tex'), 'w') as file:
    df_results_paper.to_latex(buf=file, index=False, float_format="{:0.3f}".format, escape=False)

print("Done")
