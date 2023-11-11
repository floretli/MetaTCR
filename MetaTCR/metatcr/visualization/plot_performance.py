import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import sem
import os

def plot_validation_auc(fold_files, save_path, true_label_col='true_label', pred_prob_col='pred_prob'):
    plt.figure(figsize=(6, 6))
    ax = plt.gca()
    all_tprs = []
    all_fprs = []
    all_aucs = []

    for fold_file in fold_files:
        data = pd.read_csv(fold_file)
        fpr, tpr, _ = roc_curve(data[true_label_col], data[pred_prob_col])
        auc_score = auc(fpr, tpr)
        all_fprs.append(fpr)
        all_tprs.append(tpr)
        all_aucs.append(auc_score)

        # Plot individual fold ROC curve
        ax.plot(1 - fpr, tpr, lw=1, alpha=0.3, label=f'ROC fold {len(all_aucs)} (AUC = {auc_score:.2f})')

    n_folds = len(fold_files)

    fpr_mean = np.linspace(0, 1, 100)
    interp_tprs = []

    for i in range(n_folds):
        fpr = all_fprs[i]
        tpr = all_tprs[i]
        interp_tpr = np.interp(fpr_mean, fpr, tpr)
        interp_tpr[0] = 0.0
        interp_tprs.append(interp_tpr)

    tpr_mean = np.mean(interp_tprs, axis=0)
    tpr_mean[-1] = 1.0
    tpr_std = 2 * np.std(interp_tprs, axis=0)
    tpr_upper = np.clip(tpr_mean + tpr_std, 0, 1)
    tpr_lower = tpr_mean - tpr_std
    mean_auc = np.mean(all_aucs)
    std_auc = np.std(all_aucs)

    ax.plot(
        1 - fpr_mean,
        tpr_mean,
        label="Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )

    ax.fill_between(
        1 - fpr_mean,
        tpr_lower,
        tpr_upper,
        color="grey",
        alpha=0.1,
        label=r"$\pm$ 1 std. dev.",
    )

    ax.yaxis.set_major_formatter(plt.FuncFormatter('{0:.0%}'.format))
    ax.xaxis.set_major_formatter(plt.FuncFormatter('{0:.0%}'.format))
    ax.invert_xaxis()
    ax.set_ylabel("Sensitivity", fontsize=10)
    ax.set_xlabel("Specificity", fontsize=10)
    ax.plot([0, 1], [1, 0],'r--', lw=1)

    ax.set_xlim([1.05, -0.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_title("Validation AUC-ROC")
    ax.legend(loc="lower right", frameon=False)

    plt.savefig(os.path.join(save_path, 'validation_roc.png'), dpi=300)

    # plt.show()

def find_val_predictions_files(save_path, file_suffix='val_predictions.csv'):
    file_list = []
    for root, _, files in os.walk(save_path):
        for file in files:
            if file.endswith(file_suffix):
                file_list.append(os.path.join(root, file))
    return file_list

if __name__ == "__main__":
    save_path = "../results/clf_models/"
    fold_files = find_val_predictions_files(save_path)

    print(fold_files)

    plot_validation_auc(fold_files)