#!/usr/bin/env python3
import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    recall_score,
    precision_score,
    f1_score,
)
from sklearn.calibration import calibration_curve

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

##############################
# 1.  Helper Functions
##############################

def compute_eer(fpr, tpr):
    """
    Compute Equal Error Rate (EER) from arrays of false-positive rate and true-positive rate.
    EER is where FPR = 1 − TPR (i.e. FNR). We find the index where |FPR − (1−TPR)| is minimal,
    and interpolate if needed.
    """
    fnr = 1 - tpr
    abs_diffs = np.abs(fpr - fnr)
    idx_eer = np.nanargmin(abs_diffs)
    return (fpr[idx_eer] + fnr[idx_eer]) / 2

def bootstrap_metric(y_true, y_prob, y_pred, metric_fn, n_bootstraps=1000, seed=42):
    """
    Generic bootstrap for a metric that depends on y_true, y_prob, y_pred or subset thereof.
    metric_fn should accept (y_true_sub, y_prob_sub, y_pred_sub) and return a scalar.
    We'll resample with replacement the indices of the dataset, compute metric on each resample,
    and then return the 2.5% and 97.5% quantiles of the distribution.
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    stats = []
    for _ in range(n_bootstraps):
        idxs = rng.randint(0, n, n)         # sample with replacement
        yt = y_true[idxs]
        yp = y_prob[idxs]
        ypr = y_pred[idxs]
        try:
            m = metric_fn(yt, yp, ypr)
            stats.append(m)
        except:
            pass
    if len(stats) == 0:
        return (np.nan, np.nan)
    arr = np.array(stats)
    lower = np.percentile(arr, 2.5)
    upper = np.percentile(arr, 97.5)
    return lower, upper

def metric_accuracy(yt, yp, ypr):
    return accuracy_score(yt, ypr)

def metric_ap(yt, yp, ypr):
    if len(np.unique(yt)) < 2:
        return np.nan
    return average_precision_score(yt, yp)

def metric_auc(yt, yp, ypr):
    if len(np.unique(yt)) < 2:
        return np.nan
    return roc_auc_score(yt, yp)

def metric_eer(yt, yp, ypr):
    if len(np.unique(yt)) < 2:
        return np.nan
    fpr, tpr, _ = roc_curve(yt, yp)
    return compute_eer(fpr, tpr)

def metric_recall(yt, yp, ypr):
    return recall_score(yt, ypr, zero_division=0)

def metric_precision(yt, yp, ypr):
    return precision_score(yt, ypr, zero_division=0)

def metric_f1(yt, yp, ypr):
    return f1_score(yt, ypr, zero_division=0)

##############################
# 2.  Plotting Functions
##############################

def plot_confusion_matrix(cm, classes, output_path):
    """
    Render and save a confusion matrix (2×2) as an image.
    """
    fig, ax = plt.subplots(figsize=(4,4))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        ylabel='True label',
        xlabel='Predicted label'
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

def plot_reliability_curve(y_true, y_prob, output_path):
    """
    Plot a reliability (calibration) curve (10 bins) and save it.
    """
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy='uniform')
    fig, ax = plt.subplots(figsize=(5,5))
    ax.plot(prob_pred, prob_true, marker='o', linewidth=1, label='Calibration curve')
    ax.plot([0,1], [0,1], linestyle='--', label='Perfectly calibrated')
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Fraction of positives')
    ax.set_title('Reliability (Calibration) Curve')
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

def plot_roc_curve(y_true, y_prob, output_path, dataset_name=""):
    """
    Plot an ROC curve and save it.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_val = roc_auc_score(y_true, y_prob) if len(np.unique(y_true))>1 else np.nan
    fig, ax = plt.subplots(figsize=(5,5))
    ax.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_val:.4f})')
    ax.plot([0,1], [0,1], linestyle='--', color='gray')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve — {dataset_name}')
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

##############################
# 3.  Main: Grouping & Metrics
##############################

def main(args):
    # 1) Read CSV
    df = pd.read_csv(args.csv_path)

    # 2) Map label/response to 0/1
    def map_label(x):
        x = str(x).strip().lower()
        return 0 if x == "real" else 1

    df['y_true'] = df['label'].map(map_label)
    df['y_pred'] = df['response'].map(map_label)
    df['y_prob'] = df['y_pred'].astype(float)  # treat hard labels as 0/1 probability

    # 3) Derive “base_dataset” (strip off final “_real” or “_fake”)
    df['base_dataset'] = df['dataset'].str.rsplit('_', n=1).str[0]

    # 4) Prepare output directory
    os.makedirs(args.output_dir, exist_ok=True)
    summary_path = os.path.join(args.output_dir, "metrics_by_dataset.txt")
    fout = open(summary_path, "w")

    # 5) Group by base_dataset
    grouped = df.groupby('base_dataset')
    for dataset_name, group in grouped:
        y_true = group['y_true'].values
        y_pred = group['y_pred'].values
        y_prob = group['y_prob'].values

        # Basic metrics
        acc       = accuracy_score(y_true, y_pred)
        ap        = (average_precision_score(y_true, y_prob)
                     if len(np.unique(y_true)) > 1 else np.nan)
        auc_val   = (roc_auc_score(y_true, y_prob)
                     if len(np.unique(y_true)) > 1 else np.nan)
        if len(np.unique(y_true)) > 1:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            eer_val = compute_eer(fpr, tpr)
        else:
            fpr, tpr = np.array([0,1]), np.array([0,1])
            eer_val = np.nan

        recall_   = recall_score(y_true, y_pred, zero_division=0)
        precision_= precision_score(y_true, y_pred, zero_division=0)
        f1_       = f1_score(y_true, y_pred, zero_division=0)
        cm        = confusion_matrix(y_true, y_pred)

        # 6) Bootstrap for 95% CI
        n_boot = args.n_bootstraps
        ci = {}
        ci['acc_ci']       = bootstrap_metric(y_true, y_prob, y_pred, metric_accuracy, n_boot)
        ci['ap_ci']        = bootstrap_metric(y_true, y_prob, y_pred, metric_ap, n_boot)
        ci['auc_ci']       = bootstrap_metric(y_true, y_prob, y_pred, metric_auc, n_boot)
        ci['eer_ci']       = bootstrap_metric(y_true, y_prob, y_pred, metric_eer, n_boot)
        ci['recall_ci']    = bootstrap_metric(y_true, y_prob, y_pred, metric_recall, n_boot)
        ci['precision_ci'] = bootstrap_metric(y_true, y_prob, y_pred, metric_precision, n_boot)
        ci['f1_ci']        = bootstrap_metric(y_true, y_prob, y_pred, metric_f1, n_boot)

        # 7) Write summary for this dataset
        fout.write(f"=== Dataset: {dataset_name} ===\n")
        fout.write(f"Samples: {len(group)}\n")
        fout.write(f"Accuracy: {acc:.4f}   95% CI: ({ci['acc_ci'][0]:.4f}, {ci['acc_ci'][1]:.4f})\n")
        fout.write(f"Average Precision (AP): {ap:.4f}   95% CI: ({ci['ap_ci'][0]:.4f}, {ci['ap_ci'][1]:.4f})\n")
        fout.write(f"AUC: {auc_val:.4f}   95% CI: ({ci['auc_ci'][0]:.4f}, {ci['auc_ci'][1]:.4f})\n")
        fout.write(f"EER: {eer_val:.4f}   95% CI: ({ci['eer_ci'][0]:.4f}, {ci['eer_ci'][1]:.4f})\n")
        fout.write(f"Recall: {recall_:.4f}   95% CI: ({ci['recall_ci'][0]:.4f}, {ci['recall_ci'][1]:.4f})\n")
        fout.write(f"Precision: {precision_:.4f}   95% CI: ({ci['precision_ci'][0]:.4f}, {ci['precision_ci'][1]:.4f})\n")
        fout.write(f"F1 Score: {f1_:.4f}   95% CI: ({ci['f1_ci'][0]:.4f}, {ci['f1_ci'][1]:.4f})\n")
        fout.write(f"Confusion Matrix:\n{cm}\n\n")
        fout.flush()

        # 8) Plot and save confusion matrix
        cm_path = os.path.join(args.output_dir, f"{dataset_name}_confusion_matrix.png")
        plot_confusion_matrix(cm, classes=["real", "fake"], output_path=cm_path)

        # 9) Plot reliability curve
        calib_path = os.path.join(args.output_dir, f"{dataset_name}_reliability.png")
        plot_reliability_curve(y_true, y_prob, output_path=calib_path)

        # 10) Plot ROC curve
        roc_path = os.path.join(args.output_dir, f"{dataset_name}_roc.png")
        plot_roc_curve(y_true, y_prob, output_path=roc_path, dataset_name=dataset_name)

        print(f"Processed dataset: {dataset_name}")

    fout.close()
    print(f"\nAll dataset summaries saved to: {summary_path}")


##############################
# 4.  Command‐Line Interface
##############################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute deepfake‐detection metrics by base‐dataset with 95% bootstrap CIs"
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Path to the input CSV (must contain columns: dataset, label, response)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Directory where metrics text and plot images will be saved."
    )
    parser.add_argument(
        "--n_bootstraps",
        type=int,
        default=1000,
        help="Number of bootstrap samples to compute 95% CIs."
    )
    args = parser.parse_args()
    main(args)
