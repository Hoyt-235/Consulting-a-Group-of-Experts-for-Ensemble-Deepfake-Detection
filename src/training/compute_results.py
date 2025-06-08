import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    average_precision_score,
    accuracy_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    brier_score_loss,
    balanced_accuracy_score,
)
import argparse

def compute_eer(fpr, tpr):
    """
    Compute the Equal Error Rate (EER) given false positive rates (fpr)
    and true positive rates (tpr) from an ROC curve.
    EER is the point where FPR ≈ 1 - TPR.
    """
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    return eer

def main(csv_paths):
    # csv_paths: list of file paths. Each path is …/<model_name>/<dataset_name>.csv
    # We will extract:
    #   model_name = basename of parent directory
    #   dataset_name = filename without extension
    #
    # We will compute metrics per (model, dataset), collect into a DataFrame,
    # then save:
    #   - metrics.json (pretty-printed JSON)
    #   - metrics.tex (LaTeX table)
    #   - ROC curves: for each dataset, overlay ROC curves of all models on that dataset

    # 1) Validate input CSVs
    for p in csv_paths:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"File not found: {p}")

    # 2) Create a single "metrics" directory in current working directory
    base_metrics_dir = os.path.join(os.getcwd(), "metrics")
    os.makedirs(base_metrics_dir, exist_ok=True)

    # 3) Prepare a list to collect metric rows
    rows = []
    # Also store ROC data to plot later, keyed by dataset
    roc_data = {}  # { dataset_name: [ (model_name, fpr, tpr, auc, eer), ... ] }

    # 4) Loop over each CSV path
    for csv_path in csv_paths:
        # Extract model_name and dataset_name
        model_name = os.path.basename(os.path.dirname(csv_path))
        dataset_name = os.path.splitext(os.path.basename(csv_path))[0]

        # Load and parse
        df = pd.read_csv(csv_path, sep=",")
        if "pred_prob" not in df.columns or "true_label" not in df.columns:
            raise ValueError(
                f"CSV '{csv_path}' must contain 'prediction' and 'label' columns separated by semicolons."
            )

        y_score = df["pred_prob"].astype(float).values
        y_true = df["true_label"].astype(int).values
        y_pred = (y_score >= 0.5).astype(int)

        # Compute metrics
        ap = average_precision_score(y_true, y_score)
        acc = accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_score)
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        eer = compute_eer(fpr, tpr)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        brier = brier_score_loss(y_true, y_score)
        bal_acc = balanced_accuracy_score(y_true, y_pred)

        # Append to rows
        rows.append({
            "model": model_name,
            "dataset": dataset_name,
            "AP": float(f"{ap:.6f}"),
            "ACC": float(f"{acc:.6f}"),
            "AUC": float(f"{auc:.6f}"),
            "EER": float(f"{eer:.6f}"),
            "Brier": float(f"{brier:.6f}"),
            "Bal_ACC": float(f"{bal_acc:.6f}"),
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp)
        })

        # Store ROC for plotting
        roc_data.setdefault(dataset_name, []).append((model_name, fpr, tpr, auc, eer))

    # 5) Build a DataFrame of all metrics
    metrics_df = pd.DataFrame(rows)

    # 6) Save metrics to pretty‐printed JSON
    json_path = os.path.join(base_metrics_dir, "metrics.json")
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=4)

    # 7) Save LaTeX table
    tex_path = os.path.join(base_metrics_dir, "metrics.tex")
    with open(tex_path, "w") as f:
        f.write("\\begin{table}[ht]\n")
        f.write("  \\centering\n")
        f.write("  \\begin{tabular}{l l r r r r r r r r r r}\n")
        f.write("    \\hline\n")
        f.write(
            "    Model & Dataset & AP & ACC & AUC & EER & Brier & BalACC & TN & FP & FN & TP \\\\\n"
        )
        f.write("    \\hline\n")
        for _, row in metrics_df.iterrows():
            f.write(
                f"    {row['model']} & {row['dataset']} & "
                f"{row['AP']:.4f} & {row['ACC']:.4f} & {row['AUC']:.4f} & {row['EER']:.4f} & "
                f"{row['Brier']:.4f} & {row['Bal_ACC']:.4f} & "
                f"{int(row['TN'])} & {int(row['FP'])} & {int(row['FN'])} & {int(row['TP'])} \\\\\n"
            )
        f.write("    \\hline\n")
        f.write("  \\end{tabular}\n")
        f.write("  \\caption{Performance metrics for each model on each dataset.}\n")
        f.write("  \\label{tab:all_metrics}\n")
        f.write("\\end{table}\n")

    # 8) Plot ROC curves per dataset
    for dataset_name, entries in roc_data.items():
        plt.figure(figsize=(6, 6))
        for (model_name, fpr, tpr, auc, eer) in entries:
            plt.plot(fpr, tpr,
                     label=f"{model_name} (AUC={auc:.3f}, EER={eer:.3f})",
                     linewidth=2)
        plt.plot([0, 1], [0, 1], "k--", label="Random Chance")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve on {dataset_name}")
        plt.legend(loc="lower right", fontsize="small")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        roc_out = os.path.join(base_metrics_dir, f"roc_{dataset_name}.png")
        plt.savefig(roc_out)
        plt.close()

    print(f"All metrics and plots saved under '{base_metrics_dir}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute and aggregate metrics across multiple CSVs. "
                    "Each CSV path should be '<...>/<model_name>/<dataset_name>.csv' "
                    "with columns 'prediction;label'."
    )
    parser.add_argument(
        "csv_paths",
        nargs="+",
        help="List of CSV files to process. Each should be in a folder named after the model, "
             "with the filename as the dataset name."
    )
    args = parser.parse_args()
    main(args.csv_paths)

    print("Usage: python compute_results.py <csv_path1> <csv_path2> ...")

    """
    Copy and paste this command on the terminal to compute results for all models and datasets:
    python compute_results.py \
        ./predictions/spsl/Celeb-DF-v1.csv \
        ./predictions/spsl/Celeb-DF-v2.csv \
        ./predictions/spsl/DeepFakeDetection.csv \
        ./predictions/ucf/Celeb-DF-v1.csv \
        ./predictions/ucf/Celeb-DF-v2.csv \
        ./predictions/ucf/DeepFakeDetection.csv \
        ./predictions/stil/Celeb-DF-v1.csv \
        ./predictions/stil/Celeb-DF-v2.csv \
        ./predictions/stil/DeepFakeDetection.csv \
        ./predictions/uia_vit/Celeb-DF-v1.csv \
        ./predictions/uia_vit/Celeb-DF-v2.csv \
        ./predictions/uia_vit/DeepFakeDetection.csv 

           python compute_results.py \
        ./predictions/decision_fusion/Celeb-DF-v1.csv \
        ./predictions/decision_fusion/Celeb-DF-v2.csv \
        ./predictions/decision_fusion/deepFakeDetection.csv \
        ./predictions/feature_fusion/Celeb-DF-v1.csv \
        ./predictions/feature_fusion/Celeb-DF-v2.csv \
        ./predictions/feature_fusion/deepFakeDetection.csv 

    """