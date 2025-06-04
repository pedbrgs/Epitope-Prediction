import os
import pandas as pd
from evaluation.metrics import compute_stability_jaccard


def save_results(results_df, output_path, baseline_name, dataset_name):
    filename = f"{baseline_name.lower()}_{dataset_name}.parquet"
    full_path = os.path.join(output_path, filename)
    results_df.to_parquet(full_path, index=False)
    print(f"Saved all runs to: {full_path}")


def save_summary(summary_df, output_path, baseline_name, dataset_name):
    filename = f"{baseline_name.lower()}_{dataset_name}_summary.parquet"
    full_path = os.path.join(output_path, filename)
    summary_df.to_parquet(full_path, index=False)
    print(f"Saved summary to: {full_path}")


def save_tuning_logs(logs, output_dir, method_name, dataset_name, total_features, baseline_name):
    logs_df = pd.DataFrame({
        "k": logs["k_values"],
        "cv_score": logs["cv_scores"],
        "cv_std": logs["cv_stds"]
    })

    logs_df["dataset"] = dataset_name
    logs_df["total_features"] = total_features
    logs_df["method"] = baseline_name
    logs_df["model"] = "random_forest"

    # Reorder columns: metadata first, then tuning values
    logs_df = logs_df[
        ["dataset", "total_features", "method", "model", "k", "cv_score", "cv_std"]
    ]

    filename = f"{method_name.lower()}_{dataset_name}_tuning_logs.parquet"
    output_path = os.path.join(output_dir, filename)
    logs_df.to_parquet(output_path, index=False)
    print(f"Tuning logs saved to: {output_path}")


def summarize_results(
        results_df,
        all_selected_features,
        baseline_name,
        dataset_name,
        n_runs,
        total_features
    ):
    numeric_cols = [
        "balanced_accuracy", "roc_auc", "mcc",
        "features_%", "eng_features_%", "deep_features_%"
    ]

    summary_data = {}
    stability = compute_stability_jaccard(all_selected_features)

    summary_data["dataset"] = dataset_name
    summary_data["total_features"] = total_features
    summary_data["method"] = baseline_name
    summary_data["model"] = "random_forest"
    summary_data["n_runs"] = n_runs
    summary_data["stability"] = f"{stability:.4f}"
    # add average runtime per run
    summary_data["avg_runtime_sec"] = (
        f"{results_df['runtime_sec'].mean():.2f} ± "
        f"{results_df['runtime_sec'].std():.2f}"
    )

    for col in numeric_cols:
        mean = results_df[col].mean()
        std = results_df[col].std()
        summary_data[col] = f"{mean:.4f} ± {std:.4f}"

    summary_df = pd.DataFrame([summary_data])
    return summary_df