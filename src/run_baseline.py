import argparse
import json
import os
import random
import time

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score

from baseline import (
    MutualInformationFeatureSelection,
    MinimumRedundancyMaximumRelevance,
    RecursiveFeatureElimination,
    SequentialFeatureSelection
)
from data.splitting import input_output_split, train_test_split
from data.utils import read_parquet_data
from evaluation.metrics import compute_stability_jaccard
from evaluation.testing import holdout_eval


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, help=".parquet data path")
    parser.add_argument("--baseline-name", type=str, help="baseline algorithm name")
    parser.add_argument("--step-size", type=float, default=0.05, help="parameter step size")
    parser.add_argument("--class-col", type=str, default="label", help="class column name")
    parser.add_argument("--subset-col", type=str, default="subset", help="subset column name")
    parser.add_argument("--split-col", type=str, default="fold", help="fold column name")
    parser.add_argument("--random-state", type=int, default=1234, help="random seed")
    parser.add_argument("--output-path", type=str, help="output data path")
    parser.add_argument("--n-runs", type=int, default=30, help="number of runs")
    return parser.parse_args()


def init_baseline(args):
    BASELINE_OPTIONS = {
        "mi": MutualInformationFeatureSelection,
        "mrmr": MinimumRedundancyMaximumRelevance,
        "rfe": RecursiveFeatureElimination,
        "sfs": SequentialFeatureSelection,
    }
    name = args.baseline_name.lower()
    if name in BASELINE_OPTIONS.keys():
        baseline = BASELINE_OPTIONS[name](
            estimator=RandomForestClassifier(random_state=args.random_state, n_jobs=-1),
            eval_function=balanced_accuracy_score,
            random_state=args.random_state
        )
    else:
        raise ValueError(f"The baseline {args.baseline_name.upper()} is not implemented.")
    return baseline


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


def main(args):
    print("Loading data...")
    data = read_parquet_data(args.data_path)
    train_data, test_data = train_test_split(data=data, subset_col=args.subset_col)
    dataset_name = args.data_path.split("/")[-1].split(".")[0]

    print("Processing data...")
    feature_cols = [col for col in data.columns if col.startswith("feat")]

    X_train, y_train = input_output_split(
        data=train_data,
        feature_cols=feature_cols,
        class_col=args.class_col
    )

    print("Tuning baseline algorithm...")
    baseline = init_baseline(args)

    logs = baseline.tune(
        X_train=X_train,
        y_train=y_train,
        folds=train_data[args.split_col],
        feature_cols=feature_cols,
        step_size=args.step_size
    )

    save_tuning_logs(
        logs=logs,
        output_dir=args.output_path,
        method_name=baseline.__name__,
        dataset_name=dataset_name,
        total_features=len(feature_cols),
        baseline_name=args.baseline_name
    )

    all_results = []
    all_selected_features = []

    print("Running multiple runs...")
    for run in range(args.n_runs):
        print(f"Run {run + 1}/{args.n_runs}...")

        random_state = random.randint(0, 10_000)

        estimator = RandomForestClassifier(random_state=random_state, n_jobs=-1)

        start_time = time.time()
        selected_features = baseline.select(
            X_train=X_train,
            y_train=y_train,
            n_features=logs["best_k"],
            estimator=estimator
        )
        run_time = time.time() - start_time

        best_estimator = baseline.fit(
            X_train=X_train,
            y_train=y_train,
            estimator=estimator,
            selected_features=selected_features
        )

        print("Evaluating model...")
        result = holdout_eval(
            model=best_estimator,
            test_data=test_data,
            selected_features=selected_features,
            total_features=len(feature_cols),
            method=baseline.__name__,
            dataset_name=dataset_name,
            model_name="random_forest",
            label_col=args.class_col
        )

        # Add run-specific info and runtime
        result["run"] = run
        result["random_state"] = random_state
        result["selected_features"] = json.dumps(selected_features)
        result["runtime_sec"] = run_time

        all_results.append(result)
        all_selected_features.append(set(selected_features))

    results_df = pd.concat(all_results, ignore_index=True)

    # Save detailed results
    save_results(results_df, args.output_path, baseline.__name__, dataset_name)

    # Summarize and save
    summary_df = summarize_results(
        results_df=results_df,
        all_selected_features=all_selected_features,
        baseline_name=baseline.__name__,
        dataset_name=dataset_name,
        n_runs=args.n_runs,
        total_features=len(feature_cols)
    )
    save_summary(summary_df, args.output_path, baseline.__name__, dataset_name)

    print("Done.")


if __name__ == "__main__":
    main(parse_args())
