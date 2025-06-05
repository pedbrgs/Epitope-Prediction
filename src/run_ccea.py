import argparse
import json
import random
import time

import pandas as pd
from pyccea.coevolution import CCPSTFG
from pyccea.utils.datasets import DataLoader
from sklearn.ensemble import RandomForestClassifier

from evaluation.testing import holdout_eval
from utils.artifacts import save_results, save_summary, summarize_results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, help=".parquet data path")
    parser.add_argument("--dataset-name", type=str, help="dataset name")
    parser.add_argument("--ccea-name", type=str, help="ccea name")
    parser.add_argument("--class-col", type=str, default="label", help="class column name")
    parser.add_argument("--subset-col", type=str, default="subset", help="subset column name")
    parser.add_argument("--output-path", type=str, help="output data path")
    parser.add_argument("--n-runs", type=int, default=30, help="number of runs")
    return parser.parse_args()


def get_data_conf(random_state) -> dict:
    return {
        "general": {
            "splitter_type": "k_fold",
            "verbose": True,
            "seed": random_state
        },
        "splitter": {
            "preset": True,
            "kfolds": 3,
            "prefold": True
        },
        "normalization": {
            "normalize": False
        }
    }


def get_ccea_conf(random_state) -> dict:
    return {
        "coevolution": {
            "subpop_sizes": [30],
            "max_gen": 10000,
            "max_gen_without_improvement": 100,
            "optimized_resource_allocation": True,
            "seed" : random_state
        },
        "decomposition": {
            "method": "clustering",
            "drop": True,
            "max_n_clusters": 10,
            "max_n_pls_components": 30,
            "removal_quantile_step_size": 0.05,
            "max_removal_quantile": 0.95,
            "clustering_model_type": "agglomerative_clustering"
        },
        "collaboration": {
            "method": "best"
        },
        "wrapper": {
            "task": "classification",
            # "model_type": "random_forest",
            "model_type": "k_nearest_neighbors"
        },
        "evaluation": {
            "fitness_function": "penalty",
            "eval_function": "balanced_accuracy",
            "eval_mode": "k_fold",
            "weights": [1.0, 0.0]
        },
        "optimizer": {
            "method": "GA",
            "selection_method": "generational",
            "mutation_rate": 0.05,
            "crossover_rate": 1.0,
            "tournament_sample_size": 1,
            "elite_size": 1
        }
    }


def get_dataloader(args, data_conf) -> DataLoader:
    # Set the new dataset
    DataLoader.DATASETS[args.dataset_name] = {
        "file": args.data_path,
        "task": "classification"
    }
    # Return the dataloader object
    dataloader = DataLoader(
        dataset=args.dataset_name,
        conf=data_conf
    )
    return dataloader


def init_ccea(args, dataloader, ccea_conf):
    CCEA_OPTIONS = {
        "ccpstfg": CCPSTFG
    }
    name = args.ccea_name.lower()
    if name in CCEA_OPTIONS.keys():
        ccea = CCEA_OPTIONS[name](
            data=dataloader,
            conf=ccea_conf,
            verbose=True
        )
    else:
        raise ValueError(f"The CCEA {args.ccea_name.upper()} is not implemented.")
    return ccea


def main(args):

    all_results = []
    all_selected_features = []

    print("Running multiple runs...")
    for run in range(args.n_runs):
        print(f"Run {run + 1}/{args.n_runs}...")

        random_state = random.randint(0, 10_000)

        # Build dataloader
        data_conf = get_data_conf(random_state)
        dataloader = get_dataloader(args, data_conf)
        dataloader.get_ready()

        # Run CCEA
        print("Selecting features...")
        ccea_conf = get_ccea_conf(random_state)
        ccea = init_ccea(args, dataloader, ccea_conf)
        start_time = time.time()
        ccea.optimize()
        run_time = time.time() - start_time

        # Select the name of the best features
        feature_cols = [col for col in dataloader.data.columns if col.startswith("feat_")]
        if args.ccea_name.upper() in ["CCPSTFG"]:
            kept_feature_indices = set(range(len(feature_cols))).difference(ccea.removed_features)
            kept_feature_names = dataloader.data.columns[list(kept_feature_indices)]
            selected_features = kept_feature_names[ccea.best_context_vector.astype(bool)].tolist()
            X_train_selected = dataloader.data.query("subset == 'train'")[selected_features].copy()
        else:
            pass

        estimator = RandomForestClassifier(random_state=random_state, n_jobs=-1)
        best_estimator = estimator.fit(X_train_selected, dataloader.y_train)

        print("Evaluating model...")
        test_data = dataloader.data.loc[dataloader.data[args.subset_col] == "test"].copy()
        result = holdout_eval(
            model=best_estimator,
            test_data=test_data,
            selected_features=selected_features,
            total_features=dataloader.n_features,
            method=args.ccea_name,
            dataset_name=args.dataset_name,
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
    save_results(results_df, args.output_path, args.ccea_name, args.dataset_name)

    # Summarize and save
    summary_df = summarize_results(
        results_df=results_df,
        all_selected_features=all_selected_features,
        baseline_name=args.ccea_name,
        dataset_name=args.dataset_name,
        n_runs=args.n_runs,
        total_features=dataloader.n_features
    )
    save_summary(summary_df, args.output_path, args.ccea_name, args.dataset_name)

    print("Done.")


if __name__ == "__main__":
    main(parse_args())