import argparse
import os

import pandas as pd

import processing


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, help=".csv data path")
    parser.add_argument("--n-splits", type=int, help="number of folds + 1 for testing")
    parser.add_argument("--group-col", type=str, default="Info_group", help="group column name")
    parser.add_argument("--class-col", type=str, default="Class", help="class column name")
    parser.add_argument("--split-col", type=str, default="Info_split", help="split column name")
    parser.add_argument("--pos-class-id", type=int, default=1, help="positive class value")
    parser.add_argument("--neg-class-id", type=int, default=-1, help="negative class value")
    parser.add_argument("--random-state", type=int, default=1234, help="random seed")
    parser.add_argument("--output-path", type=str, help="output data path")
    parser.add_argument("--subsampling", action="store_true", help="enable subsampling")
    return parser.parse_args()


def main(args):

    print("Loading data...")
    data = processing._read_csv_data(args.data_path)
    feature_names = processing._get_feature_names(data)

    # Some data statistics
    n_features = len(feature_names)
    print(f"Number of features: {n_features}")
    class_counts = data[args.class_col].value_counts(normalize=False).to_dict()
    majority_class_count = max(class_counts[args.pos_class_id], class_counts[args.neg_class_id])
    minority_class_count = min(class_counts[args.pos_class_id], class_counts[args.neg_class_id])
    imbalance_ratio = round(majority_class_count/minority_class_count, 2)
    print(f"Class counts: {class_counts}")
    print(f"Imbalance ratio: {imbalance_ratio}")

    data = processing._split_group_kfolds(
        data=data,
        n_splits=args.n_splits,
        group_col=args.group_col,
        split_col=args.split_col,
        random_state=args.random_state
    )
    processing._check_group_leakage(data=data, group_col=args.group_col, split_col=args.split_col)

    train_data, test_data = processing._train_test_split(
        data=data,
        split_col=args.split_col,
        n_splits=args.n_splits
    )

    # Some subset statistics
    n_training_samples = train_data.shape[0]
    n_test_samples = test_data.shape[0]
    print(f"Training data size: {n_training_samples}")
    print(f"Test data size: {n_test_samples}")

    # Subsampling data
    if args.subsampling:

        fold_rows = processing._count_class_rows_by_fold(
            data=train_data,
            class_col=args.class_col,
            split_col=args.split_col,
            class_id=args.pos_class_id
        )
        print(f"Fold rows: {fold_rows}")
        target_fold_rows = processing._get_min_rows_across_folds(fold_rows)
        print(f"Min target value: {target_fold_rows}")
        fold_rows = processing._assign_rows_per_fold(
            fold_counts=fold_rows, n_rows_per_fold=target_fold_rows)
        print(f"Fold rows: {fold_rows}")

        print("Subsampling data...")
        pos_train_sample = processing._stratified_sample_with_balanced_groups(
            data=train_data,
            split_col=f"new_{args.split_col}",
            group_col=args.group_col,
            target_sizes=fold_rows,
            class_col=args.class_col,
            class_id=args.pos_class_id,
            random_state=args.random_state
        )

        neg_train_sample = processing._stratified_sample_with_balanced_groups(
            data=train_data,
            split_col=f"new_{args.split_col}",
            group_col=args.group_col,
            target_sizes=fold_rows,
            class_col=args.class_col,
            class_id=args.neg_class_id,
            random_state=args.random_state
        )
        train_data = pd.concat([pos_train_sample, neg_train_sample]).reset_index(drop=True)
        class_counts = train_data[args.class_col].value_counts(normalize=False).to_dict()
        majority_class_count = max(class_counts[args.pos_class_id], class_counts[args.neg_class_id])
        minority_class_count = min(class_counts[args.pos_class_id], class_counts[args.neg_class_id])
        imbalance_ratio = round(majority_class_count/minority_class_count, 2)
        print(f"Class counts (training data): {class_counts}")
        print(f"Imbalance ratio (training data): {imbalance_ratio}")

    train_data = processing._rename_data_columns(
        data=train_data, split_col=args.split_col, class_col=args.class_col)
    train_data["subset"] = "train"
    test_data = processing._rename_data_columns(
        data=test_data, split_col=args.split_col, class_col=args.class_col)
    test_data["subset"] = "test"

    sample = (
        pd.concat([train_data, test_data])
        .reset_index(drop=True)
        .loc[:, feature_names + ["fold", "label", "subset"]]
        .drop_duplicates()
    )

    n_samples = sample.shape[0]
    n_training_samples = sample.query("subset == 'train'").shape[0]
    n_test_samples = sample.query("subset == 'test'").shape[0]
    print(f"Number of samples: {n_samples}")
    print(f"Number of training samples: {n_training_samples}")
    print(f"Number of test samples: {n_test_samples}")

    filename = args.data_path.split("/")[-1].split(".")[0]
    filename = f"{filename}_subsampled" if args.subsampling else filename
    output_path = os.path.join(args.output_path, f"{filename}.parquet")
    sample.to_parquet(output_path, index=False)


if __name__ == "__main__":

    main(parse_args())
