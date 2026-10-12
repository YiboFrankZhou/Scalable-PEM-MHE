#!/usr/bin/env python3

import argparse
import pickle
from itertools import product
from pathlib import Path

import numpy as np


FEATURE_NAMES = ("vx", "vy", "yaw_rate", "steering", "throttle", "alpha_f", "alpha_r")


def load_deep_dynamics_trajectory(path):
    data = np.load(path)
    xs = data["dstates"][:6, :-1].T.copy()
    us = np.column_stack((data["inputs"][1, :-1], data["inputs"][0, :-1]))
    ys = data["dstates"][3:6, 1:-1].T.copy()
    return xs, us, ys


def build_windows(x_trajectories, u_trajectories, y_trajectories, window):
    windows = {"x": [], "y": [], "u": []}
    for xs, us, ys in zip(x_trajectories, u_trajectories, y_trajectories):
        n_windows = len(us) - window + 1
        windows["x"].append(np.stack([xs[index:index+window+1] for index in range(n_windows)]))
        windows["y"].append(np.stack([ys[index:index+window] for index in range(n_windows)]))
        windows["u"].append(np.stack([us[index:index+window] for index in range(n_windows)]))
    return {key: np.vstack(value) for key, value in windows.items()}


def split_trajectory_into_blocks(path, block_size, window):
    xs, us, ys = load_deep_dynamics_trajectory(path)
    blocks = []
    for first in range(0, len(us), block_size):
        last = min(first + block_size, len(us))
        if last - first < window: continue
        blocks.append((xs[first:last+1], us[first:last], ys[first:last]))
    return blocks


def identification_features(block):
    xs, us, _ = block
    vx, vy, yaw_rate = xs[:-1, 3], xs[:-1, 4], xs[:-1, 5]
    alpha_f = us[:, 0] - np.arctan2(yaw_rate * 0.029 + vy, np.abs(vx))
    alpha_r = np.arctan2(yaw_rate * 0.033 - vy, np.abs(vx))
    return np.column_stack((vx, vy, yaw_rate, us, alpha_f, alpha_r))


def distribution_score(blocks_by_trajectory, phases, period):
    train_features, val_features = [], []
    for blocks, phase in zip(blocks_by_trajectory, phases):
        for index, block in enumerate(blocks):
            target = val_features if index % period == phase else train_features
            target.append(identification_features(block))
    train_features, val_features = np.vstack(train_features), np.vstack(val_features)
    all_features = np.vstack((train_features, val_features))
    scale = np.maximum(all_features.std(axis=0), 1e-8)
    quantiles = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
    moment_error = np.mean(((train_features.mean(axis=0) - val_features.mean(axis=0)) / scale) ** 2)
    moment_error += 0.5 * np.mean(((train_features.std(axis=0) - val_features.std(axis=0)) / scale) ** 2)
    quantile_error = np.mean(((np.quantile(train_features, quantiles, axis=0) -
                               np.quantile(val_features, quantiles, axis=0)) / scale) ** 2)
    outside = np.any((val_features < train_features.min(axis=0)) |
                     (val_features > train_features.max(axis=0)), axis=1)
    return moment_error + quantile_error + 5.0 * outside.mean(), train_features, val_features


def select_validation_phases(blocks_by_trajectory, period):
    candidates = []
    for phases in product(range(period), repeat=len(blocks_by_trajectory)):
        score, train_features, val_features = distribution_score(blocks_by_trajectory, phases, period)
        candidates.append((score, phases, train_features, val_features))
    return min(candidates, key=lambda candidate: candidate[0])


def build_split(blocks_by_trajectory, phases, period, window, validation):
    x_trajectories, u_trajectories, y_trajectories = [], [], []
    for blocks, phase in zip(blocks_by_trajectory, phases):
        for index, (xs, us, ys) in enumerate(blocks):
            if (index % period == phase) != validation: continue
            x_trajectories.append(xs)
            u_trajectories.append(us)
            y_trajectories.append(ys)
    return build_windows(x_trajectories, u_trajectories, y_trajectories, window=window)


def print_distribution(train_features, val_features):
    print("Feature distribution: mean +- std [min, max]")
    for index, name in enumerate(FEATURE_NAMES):
        train = train_features[:, index]
        val = val_features[:, index]
        print(f"  {name:>9}: train {train.mean(): .4f} +- {train.std():.4f} [{train.min(): .4f}, {train.max(): .4f}] | "
              f"val {val.mean(): .4f} +- {val.std():.4f} [{val.min(): .4f}, {val.max(): .4f}]")
    outside = np.any((val_features < train_features.min(axis=0)) |
                     (val_features > train_features.max(axis=0)), axis=1)
    print(f"Validation transitions outside the complete train feature range: {outside.mean():.2%}")


def prepare_dataset(dataset_dir, window=40, block_size=100, val_fraction=0.2):
    period = round(1.0 / val_fraction)
    if period < 2 or not np.isclose(val_fraction, 1.0 / period):
        raise ValueError("val_fraction must be the reciprocal of an integer, for example 0.2 or 0.25.")
    raw_dir = dataset_dir
    paths = [raw_dir / "DYN-PP-ETHZ.npz", raw_dir / "DYN-PP-ETHZMobil.npz"]
    blocks_by_trajectory = [split_trajectory_into_blocks(path, block_size, window) for path in paths]
    score, phases, train_features, val_features = select_validation_phases(blocks_by_trajectory, period)
    train = build_split(blocks_by_trajectory, phases, period, window, validation=False)
    val = build_split(blocks_by_trajectory, phases, period, window, validation=True)

    print(f"Selected validation block phases: ETHZ={phases[0]}, ETHZMobil={phases[1]}, period={period}, score={score:.6f}")
    print(f"Dataset windows: train={len(train['u'])}, validation={len(val['u'])}, horizon={window}")
    print_distribution(train_features, val_features)
    return train, val


def main():
    parser = argparse.ArgumentParser(description="Build blocked, distribution-balanced Deep Dynamics datasets for tyre system identification.")
    parser.add_argument("--dataset-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--window", type=int, default=40)
    parser.add_argument("--block-size", type=int, default=100)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    args = parser.parse_args()

    train, val = prepare_dataset(args.dataset_dir, args.window, args.block_size, args.val_fraction)
    with open(args.dataset_dir / "dataset_train.pkl", "wb") as file:
        pickle.dump(train, file)
    with open(args.dataset_dir / "dataset_val.pkl", "wb") as file:
        pickle.dump(val, file)
    print(f"Saved {args.dataset_dir / 'dataset_train.pkl'}")
    print(f"Saved {args.dataset_dir / 'dataset_val.pkl'}")


if __name__ == "__main__":
    main()
