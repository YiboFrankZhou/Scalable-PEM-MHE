"""Simulation and IAC data, observation windows, and learning features."""
import csv
import pickle
from itertools import product
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from utils import ROOT

OBSERVED_STATE_INDICES = np.array([0, 2])
IAC_SAMPLE_TIME = 0.04
MAX_BRAKE_PRESSURE = 2757.89990234


def load_data(experiment):
    if experiment == 2:
        return {split: build_partial_observation_dataset(
            pickle.loads((ROOT / "datasets" / f"dataset_{split}.pkl").read_bytes()))
            for split in ("train", "val")}
    data, _ = build_iac_datasets(ROOT / "datasets" / "IAC", horizon=40,
                                block_size=400, stride=20, validation_fraction=0.2)
    return data


def build_partial_observation_dataset(dataset):
    return {
        'x': np.asarray(dataset['x'])[..., 3:6].copy(),
        'y': np.asarray(dataset['y'])[..., OBSERVED_STATE_INDICES].copy(),
        'u': np.asarray(dataset['u']).copy()
    }


def _header_indices(header):
    return {name.split('(')[0].lstrip('# '): index for index, name in enumerate(header)}


def load_iac_trajectory(path, minimum_speed=5.0):
    """Load the first >5 m/s segment exactly as the authors' CSV parser does.

    Returns x=[vx, vy, yaw_rate], u=[steering, normalized throttle/brake], time.
    An input u[k] acts on the transition x[k] -> x[k+1].
    """
    path = Path(path)
    states, controls, times = ([], [], [])
    started = False
    with path.open(newline='') as file:
        rows = csv.reader(file)
        indices = _header_indices(next(rows))
        for row in rows:
            vx = float(row[indices['vx']])
            if abs(vx) < minimum_speed:
                if started:
                    break
                continue
            started = True
            brake = float(row[indices['brake_ped_cmd']])
            throttle = -brake / MAX_BRAKE_PRESSURE if brake > 0.0 else float(row[indices['throttle_ped_cmd']]) / 100.0
            states.append((vx, float(row[indices['vy']]), float(row[indices['omega']])))
            controls.append((float(row[indices['delta']]), throttle))
            times.append(float(row[indices['time']]))
    if len(states) < 2:
        raise ValueError(f'no usable IAC high-speed segment in {path}')
    return (np.asarray(states), np.asarray(controls), np.asarray(times))


def _trajectory_blocks(states, controls, block_size, horizon):
    blocks = []
    for first in range(0, len(states) - 1, block_size):
        last = min(first + block_size, len(states) - 1)
        if last - first < horizon:
            continue
        blocks.append((states[first:last + 1], controls[first:last]))
    return blocks


def _block_features(block):
    states, controls = block
    vx, _, yaw_rate = states[:-1].T
    steering, throttle = controls.T
    return np.column_stack((vx, yaw_rate, steering, throttle))


def _split_score(blocks_by_run, phases, period):
    train, val = ([], [])
    for blocks, phase in zip(blocks_by_run, phases):
        for index, block in enumerate(blocks):
            (val if index % period == phase else train).append(_block_features(block))
    train, val = (np.vstack(train), np.vstack(val))
    scale = np.maximum(np.vstack((train, val)).std(axis=0), 1e-08)
    quantiles = (0.05, 0.25, 0.5, 0.75, 0.95)
    score = np.mean(((train.mean(0) - val.mean(0)) / scale) ** 2)
    score += 0.5 * np.mean(((train.std(0) - val.std(0)) / scale) ** 2)
    score += np.mean(((np.quantile(
        train,
        quantiles,
        axis=0,
    ) - np.quantile(val, quantiles, axis=0)) / scale) ** 2)
    return score


def _windows_from_blocks(blocks, horizon, stride):
    result = {'x': [], 'u': [], 'y': []}
    for states, controls in blocks:
        for first in range(0, len(controls) - horizon + 1, stride):
            x = states[first:first + horizon + 1]
            u = controls[first:first + horizon]
            result['x'].append(x)
            result['u'].append(u)
            result['y'].append(x[1:, OBSERVED_STATE_INDICES])
    if not result['u']:
        raise ValueError('no windows were created; reduce horizon or stride')
    return {key: np.stack(value) for key, value in result.items()}


def build_iac_datasets(data_dir, horizon=40, block_size=400, stride=20, validation_fraction=0.2):
    """Build blocked Putnam train/validation and untouched LVMS test windows."""
    data_dir = Path(data_dir)
    period = round(1.0 / validation_fraction)
    if period < 2 or not np.isclose(validation_fraction, 1.0 / period):
        raise ValueError('validation_fraction must be a reciprocal such as 0.2 or 0.25')
    putnam_paths = sorted(data_dir.glob('Putnam*.csv'))
    lvms_paths = sorted(data_dir.glob('LVMS*.csv'))
    if len(putnam_paths) != 3 or len(lvms_paths) != 2:
        raise FileNotFoundError('expected three Putnam and two LVMS IAC CSV files')
    putnam_runs = [load_iac_trajectory(path) for path in putnam_paths]
    blocks_by_run = [_trajectory_blocks(x, u, block_size, horizon) for x, u, _ in putnam_runs]
    phase_candidates = product(range(period), repeat=len(blocks_by_run))
    phases = min(phase_candidates, key=lambda candidate: _split_score(blocks_by_run, candidate, period))
    train_blocks, val_blocks = ([], [])
    for blocks, phase in zip(blocks_by_run, phases):
        for index, block in enumerate(blocks):
            (val_blocks if index % period == phase else train_blocks).append(block)
    test_blocks = []
    lvms_lengths = []
    for path in lvms_paths:
        states, controls, _ = load_iac_trajectory(path)
        lvms_lengths.append(len(states))
        test_blocks.extend(_trajectory_blocks(states, controls, block_size, horizon))
    datasets = {
        'train': _windows_from_blocks(train_blocks, horizon, stride),
        'val': _windows_from_blocks(val_blocks, horizon, stride),
        'test': _windows_from_blocks(test_blocks, horizon, stride),
    }
    metadata = {
        'putnam_files': [path.name for path in putnam_paths],
        'lvms_files': [path.name for path in lvms_paths],
        'putnam_segment_lengths': [len(run[0]) for run in putnam_runs],
        'lvms_segment_lengths': lvms_lengths,
        'validation_phases': phases,
        'block_size': block_size,
        'stride': stride,
        'horizon': horizon,
        'sample_time': IAC_SAMPLE_TIME,
    }
    return (datasets, metadata)


def partial_observation_model_inputs(dataset, l1=1.248, l2=1.7328):
    """Build [vx, vy_proxy, yaw_rate, steering, throttle] without full-state data."""
    ys = np.asarray(dataset['y'], dtype=float)
    us = np.asarray(dataset['u'], dtype=float)
    observed_state = np.concatenate((ys[:, :1], ys[:, :-1]), axis=1)
    vx, yaw_rate = (observed_state[..., 0], observed_state[..., 1])
    steering = us[..., 0]
    vy_proxy = 0.5 * (np.abs(vx) * np.tan(np.clip(steering, -0.8, 0.8)) + (l2 - l1) * yaw_rate)
    return np.stack((vx, vy_proxy, yaw_rate, steering, us[..., 1]), axis=-1)


class DatasetPytorch(Dataset):
    def __init__(self, dataset, split="train"):
        self.u, self.y = dataset[split]["u"], dataset[split]["y"]
        self.xhats = self.whats = self.mus = None

    def __len__(self):
        return len(self.u)

    def __getitem__(self, index):
        sample = (torch.as_tensor(self.u[index], dtype=torch.float32),
                  torch.as_tensor(self.y[index], dtype=torch.float32))
        if self.xhats is None:
            return sample
        return sample + (self.xhats[index], self.whats[index], self.mus[index])


def evaluation_tensors(data, histories, dtype=torch.float64):
    return {
        split: tuple((torch.as_tensor(a, dtype=dtype) for a in (histories[split][:, -1], d['u'][:, -1], d['y'][:, -1]))) for split,
        d in data.items()
    }


def aligned_features(dataset, states):
    controls = dataset['u'][:, :-1]
    previous = np.concatenate((controls[:, :1], controls[:, :-1]), axis=1)
    increments = controls - previous
    network = np.concatenate(
        (states[:, 1:], previous[..., 1:2], previous[..., 0:1], increments[..., 1:2], increments[..., 0:1]),
        axis=2
    )
    physics = network.copy()
    previous, current = (dataset['u'][:, -2], dataset['u'][:, -1])
    physics[:, -1, 3:] = np.column_stack((
        previous[:, 1],
        previous[:, 0],
        current[:, 1] - previous[:, 1],
        current[:, 0] - previous[:, 0]
    ))
    return (network, physics, dataset['y'][:, -1])
