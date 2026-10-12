"""Check the archived checkpoints and TensorBoard records against the paper tables."""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import ROOT, configure_runtime
configure_runtime()

import numpy as np
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from models import TRUE_PHYSICS_PARAMS, TYRE_PARAM_NAMES



def scalar_at(directory, tag, step):
    events = EventAccumulator(str(directory), size_guidance={"scalars": 0})
    events.Reload()
    return next(event.value for event in events.Scalars(tag) if event.step == step)


def checkpoint_metrics(name, directory, checkpoint):
    if "SUBNET" in name:
        return checkpoint["train_pem"], checkpoint["val_pem"], checkpoint["epoch"]
    if "adaptive" in name:
        return checkpoint["train_metrics"]["pem"], checkpoint["val_metrics"]["pem"], checkpoint["epoch"]
    if "Joint" in name:
        return checkpoint["metrics"]["train_pem"], checkpoint["metrics"]["val_pem"], checkpoint["epoch"]
    if name == "Ex2_EKF-PEM":
        with np.load(directory / "candidate_selection.npz") as selection:
            index = int(np.argmin(selection["val_pem"]))
            return (selection["train_pem"][index], selection["val_pem"][index],
                    int(selection["best_iteration"][index]))
    events = EventAccumulator(str(directory / "PEM_loss_val_current"), size_guidance={"scalars": 0})
    events.Reload()
    selected = min(events.Scalars("PEM_loss_val"), key=lambda event: event.value)
    train_directory = directory / "PEM_loss_train_current" if name.startswith("Ex2") else directory
    train = scalar_at(train_directory, "PEM_loss_train", selected.step)
    # The physical parameters logged at this step must belong to the saved model.
    for index, parameter in enumerate(TYRE_PARAM_NAMES):
        logged = scalar_at(directory / ("physics_params_" + parameter), "physics_params", selected.step)
        np.testing.assert_allclose(
            float(checkpoint["physics_param_value"][index]),
            logged,
            rtol=2e-6,
            atol=1e-9,
        )
    return train, selected.value, selected.step


def verify():
    manifest = json.loads((ROOT / "runs" / "manifest.json").read_text())
    print(f'{"Experiment":28s} {"Train PEM":>12s} {"Val PEM":>12s} {"Step":>6s}')
    for entry in manifest:
        name = entry["experiment"]
        directory = ROOT / "runs" / name
        for relative, expected in entry["sha256"].items():
            actual = hashlib.sha256((directory / relative).read_bytes()).hexdigest()
            assert actual == expected, f"File changed: {name}/{relative}"
        checkpoint = torch.load(directory / "model_params_opt.pth", map_location="cpu", weights_only=False)
        train, val, step = checkpoint_metrics(name, directory, checkpoint)
        recorded = json.loads((directory / "metrics.json").read_text())
        assert step == recorded["selected_step"]
        for key, value in (("train_pem", train), ("val_pem", val)):
            assert float(f"{value:.2e}") == entry["paper_values"][key], (name, key, value)
            np.testing.assert_allclose(value, recorded[key], rtol=1e-7)
        expected_error = entry["paper_values"]["parameter_error_percent"]
        if expected_error is not None:
            if "adaptive" in name:
                names = [next(iter(parameter)) for parameter in checkpoint["config"]["PARAMETERS"]]
                values = dict(zip(names, checkpoint["val_metrics"]["parameter_mean"]))
                theta = np.array([values[key] for key in TYRE_PARAM_NAMES])
            else:
                theta = checkpoint["physics_param_value"].numpy()
            error = 100*np.mean(np.abs(theta[:16]-TRUE_PHYSICS_PARAMS[:16])/np.abs(TRUE_PHYSICS_PARAMS[:16]))
            assert round(float(error), 2) == expected_error, (name, error)
        print(f'{name:28s} {train:12.6e} {val:12.6e} {step:6d}')
    print("All 11 archived results match the paper's displayed values.")
    print("This checks recorded results and file integrity; it does not rerun training or nonlinear MHE.")


if __name__ == "__main__":
    verify()
