"""Runtime setup, checkpoint/result I/O, and timing."""
import ctypes
import gc
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def configure_runtime():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    if sys.platform == "darwin":
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")
    root = Path(__file__).resolve().parent
    os.environ.setdefault("MPLCONFIGDIR", str(root / "build" / "matplotlib"))
    if sys.platform == "darwin" and "ACADOS_SOURCE_DIR" in os.environ:
        lib_dir = Path(os.environ["ACADOS_SOURCE_DIR"]) / "lib"
        for name in ("libblasfeo.0.dylib", "libhpipm.dylib", "libqpOASES_e.dylib", "libacados.dylib"):
            path = lib_dir / name
            if path.exists():
                ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)


def load_initialization(run_name):
    import numpy as np

    with np.load(ROOT / "runs" / run_name / "initialization.npz") as saved:
        return {key: saved[key].copy() for key in saved.files}


def save_metrics(path, train_pem, val_pem, step):
    result = {"train_pem": float(train_pem), "val_pem": float(val_pem), "selected_step": int(step)}
    (Path(path) / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def show_model(directory, model, parameters=None):
    """Display the loaded model and its parameters."""
    import pandas as pd
    directory = Path(directory)
    print("Checkpoint:", (directory / "model_params_opt.pth").relative_to(ROOT))
    print(model)
    print("Model parameters:", sum(p.numel() for p in model.parameters()))
    if parameters is not None:
        print(pd.Series(parameters, name="Physical parameters").to_string(float_format=lambda v: f"{v:.8g}"))


def show_metrics(train_pem, val_pem, mhe_warmstart=False):
    print(f"Train PEM cost: {train_pem:.6e}")
    print(f"Validation PEM cost: {val_pem:.6e}")
    if mhe_warmstart:
        print("MHE results depend on the solver warm start. EKF/RTS warm starts are used here, "
              "so the costs may differ from those reported in the paper.")


def save_car_model_params(model, path):
    import torch

    torch.save({"state_dict": model.state_dict(), "physics_param_value": model.physics_param_value.detach().cpu(),
                "physics_param_names": model.physics_param_names, "use_nn_flag": model.use_nn_flag}, path)


def load_car_model_params(model, path):
    import torch

    checkpoint = torch.load(path, map_location=next(model.parameters()).device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    return checkpoint


def save_joint_checkpoint(model, states, optimizer, epoch, metrics, path):
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(),
                "physics_param_value": model.physics_param_value.detach().cpu(),
                "physics_param_names": model.physics_param_names,
                "state_estimates": states.detach().cpu(),
                "optimizer": optimizer.state_dict(), "epoch": epoch, "metrics": metrics}, path)


def timed_gradient_call(callable_):
    was_gc_enabled = gc.isenabled()
    gc.disable()
    try:
        t0 = time.perf_counter_ns()
        result = callable_()
        elapsed_ms = (time.perf_counter_ns() - t0) / 1000000.0
    finally:
        if was_gc_enabled:
            gc.enable()
    return (elapsed_ms, result)
