"""Prediction-error objectives and the PEM-MHE, Joint-MHE, EKF, and SUBNET methods."""
import copy
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from data import DatasetPytorch, OBSERVED_STATE_INDICES, aligned_features
from estimation import (
    ObserverMHEAcados, calculate_MHE_constraint_batch, calculate_MHE_loss_batch,
    tensor_parameters,
)
from gradients import (
    adjoint_decomposition_batch, build_KKT_blocks_batch, solve_block_tridiagonal_batch,
)
from models import car_physics_raw_params, load_authors_model
from utils import ROOT, save_car_model_params


def calculate_PEM_loss(y_batch, xest_batch, C):
    C = torch.as_tensor(C, dtype=xest_batch.dtype, device=xest_batch.device)
    residual = xest_batch[:, -1]@C.T-y_batch[:, -1]
    return .5*residual.square().sum(dim=1).mean()


def evaluate_PEM_loss(loader, C):
    dataset = loader.dataset
    ys = torch.as_tensor(dataset.y, dtype=dataset.xhats.dtype)
    return calculate_PEM_loss(ys, dataset.xhats, C).item()


class PEMMHE:
    def __init__(self, model, config, data):
        self.model, self.config = model, config
        self.params = tensor_parameters(config)
        self.observer = ObserverMHEAcados(config)
        self.loaders = {split: DataLoader(DatasetPytorch(data, split),
                        batch_size=config.batch_size, shuffle=False) for split in data}

    def estimate(self, split, warmstart=None):
        self.observer.update_params(self.model.sbar.detach().numpy(), self.model.Sigmainv.detach().numpy())
        dataset = self.loaders[split].dataset
        if warmstart is None and dataset.xhats is not None:
            warmstart = dataset.xhats.numpy()
        results = []
        batch_size = self.config.batch_size
        for first in range(0, len(dataset), batch_size):
            end = first + batch_size
            seed = None if warmstart is None else warmstart[first:end]
            results.append(self.observer.estimate_batch(dataset.u[first:end], dataset.y[first:end],
                                                        train_flag=True, warmstart_xs=seed))
        dataset.xhats, dataset.whats, dataset.mus = (
            torch.as_tensor(np.concatenate(values), dtype=torch.float32) for values in zip(*results))
        return evaluate_PEM_loss(self.loaders[split], self.config.C)

    def backward(self):
        """Accumulate -d(F^T lambda)/d alpha with primal, dual, and adjoint fixed."""
        loader = self.loaders["train"]
        for us, ys, xs, ws, mus in loader:
            with torch.no_grad():
                blocks = build_KKT_blocks_batch(self.params, self.model, xs, ws, us, ys)
                adjoint = solve_block_tridiagonal_batch(*blocks)
            lambda_x, lambda_w, lambda_H = adjoint_decomposition_batch(self.params, adjoint)
            xs = xs.detach().requires_grad_(True)
            ws = ws.detach().requires_grad_(True)
            M = calculate_MHE_loss_batch(self.params, self.model, xs, ws, us, ys)
            H = calculate_MHE_constraint_batch(self.params, self.model, xs, ws, us, ys)
            H = H.reshape(-1, self.config.horizon, self.config.nx)
            lagrangian = M.sum() + torch.sum(mus * H)
            grad_x, grad_w = torch.autograd.grad(lagrangian, (xs, ws), create_graph=True)
            surrogate = -(torch.sum(grad_x*lambda_x) + torch.sum(grad_w*lambda_w)
                          + torch.sum(H*lambda_H)) / len(loader.dataset)
            surrogate.backward()


def joint_mhe_components(model, states, us, ys, sbar, Sigmainv, Qinv, Rinv, C, sample_time=0.02):
    """Reduced MHE objective after eliminating w and eta for E=I and F=I."""
    batch_size, horizon, nx = us.shape[0], us.shape[1], states.shape[2]
    predicted = model.discrete(states[:, :-1].reshape(batch_size * horizon, nx),
                               us.reshape(batch_size * horizon, -1)).reshape(batch_size, horizon, nx)
    process_residual = (states[:, 1:] - predicted) / sample_time
    measurement_residual = states[:, 1:-1] @ C.T - ys[:, :-1]
    arrival_residual = states[:, 0] - sbar
    arrival = torch.sum(Sigmainv * arrival_residual.square(), dim=1)
    process = torch.einsum("bki,ij,bkj->b", process_residual, Qinv, process_residual)
    measurement = torch.einsum("bki,ij,bkj->b", measurement_residual, Rinv, measurement_residual)
    return {"total": torch.mean(arrival + process + measurement), "arrival": torch.mean(arrival),
            "process": torch.mean(process), "measurement": torch.mean(measurement)}


def one_step_pem(model, states, us, ys, observed_indices=OBSERVED_STATE_INDICES):
    """Predict y[N] from xhat[N-1] and u[N-1], without using y[N] in estimation."""
    prediction = model.discrete(states[:, -2], us[:, -1])
    residual = prediction[:, observed_indices] - ys[:, -1]
    return 0.5 * torch.mean(torch.sum(residual.square(), dim=1)), prediction


def ekf_pem_loss(model, tensors, C):
    x, u, y = tensors
    residual = model.discrete(x, u) @ C.T - y
    return 0.5 * residual.square().sum(dim=1).mean()


def train_ekf_pem(model, tensors, C, output_dir, writer, iterations=100, patience=15):
    model.double().set_train_flag('physics')
    optimizer = torch.optim.LBFGS(
        car_physics_raw_params(model),
        lr=0.5,
        max_iter=20,
        max_eval=25,
        history_size=50,
        tolerance_grad=1e-10,
        tolerance_change=1e-12,
        line_search_fn='strong_wolfe'
    )
    best, stale = (float('inf'), 0)
    history = []
    for step in range(iterations + 1):
        if step:

            def closure():
                optimizer.zero_grad(set_to_none=True)
                loss = ekf_pem_loss(model, tensors['train'], C)
                loss.backward()
                return loss
            optimizer.step(closure)
        with torch.no_grad():
            train_pem = float(ekf_pem_loss(model, tensors['train'], C))
            val_pem = float(ekf_pem_loss(model, tensors['val'], C))
        history.append((step, train_pem, val_pem))
        if val_pem < best - 1e-12:
            best, stale = (val_pem, 0)
            best_state = copy.deepcopy(model.state_dict())
            metrics = {'train_pem': train_pem, 'val_pem': val_pem, 'selected_step': step}
            save_car_model_params(model, output_dir / 'model_params_opt.pth')
        else:
            stale += int(step > 0)
        writer.add_scalar('PEM/train', train_pem, step)
        writer.add_scalar('PEM/val', val_pem, step)
        if stale >= patience:
            break
    model.load_state_dict(best_state)
    return (metrics, np.asarray(history))


class SUBNET:

    def __init__(self, data, nx=3, width=128, depth=2, batch_size=256, weight_decay=3e-05, seed=0):
        sys.path.insert(0, str(ROOT / 'repos' / 'deepSI'))
        import deepSI

        self.batch_size, self.weight_decay, self.seed = (batch_size, weight_decay, seed)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        train = data['train']
        self.nx, self.width, self.depth = (nx, width, depth)
        self.horizon, self.nu, self.ny = (train['u'].shape[1], train['u'].shape[2], train['y'].shape[2])
        self.normalization = {}
        for name in ('u', 'y'):
            self.normalization[name + '_mean'] = train[name].mean(axis=(0, 1))
            self.normalization[name + '_std'] = train[name].std(axis=(0, 1))
        for name in ('u_std', 'y_std'):
            self.normalization[name] = np.maximum(self.normalization[name], 1e-08)
        self.data = data
        self.prepare_tensors()
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.model = deepSI.fit_systems.SS_encoder(
            nx=nx,
            na=self.horizon - 1,
            nb=self.horizon - 1,
            feedthrough=False
        )
        for prefix in ('e', 'f', 'h'):
            setattr(self.model, prefix + '_n_nodes_per_layer', width)
            setattr(self.model, prefix + '_n_hidden_layers', depth)
        self.model.init_model(
            nu=self.nu,
            ny=self.ny,
            device=str(self.device),
            auto_fit_norm=False,
            optimizer_kwargs={'lr': 0.0003, 'weight_decay': weight_decay}
        )
        self.parameters = [p for group in self.model.parameters for p in group['params']]

    def prepare_tensors(self):
        norm = self.normalization
        self.datasets = {}
        for split, values in self.data.items():
            u = (values['u'] - norm['u_mean']) / norm['u_std']
            y = (values['y'] - norm['y_mean']) / norm['y_std']
            arrays = (u[:, :-1], y[:, :-1], u[:, -1], y[:, -1], values['y'][:, -1])
            self.datasets[split] = TensorDataset(*(torch.as_tensor(a, dtype=torch.float32) for a in arrays))
        self.y_mean = torch.as_tensor(norm['y_mean'], dtype=torch.float32, device=self.device)
        self.y_std = torch.as_tensor(norm['y_std'], dtype=torch.float32, device=self.device)

    def predict(self, u_history, y_history, u_terminal):
        history = torch.cat((u_history.flatten(1), y_history.flatten(1)), dim=1)
        z = self.model.encoder(history)
        z = self.model.fn(torch.cat((z, u_terminal), dim=1))
        return self.model.hn(z)

    def evaluate(self, split):
        self.model.eval()
        raw_error, normalized_error, count = (0.0, 0.0, 0)
        with torch.no_grad():
            for batch in DataLoader(self.datasets[split], batch_size=self.batch_size):
                u, y, terminal_u, target_norm, target = (v.to(self.device) for v in batch)
                prediction_norm = self.predict(u, y, terminal_u)
                raw_error += ((prediction_norm * self.y_std + self.y_mean - target) ** 2).sum().item()
                normalized_error += ((prediction_norm - target_norm) ** 2).sum().item()
                count += len(target)
        return (raw_error / (2 * count), normalized_error / (self.ny * count))

    def state(self):
        return {key: copy.deepcopy(getattr(self.model, key).state_dict()) for key in ('encoder', 'fn', 'hn')}

    def load(self, checkpoint):
        for name in ('encoder', 'fn', 'hn'):
            getattr(self.model, name).load_state_dict(checkpoint[name])
        for name in self.normalization:
            self.normalization[name] = np.asarray(checkpoint[name])
        self.prepare_tensors()

    def train_phase(
        self,
        phase,
        lr,
        epochs,
        writer,
        output_dir,
        offset=0,
        patience=200,
        cooldown=20,
        stop_patience=1000,
        min_lr=1e-06
    ):
        optimizer = torch.optim.Adam(self.parameters, lr=lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=0.3,
            patience=patience,
            threshold=0.0001,
            cooldown=cooldown,
            min_lr=min_lr
        )
        loader = DataLoader(
            self.datasets['train'],
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.seed)
        )
        best, best_epoch, history = (float('inf'), 0, [])
        for epoch in range(epochs + 1):
            if epoch:
                self.model.train()
                for batch in loader:
                    u, y, terminal_u, target_norm, target = (v.to(self.device) for v in batch)
                    optimizer.zero_grad(set_to_none=True)
                    prediction = self.predict(u, y, terminal_u)
                    loss = ((prediction - target_norm) ** 2).mean() if phase == 'normalized' else 0.5 * ((prediction * self.y_std + self.y_mean - target) ** 2).sum(dim=1).mean()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters, 10.0)
                    optimizer.step()
            train_pem, train_norm = self.evaluate('train')
            val_pem, val_norm = self.evaluate('val')
            monitor = val_norm if phase == 'normalized' else val_pem
            if epoch:
                scheduler.step(monitor)
            if monitor < best:
                best, best_epoch, best_state = (monitor, epoch, self.state())
                torch.save(
                    {**best_state, **{k: torch.as_tensor(v) for k, v in self.normalization.items()}, 'phase': phase, 'epoch': epoch, 'train_pem': train_pem, 'val_pem': val_pem, 'nx': self.nx, 'network_width': self.width, 'network_depth': self.depth, 'window': self.horizon, 'history': self.horizon - 1},
                    output_dir / 'model_params_opt.pth'
                )
            writer.add_scalar('PEM/train', train_pem, offset + epoch)
            writer.add_scalar('PEM/val', val_pem, offset + epoch)
            history.append((epoch, train_pem, val_pem, train_norm, val_norm))
            if epoch % 100 == 0:
                print(f'{phase} {epoch}: {train_pem:.6e} / {val_pem:.6e}')
            if optimizer.param_groups[0]['lr'] <= min_lr * 1.01 and epoch - best_epoch >= stop_patience:
                break
        for name, values in best_state.items():
            getattr(self.model, name).load_state_dict(values)
        return np.asarray(history)


class AdaptivePEM:

    def __init__(self, data, histories, config, C, iac=False, batch_size=32, seed=0, scaler=None):
        from sklearn.preprocessing import StandardScaler
        self.config, self.batch_size, self.seed = (config, batch_size, seed)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.C = torch.as_tensor(C, dtype=torch.float32, device=self.device)
        self.model = load_authors_model(iac)(config).to(self.device)
        features = {s: aligned_features(d, histories[s]) for s, d in data.items()}
        self.scaler = scaler or StandardScaler().fit(features['train'][0].reshape(-1, 7))
        self.datasets = {}
        for split, (network, physics, target) in features.items():
            normalized = self.scaler.transform(network.reshape(-1, 7)).reshape(network.shape)
            self.datasets[split] = TensorDataset(*(torch.as_tensor(v, dtype=torch.float32) for v in (
                physics,
                normalized,
                target
            )))

    def predict(self, physics, history):
        hidden = self.model.init_hidden(len(physics)) if self.model.is_rnn else None
        prediction, _, parameters = self.model(physics, history, hidden)
        return (prediction, parameters)

    def evaluate(self, split):
        self.model.eval()
        squared_error = torch.zeros(len(self.C), device=self.device)
        parameter_sum = torch.zeros(len(self.model.sys_params), device=self.device)
        count = 0
        with torch.no_grad():
            for physics, history, target in DataLoader(self.datasets[split], batch_size=self.batch_size):
                prediction, parameters = self.predict(physics.to(self.device), history.to(self.device))
                squared_error += ((prediction @ self.C.T - target.to(self.device)) ** 2).sum(dim=0)
                parameter_sum += parameters.sum(dim=0)
                count += len(target)
        return {
            'pem': 0.5 * (squared_error / count).sum().item(),
            'parameter_mean': (parameter_sum / count).cpu().numpy()
        }

    def train_epoch(self):
        self.model.train()
        loader = DataLoader(
            self.datasets['train'],
            batch_size=self.batch_size,
            shuffle=True,
            generator=self.generator
        )
        for physics, history, target in loader:
            self.model.optimizer.zero_grad(set_to_none=True)
            prediction, _ = self.predict(physics.to(self.device), history.to(self.device))
            loss = 0.5 * ((prediction @ self.C.T - target.to(self.device)) ** 2).sum(dim=1).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
            self.model.optimizer.step()

    def save(self, path, step, metrics, ekf_params):
        torch.save(
            {'state_dict': self.model.state_dict(), 'config': self.config, 'epoch': step, 'train_metrics': metrics['train'], 'val_metrics': metrics['val'], 'scaler_mean': self.scaler.mean_, 'scaler_scale': self.scaler.scale_, 'ekf_physics_params': ekf_params},
            path
        )
