"""Inner MHE configuration, objective, solver, and EKF/RTS initial states."""
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import casadi as ca
import numpy as np
import torch
from scipy.linalg import block_diag

from data import OBSERVED_STATE_INDICES
from models import CarModelCasadi, integrate
from utils import ROOT


@dataclass
class MHEConfig:
    nx: int
    nu: int
    nw: int
    horizon: int
    sample_time: float
    C: np.ndarray
    E: np.ndarray
    Qinv: np.ndarray
    Rinv: np.ndarray
    sbar: np.ndarray
    Sigmainv: np.ndarray
    f: object
    theta: object
    theta_values: object
    integration: str = "euler"
    rk4_steps: int = 1
    batch_size: int = 32
    num_threads: int = 1
    build_dir: Path = Path("build")
    warmstart: object = None


class ObserverMHEAcados:
    def __init__(self, config):
        from acados_template import AcadosOcpBatchSolver

        self.config = config
        self.nx, self.nu, self.nw = config.nx, config.nu, config.nw
        self.ny, self.N = config.C.shape[0], config.horizon
        self.T, self.mhe_batch_size = config.sample_time, config.batch_size
        self.C, self.Qinv, self.Rinv = config.C, config.Qinv, config.Rinv
        self.sbar, self.Sigmainv = config.sbar, config.Sigmainv
        self.accepted_nonzero_statuses = {2: 0, 3: 0}
        self.name = "mhe_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        config.build_dir.mkdir(parents=True, exist_ok=True)
        ocp = self.ocp()
        self.batch_solver = AcadosOcpBatchSolver(
            ocp, config.batch_size, num_threads_in_batch_solve=config.num_threads,
            json_file=str(config.build_dir / (self.name + ".json")), verbose=False)
        self.solver = self.batch_solver.ocp_solvers[0]

    def ocp(self):
        from acados_template import AcadosModel, AcadosOcp

        cfg = self.config
        nx, nu, nw, ny = self.nx, self.nu, self.nw, self.ny
        x, u, w = ca.MX.sym("x", nx), ca.MX.sym("u", nu), ca.MX.sym("w", nw)
        arrival = ca.MX.sym("arrival", 2*nx)
        model = AcadosModel()
        model.name, model.x, model.u = self.name, x, ca.vertcat(u, w)
        model.p_global = ca.vertcat(arrival, cfg.theta)
        model.disc_dyn_expr = integrate(cfg.f, x, u, w, self.T, cfg.integration, cfg.rk4_steps)
        model.p = ca.MX.sym("p", 0)
        self.rhs = ca.Function(self.name + "_rhs", [x, u, cfg.theta], [cfg.f(x, u, ca.MX.zeros(nw))])

        ocp = AcadosOcp()
        ocp.model = model
        ocp.code_export_directory = str(cfg.build_dir / self.name)
        ocp.solver_options.N_horizon = self.N
        ocp.solver_options.tf = self.N * self.T
        ocp.cost.cost_type_0 = "NONLINEAR_LS"
        ocp.model.cost_y_expr_0 = ca.vertcat(
            ca.diag(ca.sqrt(arrival[nx:])) @ (x - arrival[:nx]),
            np.linalg.cholesky(cfg.Qinv).T @ w)
        ocp.cost.W_0, ocp.cost.yref_0 = np.eye(nx+nw), np.zeros(nx+nw)
        ocp.cost.cost_type = "LINEAR_LS"
        ocp.cost.W = block_diag(cfg.Rinv, cfg.Qinv)
        ocp.cost.Vx = np.vstack((cfg.C, np.zeros((nw, nx))))
        ocp.cost.Vu = np.zeros((ny+nw, nu+nw))
        ocp.cost.Vu[ny:, nu:] = np.eye(nw)
        ocp.cost.Vz = np.array([[]])
        ocp.cost.yref = np.zeros(ny+nw)
        ocp.cost.W_e, ocp.cost.Vx_e = np.zeros((nx, nx)), np.eye(nx)
        ocp.cost.yref_e = np.zeros(nx)
        ocp.constraints.lbu = ocp.constraints.ubu = np.zeros(nu)
        ocp.constraints.idxbu = np.arange(nu)
        ocp.parameter_values = np.array([])
        ocp.p_global_values = np.concatenate((self.sbar, self.Sigmainv, cfg.theta_values()))
        options = ocp.solver_options
        options.qp_solver = "FULL_CONDENSING_HPIPM"
        options.hessian_approx = "GAUSS_NEWTON"
        options.integrator_type = "DISCRETE"
        options.nlp_solver_type = "SQP"
        options.nlp_solver_max_iter = 300
        options.nlp_solver_tol_stat = 2e-5
        options.nlp_solver_tol_eq = options.nlp_solver_tol_ineq = options.nlp_solver_tol_comp = 1e-5
        options.globalization = "FUNNEL_L1PEN_LINESEARCH"
        options.globalization_line_search_use_sufficient_descent = 1
        options.globalization_use_SOC = 1
        options.levenberg_marquardt = 1e-2 if self.N >= 10 else 1e-3
        options.with_batch_functionality = True
        return ocp

    def update_params(self, sbar, Sigmainv):
        self.sbar, self.Sigmainv = np.asarray(sbar), np.asarray(Sigmainv)
        values = np.concatenate((self.sbar, self.Sigmainv, self.config.theta_values()))
        self.batch_solver.set_p_global_and_precompute_dependencies(
            np.broadcast_to(values, (self.mhe_batch_size, values.size)).copy())

    def estimate_batch_once(self, us, ys, warmstart_xs, reset=True):
        us, ys = np.asarray(us), np.asarray(ys)
        count = len(us)
        theta = self.config.theta_values()
        if reset:
            self.batch_solver.reset(count)
        for k in range(self.N):
            self.batch_solver.constraints_set(k, "lbu", us[:, k])
            self.batch_solver.constraints_set(k, "ubu", us[:, k])
        for i, solver in enumerate(self.batch_solver.ocp_solvers[:count]):
            for k in range(1, self.N):
                solver.set(k, "yref", np.r_[ys[i, k-1], np.zeros(self.nw)])
            for k in range(self.N+1):
                solver.set(k, "x", warmstart_xs[i, k])
            for k in range(self.N):
                residual = (warmstart_xs[i, k+1] - warmstart_xs[i, k]) / self.T
                residual -= np.asarray(self.rhs(warmstart_xs[i, k], us[i, k], theta)).ravel()
                w = np.linalg.pinv(self.config.E) @ residual
                solver.set(k, "u", np.r_[us[i, k], w])
        self.batch_solver.solve(count)
        costs = np.array([s.get_cost() for s in self.batch_solver.ocp_solvers[:count]])
        return self._result(count), costs, self.batch_solver.status[:count].copy()

    def _result(self, count):
        xs = self.batch_solver.get_flat("x", count).reshape(count, self.N+1, self.nx)
        inputs = self.batch_solver.get_flat("u", count).reshape(count, self.N, self.nu+self.nw)
        pis = self.batch_solver.get_flat("pi", count).reshape(count, self.N, self.nx)
        return xs, inputs[..., self.nu:], -(2.0/self.T)*pis

    def estimate_batch(self, us, ys, train_flag=False, warmstart_xs=None):
        us, ys = np.asarray(us), np.asarray(ys)
        supplied_warmstart = warmstart_xs is not None
        if warmstart_xs is None:
            if self.config.warmstart is not None:
                warmstart_xs = np.stack([self.config.warmstart(u, y) for u, y in zip(us, ys)])
            else:
                warmstart_xs = np.broadcast_to(self.sbar, (len(us), self.N+1, self.nx)).copy()
                warmstart_xs[:, 1:self.N] += (ys[:, :-1] - self.sbar @ self.C.T) @ np.linalg.pinv(self.C).T
        result, _, status = self.estimate_batch_once(us, ys, warmstart_xs, reset=False)
        for _ in range(2):
            if np.all(status == 0):
                break
            self.batch_solver.solve(len(us))
            result, status = self._result(len(us)), self.batch_solver.status[:len(us)]
        if np.any(status) and supplied_warmstart:
            if len(us) == 1:
                return self.estimate_batch(us, ys, train_flag)
            results = [self.estimate_batch(us[i:i+1], ys[i:i+1], True, warmstart_xs[i:i+1])
                       for i in range(len(us))]
            result = tuple(np.concatenate(values) for values in zip(*results))
            return result if train_flag else result[0]
        # Accept finite SQP iterates with status 2 or 3 after retries.
        if not np.isin(status, [0, 2, 3]).all() or not all(np.isfinite(v).all() for v in result):
            raise RuntimeError(f"MHE failed: {status}")
        for value in (2, 3):
            self.accepted_nonzero_statuses[value] += int(np.sum(status == value))
        return result if train_flag else result[0]

    def estimate(self, us, ys, train_flag=False):
        result = self.estimate_batch(us[None], ys[None], train_flag)
        return tuple(v[0] for v in result) if train_flag else result[0]


def calculate_MHE_loss_batch(params, model, xs, ws, us, ys):
    C, Qinv, Rinv = params["C_torch"], params["mhe_Qinv_torch"], params["mhe_Rinv_torch"]
    arrival_residual = xs[:, 0]-model.sbar
    measurement_residual = xs[:, 1:-1]@C.T-ys[:, :-1]
    arrival = (model.Sigmainv*arrival_residual.square()).sum(dim=-1)
    measurement = torch.einsum("bki,ij,bkj->b", measurement_residual, Rinv, measurement_residual)
    process = torch.einsum("bki,ij,bkj->b", ws, Qinv, ws)
    return arrival + measurement + process


def calculate_MHE_constraint_batch(params, model, xs, ws, us, ys):
    B, N, nx = xs.shape[0], params["mhe_window"], params["nx"]
    predicted = model.discrete(xs[:, :N].reshape(B*N, nx), us.reshape(B*N, -1),
        ws.reshape(B*N, -1), params["E_torch"], T=params["T"],
        rk4_steps=params.get("rk4_steps", model.rk4_steps)).reshape(B, N, nx)
    return (xs[:, 1:]-predicted).reshape(B, N*nx, 1)


def calculate_MHE_loss(params, model, xs, ws, us, ys):
    return calculate_MHE_loss_batch(params, model, xs[None], ws[None], us[None], ys[None])[0]


def calculate_MHE_constraint(params, model, xs, ws, us, ys):
    return calculate_MHE_constraint_batch(params, model, xs[None], ws[None], us[None], ys[None])[0]


def tensor_parameters(config):
    params = {"nx": config.nx, "nu": config.nu, "nw": config.nw,
              "ny": config.C.shape[0], "mhe_window": config.horizon,
              "T": config.sample_time, "rk4_steps": config.rk4_steps}
    for name, value in (("E", config.E), ("C", config.C),
                        ("mhe_Qinv", config.Qinv), ("mhe_Rinv", config.Rinv)):
        params[name+"_torch"] = torch.as_tensor(value, dtype=torch.float32)
    return params


def _partial_state_scales(dataset, l1=0.029, l2=0.033):
    ys, us = (np.asarray(dataset['y'], dtype=float), np.asarray(dataset['u'], dtype=float))
    vx, yaw_rate, steering = (ys[..., 0], ys[..., 1], us[..., 0])
    vy_proxy = 0.5 * (np.abs(vx) * np.tan(np.clip(steering, -0.8, 0.8)) + (l2 - l1) * yaw_rate)
    observed_scale = np.maximum(ys.reshape(-1, 2).std(axis=0), np.array([0.05, 0.1]))
    hidden_scale = max(vy_proxy.std(), 0.1 * observed_scale[0], 0.1)
    return (np.array([observed_scale[0], hidden_scale, observed_scale[1]]), vy_proxy)


def ekf_warmstart_windows(model, dataset, initial_hidden_state=None):
    """Run an independent EKF/RTS smoother over each window using y=[vx, yaw_rate] and u only."""
    ys, us = (np.asarray(dataset['y'], dtype=float), np.asarray(dataset['u'], dtype=float))
    n_batch, horizon, _ = ys.shape
    state_scale, vy_proxy = _partial_state_scales({'y': ys, 'u': us}, model.l1, model.l2)
    measurement_scale = state_scale[OBSERVED_STATE_INDICES]
    Q = torch.diag(torch.tensor((0.05 * state_scale) ** 2, dtype=torch.float32)).expand(n_batch, -1, -1).clone()
    R = torch.diag(torch.tensor(
        np.maximum(0.01 * measurement_scale, [0.001, 0.002]) ** 2,
        dtype=torch.float32
    )).expand(n_batch, -1, -1).clone()
    C = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32)
    eye = torch.eye(3, dtype=torch.float32).expand(n_batch, -1, -1)
    x = torch.zeros((n_batch, 3), dtype=torch.float32)
    x[:, OBSERVED_STATE_INDICES] = torch.as_tensor(ys[:, 0], dtype=torch.float32)
    if initial_hidden_state is None:
        x[:, 1] = torch.as_tensor(vy_proxy[:, 0], dtype=torch.float32)
    else:
        hidden_state = np.broadcast_to(np.asarray(initial_hidden_state, dtype=float), (n_batch,))
        x[:, 1] = torch.as_tensor(hidden_state.copy(), dtype=torch.float32)
    P = torch.diag(torch.tensor(state_scale ** 2, dtype=torch.float32)).expand(n_batch, -1, -1).clone()
    filtered_states = torch.empty((n_batch, horizon + 1, 3), dtype=torch.float32)
    filtered_covariances = torch.empty((n_batch, horizon + 1, 3, 3), dtype=torch.float32)
    predicted_states = torch.empty_like(filtered_states)
    predicted_covariances = torch.empty_like(filtered_covariances)
    transition_jacobians = torch.empty((n_batch, horizon, 3, 3), dtype=torch.float32)
    filtered_states[:, 0], filtered_covariances[:, 0] = (x, P)
    state_limit = torch.tensor(
        [2.0 * np.max(np.abs(ys[..., 0])) + 1.0, max(4.0 * state_scale[1], 2.5), 2.0 * np.max(np.abs(ys[..., 1])) + 1.0],
        dtype=torch.float32
    )

    def transition_jacobian(current_x, current_u):
        columns = []
        for state_index in range(3):
            epsilon = max(1e-05, 0.0001 * state_scale[state_index])
            offset = torch.zeros_like(current_x)
            offset[:, state_index] = epsilon
            with torch.no_grad():
                plus = model.discrete(current_x + offset, current_u)
                minus = model.discrete(current_x - offset, current_u)
            columns.append((plus - minus) / (2.0 * epsilon))
        return torch.stack(columns, dim=2)
    for k in range(horizon):
        current_u = torch.as_tensor(us[:, k], dtype=torch.float32)
        A = transition_jacobian(x, current_u)
        with torch.no_grad():
            x_pred = model.discrete(x, current_u)
        x_pred = torch.nan_to_num(x_pred, nan=0.0, posinf=0.0, neginf=0.0).clamp(-state_limit, state_limit)
        P_pred = A @ P @ A.transpose(1, 2) + Q + 1e-09 * eye
        innovation = torch.as_tensor(ys[:, k], dtype=torch.float32) - x_pred @ C.T
        S = C @ P_pred @ C.T + R
        K = torch.linalg.solve(S, (P_pred @ C.T).transpose(1, 2)).transpose(1, 2)
        x = x_pred + (K @ innovation.unsqueeze(-1)).squeeze(-1)
        IKC = eye - K @ C
        P = IKC @ P_pred @ IKC.transpose(1, 2) + K @ R @ K.transpose(1, 2)
        P = 0.5 * (P + P.transpose(1, 2)) + 1e-09 * eye
        predicted_states[:, k + 1], predicted_covariances[:, k + 1] = (x_pred, P_pred)
        filtered_states[:, k + 1], filtered_covariances[:, k + 1] = (x, P)
        transition_jacobians[:, k] = A
    states = filtered_states.clone()
    for k in reversed(range(horizon)):
        PA = filtered_covariances[:, k] @ transition_jacobians[:, k].transpose(1, 2)
        smoother_gain = torch.linalg.solve(predicted_covariances[:, k + 1], PA.transpose(1, 2)).transpose(1, 2)
        correction = (smoother_gain @ (states[:, k + 1] - predicted_states[:, k + 1]).unsqueeze(-1)).squeeze(-1)
        states[:, k] = torch.nan_to_num(
            filtered_states[:, k] + correction,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        ).clamp(-state_limit, state_limit)
    states[:, 0] = filtered_states[:, 0]
    return states.detach().cpu().numpy()


def terminal_excluded_ekf_warmstart(model, dataset, initial_hidden_state=None):
    """Build x[0:N] from y[1:N-1], then predict x[N] without using y[N]."""
    history = {"y": np.asarray(dataset["y"])[:, :-1], "u": np.asarray(dataset["u"])[:, :-1]}
    states = ekf_warmstart_windows(model, history,
                                      initial_hidden_state=initial_hidden_state)
    dtype = next(model.parameters()).dtype
    with torch.no_grad():
        terminal = model.discrete(torch.as_tensor(states[:, -1], dtype=dtype),
                                  torch.as_tensor(dataset["u"][:, -1], dtype=dtype)).cpu().numpy()
    return np.concatenate((states, terminal[:, None]), axis=1)


def fixed_histories(model, data):
    return {
        split: ekf_warmstart_windows(model, {'u': d['u'][:, :-1], 'y': d['y'][:, :-1]}) for split,
        d in data.items()
    }


def initial_ekf_multistart_warmstarts(model, dataset, state_min, state_max, baseline=None):
    """Build the minimal symmetric three-start set for the single hidden state."""
    if baseline is None:
        baseline = ekf_warmstart_windows(model, dataset)
    lower = ekf_warmstart_windows(model, dataset, initial_hidden_state=np.asarray(state_min)[1])
    upper = ekf_warmstart_windows(model, dataset, initial_hidden_state=np.asarray(state_max)[1])
    return np.stack((np.asarray(baseline), lower, upper))


def compute_mhe_estimations_multistart(observer, loader, warmstart_candidates):
    """Use three independent starts once; prefer converged solutions, then the lowest MHE cost."""
    dataset = loader.dataset
    us, ys = (np.asarray(dataset.u), np.asarray(dataset.y))
    warmstart_candidates = np.asarray(warmstart_candidates, dtype=float)
    results, selection_counts, fallback_count = ([], np.zeros(3, dtype=int), 0)
    for first in range(0, len(us), observer.mhe_batch_size):
        last = min(first + observer.mhe_batch_size, len(us))
        candidate_results, candidate_costs, candidate_statuses = ([], [], [])
        for candidate in warmstart_candidates:
            result, costs, statuses = observer.estimate_batch_once(
                us[first:last],
                ys[first:last],
                candidate[first:last]
            )
            candidate_results.append(result)
            candidate_costs.append(costs)
            candidate_statuses.append(statuses)
        candidate_costs, candidate_statuses = (np.asarray(candidate_costs), np.asarray(candidate_statuses))
        finite = np.isfinite(candidate_costs)
        converged = finite & (candidate_statuses == 0)
        acceptable = finite & np.isin(candidate_statuses, (0, 2, 3))
        selected = np.empty(last - first, dtype=int)
        for batch_index in range(last - first):
            valid = converged[:, batch_index]
            if not np.any(valid):
                valid = acceptable[:, batch_index]
                fallback_count += 1
            if not np.any(valid):
                raise RuntimeError(f'all initial MHE candidates failed for dataset window {first + batch_index}; statuses={candidate_statuses[:, batch_index]}')
            scores = np.where(valid, candidate_costs[:, batch_index], np.inf)
            selected[batch_index] = int(np.argmin(scores))
        selection_counts += np.bincount(selected, minlength=3)
        results.append(tuple((np.stack([candidate_results[selected[k]][value_index][k] for k in range(last - first)]) for value_index in range(3))))
    dataset.xhats, dataset.whats, dataset.mus = (torch.tensor(
        np.concatenate(values),
        dtype=torch.float32
    ) for values in zip(*results))
    return {'selection_counts': selection_counts, 'nonconverged_fallbacks': fallback_count}


def grid_warmstart(config, hidden_limit=2.5):
    """Initialize the unmeasured lateral velocity on a grid."""
    x, u, w = ca.MX.sym("x", 3), ca.MX.sym("u", 2), ca.MX.zeros(3)
    rhs = ca.Function("seed_rhs", [x, u, config.theta], [config.f(x, u, w)])
    step = ca.Function("seed_step", [x, u, config.theta], [
        integrate(config.f, x, u, w, config.sample_time, config.integration, config.rk4_steps)])
    grid = np.linspace(-hidden_limit, hidden_limit, 81)
    transition = step.map(len(grid))

    def seed(us, ys):
        N, T = config.horizon, config.sample_time
        theta = config.theta_values()
        states = np.zeros((N+1, 3))
        for k in range(1, N):
            candidates = np.column_stack((np.full(len(grid), ys[k-1, 0]), grid,
                                          np.full(len(grid), ys[k-1, 1])))
            predicted = np.asarray(transition(candidates.T, np.tile(us[k, :, None], len(grid)),
                                              np.tile(theta[:, None], len(grid)))).T
            residual = predicted @ config.C.T - ys[k]
            score = np.einsum("bi,ij,bj->b", residual, config.Rinv, residual)
            states[k] = candidates[np.argmin(score)]
        states[0] = states[1] - T*np.asarray(rhs(states[1], us[0], theta)).ravel()
        states[N] = np.asarray(step(states[N-1], us[N-1], theta)).ravel()
        if config.integration == "euler":
            states[N, OBSERVED_STATE_INDICES] = ys[N-1]
        return states
    return seed


def make_mhe_config(system, model, initialization, batch_size=32):
    """Bind the vehicle equations; the MHE solver itself has no vehicle assumptions."""
    symbolic = system.get("casadi_model", CarModelCasadi)(
        E=system["E"], T=system["T"], m=system["m"], l1=system["l1"], l2=system["l2"],
        integration_method=system["integration"], rk4_steps=system["rk4_steps"],
        physics_param_value=model.physics_param_value.detach().numpy(),
        use_nn_flag=model.use_nn_flag, NN_residual=model.nn_residual,
        mhe_nn_backend="full_nn_casadi")
    physics, _, neural, _ = symbolic.get_acados_p()

    def theta_values():
        physical = model.physics_param_value.detach().cpu().numpy()
        neural = symbolic.NN_casadi.get_params() if model.use_nn_flag else np.array([])
        return np.r_[physical, neural]

    config = MHEConfig(
        nx=system["nx"], nu=system["nu"], nw=system["nw"], horizon=system["N"],
        sample_time=system["T"], C=system["C"], E=system["E"],
        Qinv=initialization["mhe_Qinv"], Rinv=initialization["mhe_Rinv"],
        sbar=model.sbar.detach().numpy(), Sigmainv=model.Sigmainv.detach().numpy(),
        f=symbolic.f_continuous, theta=ca.vertcat(physics, neural), theta_values=theta_values,
        integration=system["integration"], rk4_steps=system["rk4_steps"],
        batch_size=batch_size, build_dir=ROOT / "build")
    config.warmstart = grid_warmstart(
        config,
        system["hidden_limit"]
    )
    return config
