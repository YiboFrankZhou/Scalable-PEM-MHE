"""Continuous dynamics, discretization, and physical/neural model definitions."""
import importlib.util
import sys
import types
from datetime import datetime

import casadi as ca
import numpy as np
import torch
from torch import nn

from utils import ROOT


def integrate(f, x, u, w, sample_time, method="euler", steps=1):
    if method == "euler":
        return x + sample_time*f(x, u, w)
    dt = sample_time/steps
    for _ in range(steps):
        k1 = f(x, u, w)
        k2 = f(x+dt*k1/2, u, w)
        k3 = f(x+dt*k2/2, u, w)
        k4 = f(x+dt*k3, u, w)
        x = x+dt/6*(k1+2*k2+2*k3+k4)
    return x


class Dynamics(nn.Module):
    def __init__(self, f, theta, sbar, Sigmainv, sample_time,
                 integration_method="euler", rk4_steps=1):
        super().__init__()
        self.f = f
        self.theta = nn.Parameter(torch.as_tensor(theta).clone())
        self.sbar = nn.Parameter(torch.as_tensor(sbar).clone())
        precision = torch.as_tensor(Sigmainv)
        self.Sigmainv_raw = nn.Parameter(precision + torch.log(-torch.expm1(-precision)))
        self.sample_time = sample_time
        self.integration_method, self.rk4_steps = integration_method, rk4_steps

    @property
    def Sigmainv(self):
        return nn.functional.softplus(self.Sigmainv_raw)

    def f_continuous(self, x, u):
        return self.f(x, u, self.theta)

    def discrete(self, x, u, wk=None, E=None, T=None, rk4_steps=None):
        noise = 0.0 if wk is None else wk @ E.T
        f = lambda state, control, _: self.f_continuous(state, control) + noise
        return integrate(f, x, u, None, self.sample_time if T is None else T,
                         self.integration_method, self.rk4_steps if rk4_steps is None else rk4_steps)


class MLP_Residual(nn.Module):

    def __init__(self, in_dim, out_dim, input_center=None, input_scale=None, hidden_dim=16):
        super().__init__()
        hidden_dim = int(hidden_dim)
        if input_center is None:
            input_center = torch.zeros(in_dim)
        if input_scale is None:
            input_scale = torch.ones(in_dim)
        self.register_buffer('input_center', torch.as_tensor(input_center, dtype=torch.float32))
        self.register_buffer('input_scale', torch.as_tensor(input_scale, dtype=torch.float32))
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, out_dim)
        )
        with torch.no_grad():
            self.net[-1].weight.zero_()
            self.net[-1].bias.zero_()

    def forward(self, x):
        z = (x - self.input_center) / self.input_scale
        return self.net(z)


class MLP_Residual_casadi:
    """Exact native-CasADi view of an existing PyTorch ``MLP_Residual``."""

    def __init__(self, pytorch_model):
        self.pytorch_model = pytorch_model
        self._layers = []
        self._parameter_symbols = []
        for index, layer in enumerate(pytorch_model.net):
            if isinstance(layer, nn.Linear):
                weight = ca.MX.sym(f'nn_weight_{index}', layer.out_features, layer.in_features)
                bias = ca.MX.sym(f'nn_bias_{index}', layer.out_features)
                self._layers.append(('linear', weight, bias))
                self._parameter_symbols.extend((ca.vec(weight), bias))
            elif isinstance(layer, nn.Tanh):
                self._layers.append(('tanh',))
            elif isinstance(layer, nn.ReLU):
                self._layers.append(('relu',))
            elif isinstance(layer, nn.Sigmoid):
                self._layers.append(('sigmoid',))
            elif isinstance(layer, nn.Identity):
                self._layers.append(('identity',))
            else:
                raise NotImplementedError(f'Unsupported CasADi layer: {type(layer).__name__}')

    def __call__(self, x):
        center = self.pytorch_model.input_center.detach().cpu().numpy().reshape(-1, 1)
        scale = self.pytorch_model.input_scale.detach().cpu().numpy().reshape(-1)
        value = ca.diag(ca.DM(1.0 / scale)) @ (x - ca.DM(center))
        for layer in self._layers:
            if layer[0] == 'linear':
                value = layer[1] @ value + layer[2]
            elif layer[0] == 'tanh':
                value = ca.tanh(value)
            elif layer[0] == 'relu':
                value = ca.fmax(value, 0)
            elif layer[0] == 'sigmoid':
                value = 1 / (1 + ca.exp(-value))
        return value

    def get_sym_params(self):
        return ca.vertcat(*self._parameter_symbols)

    def get_params(self, _=None):
        values = []
        for layer in self.pytorch_model.net:
            if isinstance(layer, nn.Linear):
                weight = layer.weight.detach().cpu().numpy().astype(np.float64)
                bias = layer.bias.detach().cpu().numpy().astype(np.float64).reshape(-1)
                values.extend((weight.ravel(order='F'), bias))
        return np.ascontiguousarray(np.concatenate(values), dtype=np.float64)


class CarDynamicsPytorch(Dynamics):
    physics_param_names = (
        'Bf',
        'Br',
        'Cf',
        'Cr',
        'Df',
        'Dr',
        'Ef',
        'Er',
        'Shf',
        'Shr',
        'Svf',
        'Svr',
        'Cm1',
        'Cm2',
        'Cr0',
        'Cr2',
        'Iz'
    )

    def __init__(
        self,
        physics_parameter_box,
        physics_param_init=None,
        nn_residual=None,
        m=0.041,
        l1=0.029,
        l2=0.033,
        use_nn_flag=False,
        sample_time=0.02,
        integration_method='euler',
        rk4_steps=1
    ):
        nn.Module.__init__(self)
        self.m, self.l1, self.l2 = (m, l1, l2)
        self.sample_time, self.integration_method, self.rk4_steps = (sample_time, integration_method, rk4_steps)
        self.use_nn_flag, self.nn_residual = (use_nn_flag, nn_residual)
        lower, upper = (np.asarray(physics_parameter_box['lower']), np.asarray(physics_parameter_box['upper']))
        groups = {
            'B': slice(0, 2),
            'C': slice(2, 4),
            'D': slice(4, 6),
            'E': slice(6, 8),
            'SH': slice(8, 10),
            'SV': slice(10, 12),
            'Cm1': 12,
            'Cm2': 13,
            'Cr0': 14,
            'Cd': 15,
            'I': 16
        }
        for name, index in groups.items():
            setattr(self, name + '_min', lower[index])
            setattr(self, name + '_max', upper[index])
            setattr(self, name + '_raw', nn.Parameter(torch.zeros(np.shape(lower[index]))))
        initial = physics_parameter_box['nominal'] if physics_param_init is None else physics_param_init
        self.set_physics_params(initial)
        self.sbar = nn.Parameter(torch.zeros(3))
        self.Sigmainv_raw = nn.Parameter(self._inv_softplus_map(torch.ones(3)))

    def f_continuous(self, x, u):
        vx, vy, yaw_rate = (x[:, 0], x[:, 1], x[:, 2])
        delta, throttle = (u[:, 0], u[:, 1])
        longitudinal_speed = torch.abs(vx)
        sigma_12_y = delta - torch.atan2(yaw_rate * self.l1 + vy, longitudinal_speed)
        sigma_34_y = torch.atan2(yaw_rate * self.l2 - vy, longitudinal_speed)
        F_tyre_12_x = torch.zeros_like(vx)
        F_tyre_12_y = self.magic_formula(sigma_12_y, 0)
        F_tyre_34_x = self.drivetrain_formula(vx, throttle)
        F_tyre_34_y = self.magic_formula(sigma_34_y, 1)
        if self.use_nn_flag:
            dF = self.nn_residual(torch.cat((x, u), dim=1))
            F_tyre_12_x, F_tyre_12_y = (F_tyre_12_x + dF[:, 0], F_tyre_12_y + dF[:, 1])
            F_tyre_34_x, F_tyre_34_y = (F_tyre_34_x + dF[:, 2], F_tyre_34_y + dF[:, 3])
        c_delta, s_delta = (torch.cos(delta), torch.sin(delta))
        F_tyre_body_1_x = F_tyre_12_x * c_delta - F_tyre_12_y * s_delta
        F_tyre_body_1_y = F_tyre_12_x * s_delta + F_tyre_12_y * c_delta
        ax = (F_tyre_body_1_x + F_tyre_34_x) / self.m
        ay = (F_tyre_body_1_y + F_tyre_34_y) / self.m
        yaw_acceleration = (self.l1 * F_tyre_body_1_y - self.l2 * F_tyre_34_y) / self.I
        return torch.stack((ax + vy * yaw_rate, ay - vx * yaw_rate, yaw_acceleration), dim=1)

    def magic_formula(self, sigma, idx):
        """
        Standard Pacejka lateral MF (pure slip) in wheel frame.
        Fy = D sin(C arctan(B sigmabar - E (B sigmabar - arctan(B sigmabar)))) + SV
        sigmabar = sigma + SH;  D is peak force [N], not μ.
        idx: 0 -> y_12, 1 -> y_34
        """
        B = self.B[idx]
        C = self.C[idx]
        D = self.D[idx]
        E = self.E[idx]
        SH = self.SH[idx]
        SV = self.SV[idx]
        sigmabar = sigma + SH
        return D * torch.sin(C * torch.atan(B * sigmabar - E * (B * sigmabar - torch.atan(B * sigmabar)))) + SV

    def drivetrain_formula(self, vx, throttle):
        """
        Rear-axle longitudinal force (DDM).
        vx: [batch], throttle: [batch]  (T, negative if braking)
        returns: F_rx [batch]
        """
        return (self.Cm1 - self.Cm2 * vx) * throttle - self.Cr0 - self.Cd * vx ** 2

    def set_train_flag(self, mode):
        physics, neural = {"physics": (True, False), "nn": (False, True),
                           "hybrid": (True, True)}[mode]
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(neural if name.startswith("nn_residual.") else physics)
        self.sbar.requires_grad_(True)
        self.Sigmainv_raw.requires_grad_(True)
        return self

    @staticmethod
    def _inv_softplus_map(value):
        value = torch.as_tensor(value, dtype=torch.float32)
        value = torch.clamp(value, min=1e-06)
        return torch.log(torch.expm1(value))

    @staticmethod
    def _softplus_map(raw):
        return nn.functional.softplus(raw)

    @staticmethod
    def _inv_sigmoid_map(value, lo, hi):
        value = torch.as_tensor(value, dtype=torch.float32)
        lo = torch.as_tensor(lo, dtype=value.dtype, device=value.device)
        hi = torch.as_tensor(hi, dtype=value.dtype, device=value.device)
        s = torch.clamp((value - lo) / (hi - lo), 1e-06, 1.0 - 1e-06)
        return torch.log(s / (1.0 - s))

    @staticmethod
    def _sigmoid_map(raw, lo, hi):
        lo = torch.as_tensor(lo, dtype=raw.dtype, device=raw.device)
        hi = torch.as_tensor(hi, dtype=raw.dtype, device=raw.device)
        return lo + (hi - lo) * torch.sigmoid(raw)

    @property
    def B(self):
        return self._sigmoid_map(self.B_raw, self.B_min, self.B_max)

    @property
    def C(self):
        return self._sigmoid_map(self.C_raw, self.C_min, self.C_max)

    @property
    def D(self):
        return self._sigmoid_map(self.D_raw, self.D_min, self.D_max)

    @property
    def E(self):
        return self._sigmoid_map(self.E_raw, self.E_min, self.E_max)

    @property
    def SH(self):
        return self._sigmoid_map(self.SH_raw, self.SH_min, self.SH_max)

    @property
    def SV(self):
        return self._sigmoid_map(self.SV_raw, self.SV_min, self.SV_max)

    @property
    def Cm1(self):
        return self._sigmoid_map(self.Cm1_raw, self.Cm1_min, self.Cm1_max)

    @property
    def Cm2(self):
        return self._sigmoid_map(self.Cm2_raw, self.Cm2_min, self.Cm2_max)

    @property
    def Cr0(self):
        return self._sigmoid_map(self.Cr0_raw, self.Cr0_min, self.Cr0_max)

    @property
    def Cd(self):
        return self._sigmoid_map(self.Cd_raw, self.Cd_min, self.Cd_max)

    @property
    def Cr2(self):
        return self.Cd

    @property
    def I(self):
        return self._sigmoid_map(self.I_raw, self.I_min, self.I_max)

    @property
    def Sigmainv(self):
        return self._softplus_map(self.Sigmainv_raw)

    def set_physics_params(self, physics_param_value):
        values = torch.as_tensor(
            physics_param_value,
            dtype=self.B_raw.dtype,
            device=self.B_raw.device
        ).reshape(17)
        with torch.no_grad():
            self.B_raw.copy_(self._inv_sigmoid_map(values[0:2], self.B_min, self.B_max))
            self.C_raw.copy_(self._inv_sigmoid_map(values[2:4], self.C_min, self.C_max))
            self.D_raw.copy_(self._inv_sigmoid_map(values[4:6], self.D_min, self.D_max))
            self.E_raw.copy_(self._inv_sigmoid_map(values[6:8], self.E_min, self.E_max))
            self.SH_raw.copy_(self._inv_sigmoid_map(values[8:10], self.SH_min, self.SH_max))
            self.SV_raw.copy_(self._inv_sigmoid_map(values[10:12], self.SV_min, self.SV_max))
            self.Cm1_raw.copy_(self._inv_sigmoid_map(values[12], self.Cm1_min, self.Cm1_max))
            self.Cm2_raw.copy_(self._inv_sigmoid_map(values[13], self.Cm2_min, self.Cm2_max))
            self.Cr0_raw.copy_(self._inv_sigmoid_map(values[14], self.Cr0_min, self.Cr0_max))
            self.Cd_raw.copy_(self._inv_sigmoid_map(values[15], self.Cd_min, self.Cd_max))
            self.I_raw.copy_(self._inv_sigmoid_map(values[16], self.I_min, self.I_max))
        return self

    @property
    def physics_param_value(self):
        """Constrained tire and drivetrain parameters in a stable order.

        Order: B[2], C[2], D[2], E[2], SH[2], SV[2], Cm1, Cm2, Cr0, Cr2, Iz.
        """
        return torch.cat((
            self.B.reshape(-1),
            self.C.reshape(-1),
            self.D.reshape(-1),
            self.E.reshape(-1),
            self.SH.reshape(-1),
            self.SV.reshape(-1),
            self.Cm1.reshape(-1),
            self.Cm2.reshape(-1),
            self.Cr0.reshape(-1),
            self.Cd.reshape(-1),
            self.I.reshape(-1)
        ))


class CarModelCasadi:

    def __init__(self, **kwargs):
        self.m = float(kwargs.get('m', 0.041))
        self.l1 = float(kwargs.get('l1', 0.029))
        self.l2 = float(kwargs.get('l2', 0.033))
        self.l3 = float(kwargs.get('l3', 0.8))
        self.g = 9.81
        self.process_noise_matrix = ca.DM(np.asarray(kwargs['E'], dtype=float))
        self.sample_time = float(kwargs['T'])
        self.integration_method = kwargs.get('integration_method', 'euler')
        self.rk4_steps = int(kwargs.get('rk4_steps', 2))
        self.physics_param_sym = ca.MX.sym('theta', 17)
        self.update_physics_params(kwargs['physics_param_value'])
        self.use_nn_flag = kwargs['use_nn_flag']
        self.NN_residual = kwargs['NN_residual']
        self.mhe_nn_backend = kwargs['mhe_nn_backend']
        self.name = 'car_dyn_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        self.NN_casadi = None
        if self.use_nn_flag:
            self.NN_casadi = MLP_Residual_casadi(self.NN_residual)

    def f_continuous(self, x, u, w):
        vx, vy, yaw_rate = (x[0], x[1], x[2])
        delta, throttle = (u[0], u[1])
        longitudinal_speed = ca.fabs(vx)
        sigma_12_y = delta - ca.atan2(yaw_rate * self.l1 + vy, longitudinal_speed)
        sigma_34_y = ca.atan2(yaw_rate * self.l2 - vy, longitudinal_speed)
        F_tyre_12_x = ca.MX(0)
        F_tyre_12_y = self.magic_formula(sigma_12_y, 0)
        F_tyre_34_x = self.drivetrain_formula(vx, throttle)
        F_tyre_34_y = self.magic_formula(sigma_34_y, 1)
        if self.use_nn_flag:
            nn_input = self.nn_input_casadi(x, u)
            dF = self.NN_casadi(nn_input)
            F_tyre_12_x, F_tyre_12_y = (F_tyre_12_x + dF[0], F_tyre_12_y + dF[1])
            F_tyre_34_x, F_tyre_34_y = (F_tyre_34_x + dF[2], F_tyre_34_y + dF[3])
        c_delta, s_delta = (ca.cos(delta), ca.sin(delta))
        F_tyre_body_1_x = F_tyre_12_x * c_delta - F_tyre_12_y * s_delta
        F_tyre_body_1_y = F_tyre_12_x * s_delta + F_tyre_12_y * c_delta
        ax = (F_tyre_body_1_x + F_tyre_34_x) / self.m
        ay = (F_tyre_body_1_y + F_tyre_34_y) / self.m
        yaw_acceleration = (self.l1 * F_tyre_body_1_y - self.l2 * F_tyre_34_y) / self.physics_param_sym[16]
        return ca.vertcat(
            ax + vy * yaw_rate,
            ay - vx * yaw_rate,
            yaw_acceleration
        ) + self.process_noise_matrix @ w

    def magic_formula(self, sigma, idx):
        """
        Lateral Pacejka force in wheel frame.
        idx: 0 -> y_12, 1 -> y_34
        """
        B = self.physics_param_sym[idx]
        C = self.physics_param_sym[2 + idx]
        D = self.physics_param_sym[4 + idx]
        E = self.physics_param_sym[6 + idx]
        SH = self.physics_param_sym[8 + idx]
        SV = self.physics_param_sym[10 + idx]
        sigma_bar = sigma + SH
        return D * ca.sin(C * ca.atan(B * sigma_bar - E * (B * sigma_bar - ca.atan(B * sigma_bar)))) + SV

    def drivetrain_formula(self, vx, throttle):
        """Rear-axle longitudinal force (DDM)."""
        Cm1, Cm2, Cr0, Cd = [self.physics_param_sym[i] for i in range(12, 16)]
        return (Cm1 - Cm2 * vx) * throttle - Cr0 - Cd * vx ** 2

    def nn_input_casadi(self, x, u):
        return ca.vertcat(x, u)

    def get_acados_p(self):
        nn_symbols = self.NN_casadi.get_sym_params() if self.use_nn_flag else ca.MX.sym('nn', 0)
        nn_values = self.NN_casadi.get_params() if self.use_nn_flag else np.array([])
        return (self.physics_param_sym, self.physics_param_value, nn_symbols, nn_values)

    def update_physics_params(self, physics_param_value):
        self._physics_param_value = np.asarray(physics_param_value, dtype=float).reshape(-1)

    @property
    def physics_param_value(self):
        return self._physics_param_value


class NeuralForceModel(CarDynamicsPytorch):
    """Use the single-track state equations with NN-only tire-frame forces."""

    def magic_formula(self, sigma, idx):
        return torch.zeros_like(sigma)

    def drivetrain_formula(self, vx, throttle):
        return torch.zeros_like(vx)


class NeuralForceModelCasadi(CarModelCasadi):
    """CasADi counterpart of the pure neural-force model."""

    def magic_formula(self, sigma, idx):
        return ca.MX(0)

    def drivetrain_formula(self, vx, throttle):
        return ca.MX(0)


TYRE_PARAM_NAMES = CarDynamicsPytorch.physics_param_names
TRUE_PHYSICS_PARAMS = np.array([
    5.579, 5.3852, 1.2, 1.2691, 0.192, 0.1737, -0.083, -0.019,
    -0.0013, -0.00376, 0.00043, 0.00091, 0.287, 0.0545, 0.0518, 0.00035, 27.8e-6,
])


def car_physics_raw_params(model):
    return [model.B_raw, model.C_raw, model.D_raw, model.E_raw, model.SH_raw, model.SV_raw,
            model.Cm1_raw, model.Cm2_raw, model.Cr0_raw, model.Cd_raw, model.I_raw]


def parameter_dict(model):
    values = model.physics_param_value.detach().cpu().numpy()
    return dict(zip(TYRE_PARAM_NAMES, values))


def set_arrival_initialization(model, sbar, Sigmainv):
    with torch.no_grad():
        model.sbar.copy_(torch.as_tensor(sbar, dtype=model.sbar.dtype, device=model.sbar.device))
        value = torch.as_tensor(Sigmainv, dtype=model.Sigmainv_raw.dtype, device=model.Sigmainv_raw.device)
        model.Sigmainv_raw.copy_(value + torch.log(-torch.expm1(-value)))


def parameter_box(initialization):
    return {"names": initialization["parameter_names"],
            "lower": initialization["parameter_lower"],
            "upper": initialization["parameter_upper"],
            "nominal": initialization["parameter_initial"]}


def make_vehicle(system, initialization, residual=None):
    model = CarDynamicsPytorch(
        m=system["m"], l1=system["l1"], l2=system["l2"],
        sample_time=system["T"], integration_method=system["integration"],
        rk4_steps=system["rk4_steps"], nn_residual=residual,
        use_nn_flag=residual is not None, physics_parameter_box=parameter_box(initialization),
        physics_param_init=initialization["parameter_initial"])
    if "mhe_sbar" in initialization:
        set_arrival_initialization(model, initialization["mhe_sbar"], initialization["mhe_Sigmainv"])
    return model


def load_authors_model(iac=False):
    package_dir = ROOT / 'repos/DeepDynamicsModel/deep_dynamics'
    for name, path in (('deep_dynamics', package_dir), ('deep_dynamics.model', package_dir / 'model')):
        package = types.ModuleType(name)
        package.__path__ = [str(path)]
        sys.modules[name] = package
    for name in ('build_network', 'models'):
        full_name = 'deep_dynamics.model.' + name
        spec = importlib.util.spec_from_file_location(full_name, package_dir / 'model' / (name + '.py'))
        module = importlib.util.module_from_spec(spec)
        sys.modules[full_name] = module
        spec.loader.exec_module(module)
    module = sys.modules['deep_dynamics.model.models']
    return module.DeepDynamicsModelIAC if iac else module.DeepDynamicsModel
