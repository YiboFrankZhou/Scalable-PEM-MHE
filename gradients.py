"""Temporal KKT blocks, block adjoints, and explicit/adjoint–VJP hypergradients."""
import numpy as np
import torch

from estimation import calculate_MHE_constraint, calculate_MHE_loss

JACOBIAN_CHUNK_MIB = 128


def build_KKT_blocks_batch(params, model, xs, ws, us, ys):
    """Gauss–Newton KKT, ordered as (x_k, w_k, mu_k), then x_N."""
    B, N, nx, nw = xs.shape[0], params["mhe_window"], params["nx"], params["nw"]
    C, Qinv, Rinv = params["C_torch"], params["mhe_Qinv_torch"], params["mhe_Rinv_torch"]

    def transition(x, w, u):
        return model.discrete(x[None], u[None], w[None], params["E_torch"],
                              T=params["T"], rk4_steps=params.get("rk4_steps", model.rk4_steps))[0]

    dfdx, dfdw = torch.func.vmap(torch.func.jacrev(transition, argnums=(0, 1)))(
        xs[:, :-1].reshape(B*N, nx), ws.reshape(B*N, nw), us.reshape(B*N, -1))
    dfdx, dfdw = dfdx.reshape(B, N, nx, nx), dfdw.reshape(B, N, nx, nw)
    primal_size, stage_size = nx+nw, 2*nx+nw
    hessian = xs.new_zeros(B, N, primal_size, primal_size)
    hessian[:, 0, :nx, :nx] = 2*torch.diag(model.Sigmainv)
    hessian[:, 1:, :nx, :nx] = 2*C.T@Rinv@C
    hessian[:, :, nx:, nx:] = 2*Qinv
    constraint_jacobian = torch.cat((-dfdx, -dfdw), dim=-1)

    diagonal = xs.new_zeros(B, N, stage_size, stage_size)
    diagonal[:, :, :primal_size, :primal_size] = hessian
    diagonal[:, :, :primal_size, primal_size:] = constraint_jacobian.transpose(-1, -2)
    diagonal[:, :, primal_size:, :primal_size] = constraint_jacobian
    X = [diagonal[:, k] for k in range(N)] + [xs.new_zeros(B, nx, nx)]

    upper = xs.new_zeros(B, N-1, stage_size, stage_size)
    terminal_upper = xs.new_zeros(B, stage_size, nx)
    identity = torch.eye(nx, dtype=xs.dtype, device=xs.device)
    upper[:, :, primal_size:, :nx] = identity
    terminal_upper[:, primal_size:, :] = identity
    Y = [upper[:, k] for k in range(N-1)] + [terminal_upper]
    Z = [block.transpose(-1, -2) for block in Y]
    b = [xs.new_zeros(B, stage_size) for _ in range(N)]
    b.append((xs[:, -1]@C.T-ys[:, -1])@C)
    return X, Y, Z, b


def solve_block_tridiagonal_batch(X, Y, Z, b):
    """Forward elimination followed by backward substitution in K lambda = b."""
    N = len(X)-1
    diagonal, rhs = [X[0]], [b[0]]
    for k in range(1, N+1):
        gain = torch.linalg.solve(diagonal[k-1], Y[k-1])
        previous_rhs = torch.linalg.solve(diagonal[k-1], rhs[k-1].unsqueeze(-1))
        diagonal.append(X[k]-Z[k-1]@gain)
        rhs.append(b[k]-(Z[k-1]@previous_rhs).squeeze(-1))
    adjoint = [None]*(N+1)
    adjoint[N] = torch.linalg.solve(diagonal[N], rhs[N].unsqueeze(-1)).squeeze(-1)
    for k in reversed(range(N)):
        corrected_rhs = rhs[k]-(Y[k]@adjoint[k+1].unsqueeze(-1)).squeeze(-1)
        adjoint[k] = torch.linalg.solve(diagonal[k], corrected_rhs.unsqueeze(-1)).squeeze(-1)
    return torch.cat(adjoint, dim=-1)


def adjoint_decomposition_batch(params, adjoint):
    nx, nw, N = params["nx"], params["nw"], params["mhe_window"]
    stages = adjoint[:, :N*(2*nx+nw)].reshape(-1, N, 2*nx+nw)
    lambda_x = torch.cat((stages[:, :, :nx], adjoint[:, -nx:].unsqueeze(1)), dim=1)
    return lambda_x, stages[:, :, nx:nx+nw], stages[:, :, nx+nw:]


def build_KKT_blocks(params, model, xs, ws, us, ys):
    groups = build_KKT_blocks_batch(params, model, xs[None], ws[None], us[None], ys[None])
    return tuple([block[0] for block in group] for group in groups)


def solve_block_tridiagonal(X, Y, Z, b):
    return solve_block_tridiagonal_batch(*([block[None] for block in group] for group in (X,Y,Z,b)))[0]


def adjoint_decomposition(params, adjoint):
    return tuple(value[0] for value in adjoint_decomposition_batch(params, adjoint[None]))


def active_parameters(model):
    return tuple((parameter for parameter in model.parameters() if parameter.requires_grad))


def assemble_dense_kkt(X_blocks, Y_blocks, Z_blocks):
    sizes = [block.shape[0] for block in X_blocks]
    offsets = np.cumsum([0] + sizes)
    K = X_blocks[0].new_zeros((int(offsets[-1]), int(offsets[-1])))
    for k, Xk in enumerate(X_blocks):
        K[offsets[k]:offsets[k + 1], offsets[k]:offsets[k + 1]] = Xk
        if k < len(Y_blocks):
            K[offsets[k]:offsets[k + 1], offsets[k + 1]:offsets[k + 2]] = Y_blocks[k]
            K[offsets[k + 1]:offsets[k + 2], offsets[k]:offsets[k + 1]] = Z_blocks[k]
    return K


def lagrangian_stationarity_vector(params, model, xhats, whats, mus, us, ys):
    N, nx, nw = (params['mhe_window'], params['nx'], params['nw'])
    x_vec = xhats.detach().clone().reshape(-1).requires_grad_(True)
    w_vec = whats.detach().clone().reshape(-1).requires_grad_(True)
    x_mat = x_vec.view(N + 1, nx)
    w_mat = w_vec.view(N, nw)
    mhe_loss = calculate_MHE_loss(params, model, x_mat, w_mat, us, ys)
    H_mat = calculate_MHE_constraint(params, model, x_mat, w_mat, us, ys).view(N, nx)
    mhe_lagrangian = mhe_loss + torch.sum(mus.view(N, nx) * H_mat)
    grad_x, grad_w = torch.autograd.grad(mhe_lagrangian, (x_vec, w_vec), create_graph=True)
    grad_x = grad_x.view(N + 1, nx)
    grad_w = grad_w.view(N, nw)
    stage_terms = torch.cat((grad_x[:N], grad_w, H_mat), dim=1).reshape(-1)
    return torch.cat((stage_terms, grad_x[N]), dim=0)


def flatten_parameter_grads(parameters, grads):
    pieces = []
    for parameter, grad in zip(parameters, grads):
        if grad is None:
            pieces.append(torch.zeros_like(parameter).reshape(-1))
        else:
            pieces.append(grad.reshape(-1))
    return torch.cat(pieces).detach()


def jacobian_chunk_size(n_inner, n_parameters):
    bytes_per_row = n_parameters * torch.finfo(torch.float32).bits // 8
    max_bytes = int(JACOBIAN_CHUNK_MIB * 1024 ** 2)
    return max(1, min(n_inner, max_bytes // max(bytes_per_row, 1)))


def gradient_adjoint_vjp(params, model, xhats, whats, mus, us, ys):
    model.zero_grad(set_to_none=True)
    X_blocks, Y_blocks, Z_blocks, b_blocks = build_KKT_blocks(params, model, xhats, whats, us, ys)
    lambda_adjoint = solve_block_tridiagonal(X_blocks, Y_blocks, Z_blocks, b_blocks).detach()
    stationarity = lagrangian_stationarity_vector(params, model, xhats, whats, mus, us, ys)
    surrogate = -torch.dot(stationarity, lambda_adjoint)
    parameters = active_parameters(model)
    grads = torch.autograd.grad(surrogate, parameters, allow_unused=True)
    gradient = flatten_parameter_grads(parameters, grads)
    residual = params['C_torch'] @ xhats[-1] - ys[-1]
    cost = 0.5 * torch.dot(residual, residual)
    return (cost.detach(), gradient)


def gradient_explicit(params, model, xhats, whats, mus, us, ys):
    model.zero_grad(set_to_none=True)
    X_blocks, Y_blocks, Z_blocks, b_blocks = build_KKT_blocks(params, model, xhats, whats, us, ys)
    dense_kkt = assemble_dense_kkt(X_blocks, Y_blocks, Z_blocks).detach()
    inverse_kkt = torch.linalg.inv(dense_kkt)
    stationarity = lagrangian_stationarity_vector(params, model, xhats, whats, mus, us, ys)
    n_inner = stationarity.numel()
    parameters = active_parameters(model)
    n_parameters = sum((parameter.numel() for parameter in parameters))
    outer_gradient_v = torch.cat(b_blocks).detach()
    left_sensitivity = -(outer_gradient_v @ inverse_kkt)
    chunk_rows = jacobian_chunk_size(n_inner, n_parameters)
    identity = torch.eye(n_inner, dtype=stationarity.dtype, device=stationarity.device)
    gradient_parts = [torch.zeros(parameter.numel(), dtype=stationarity.dtype) for parameter in parameters]
    for start in range(0, n_inner, chunk_rows):
        stop = min(start + chunk_rows, n_inner)
        jacobian_parts = torch.autograd.grad(
            stationarity,
            parameters,
            grad_outputs=identity[start:stop],
            is_grads_batched=True,
            retain_graph=stop < n_inner,
            allow_unused=True
        )
        coefficients = left_sensitivity[start:stop]
        for i, (parameter, jacobian_part) in enumerate(zip(parameters, jacobian_parts)):
            if jacobian_part is not None:
                B_chunk = jacobian_part.reshape(stop - start, parameter.numel())
                gradient_parts[i] += coefficients @ B_chunk
    gradient = torch.cat(gradient_parts).detach()
    residual = params['C_torch'] @ xhats[-1] - ys[-1]
    cost = 0.5 * torch.dot(residual, residual)
    return (cost.detach(), gradient, n_inner, chunk_rows)
