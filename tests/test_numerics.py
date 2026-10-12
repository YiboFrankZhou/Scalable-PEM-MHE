"""Numerical checks for the dimension-independent solver and hypergradient."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import configure_runtime
configure_runtime()
import numpy as np
import torch
import casadi as ca
from models import Dynamics
from estimation import MHEConfig, ObserverMHEAcados
import gradients
from gradients import assemble_dense_kkt, gradient_adjoint_vjp, gradient_explicit


class Numerics(unittest.TestCase):
    def test_other_dimensions_and_dense_adjoint(self):
        torch.manual_seed(5)
        for nx, nu, nw, ny in [(2, 1, 1, 1), (4, 2, 2, 2)]:
            N, B = 5, 2
            E = torch.eye(nx)[:, :nw]
            C = torch.eye(nx)[:ny]
            A = -.2*torch.eye(nx)
            U = torch.randn(nx, nu)*.1
            model = Dynamics(lambda x,u,t: x@A.T+u@U.T+t[0]*x,
                torch.tensor([.05]), torch.zeros(nx), torch.ones(nx), .04)
            params = dict(nx=nx, nu=nu, nw=nw, ny=ny, mhe_window=N, T=.04,
                E_torch=E, C_torch=C, mhe_Qinv_torch=torch.eye(nw), mhe_Rinv_torch=torch.eye(ny))
            xs,ws,us,ys = torch.randn(B,N+1,nx),torch.randn(B,N,nw),torch.randn(B,N,nu),torch.randn(B,N,ny)
            blocks=gradients.build_KKT_blocks_batch(params,model,xs,ws,us,ys)
            adjoint=gradients.solve_block_tridiagonal_batch(*blocks)
            for i in range(B):
                X,Y,Z,b=([value[i] for value in group] for group in blocks)
                dense=assemble_dense_kkt(X,Y,Z)
                expected=torch.linalg.solve(dense,torch.cat(b))
                torch.testing.assert_close(adjoint[i],expected,rtol=2e-4,atol=2e-4)
            mus=torch.randn(N,nx)
            _,vjp=gradient_adjoint_vjp(params,model,xs[0],ws[0],mus,us[0],ys[0])
            _,explicit,_,_=gradient_explicit(params,model,xs[0],ws[0],mus,us[0],ys[0])
            torch.testing.assert_close(vjp,explicit,rtol=2e-4,atol=2e-5)

    def test_hypergradient_against_differentiable_linear_solve(self):
        from estimation import calculate_MHE_loss, calculate_MHE_constraint
        torch.manual_seed(9)
        nx, nw, N = 2, 1, 4
        model = Dynamics(
            lambda x, u, theta: torch.stack((theta[0]*x[:, 0]+x[:, 1], u[:, 0]), dim=1),
            torch.tensor([-.3]), torch.zeros(nx), torch.ones(nx), .1).double()
        params = dict(nx=nx, nu=1, nw=nw, ny=1, mhe_window=N, T=.1,
                      E_torch=torch.tensor([[0.], [1.]], dtype=torch.float64),
                      C_torch=torch.tensor([[1., 0.]], dtype=torch.float64),
                      mhe_Qinv_torch=torch.eye(nw, dtype=torch.float64),
                      mhe_Rinv_torch=torch.eye(1, dtype=torch.float64))
        u, y = torch.randn(N, 1, dtype=torch.float64), torch.randn(N, 1, dtype=torch.float64)
        n_states = (N+1)*nx
        origin = torch.zeros(n_states+N*nw, dtype=torch.float64)

        def unpack(z):
            return z[:n_states].reshape(N+1, nx), z[n_states:].reshape(N, nw)

        objective = lambda z: calculate_MHE_loss(params, model, *unpack(z), u, y)
        constraints = lambda z: calculate_MHE_constraint(params, model, *unpack(z), u, y).flatten()
        hessian = torch.func.hessian(objective)(origin)
        A = torch.func.jacrev(constraints)(origin)
        zero = torch.zeros(N*nx, N*nx, dtype=torch.float64)
        K = torch.cat((torch.cat((hessian, A.T), dim=1), torch.cat((A, zero), dim=1)))
        rhs = -torch.cat((torch.func.grad(objective)(origin), constraints(origin)))
        solution = torch.linalg.solve(K, rhs)
        xs, ws = unpack(solution[:len(origin)])
        loss = .5*(xs[-1]@params['C_torch'].T-y[-1]).square().sum()
        expected = torch.cat([g.flatten() for g in torch.autograd.grad(loss, tuple(model.parameters()))])
        mus = solution[len(origin):].reshape(N, nx).detach()
        _, actual = gradient_adjoint_vjp(params, model, xs.detach(), ws.detach(), mus, u, y)
        torch.testing.assert_close(actual, expected, rtol=1e-8, atol=1e-10)

    def test_generic_solver_and_held_out_target(self):
        theta=np.array([-.4,.3])
        symbol=ca.MX.sym("theta",2)
        f=lambda x,u,w: ca.vertcat(symbol[0]*x[0]+x[1],symbol[1]*u[0]+w[0])
        cfg=MHEConfig(nx=2,nu=1,nw=1,horizon=5,sample_time=.1,
            C=np.array([[1.,0.]]),E=np.array([[0.],[1.]]),Qinv=np.eye(1),Rinv=np.eye(1),
            sbar=np.zeros(2),Sigmainv=np.ones(2),f=f,theta=symbol,theta_values=lambda:theta,
            batch_size=2,build_dir=Path(__file__).resolve().parents[1]/"build")
        observer=ObserverMHEAcados(cfg)
        u=np.full((2,5,1),.2);y=np.linspace(.01,.05,5)[None,:,None].repeat(2,axis=0)
        seed=np.zeros((2,6,2))
        first=observer.estimate_batch_once(u,y,seed)[0][0]
        changed=y.copy();changed[:,-1]+=100
        second=observer.estimate_batch_once(u,changed,seed)[0][0]
        np.testing.assert_allclose(first,second,atol=1e-10)
        self.assertEqual(first.shape,(2,6,2))
        self.assertTrue(np.isfinite(first).all())


if __name__ == "__main__":
    unittest.main()
