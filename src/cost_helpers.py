import jax
import jax.numpy as jnp
import jaxopt
from jaxopt import LBFGS
import flax
from flax.core.scope import VariableDict

from ott.geometry import costs

def get_h_cost_fn(model, params, alpha=0.001):
    def cost_fn(x):
        cost = model.apply(params, x[None,:])
        print('cost', cost)
        return jnp.sum(cost[0]) + alpha*jnp.sum(x*x)
    return cost_fn

def get_grad_h_cost_fn(h_cost_fn):
    grad_fn = lambda x: jax.grad(lambda x: jnp.sum(h_cost_fn(x)))(
            x
        )
    return grad_fn

def get_grad_h_inverse_fn(h, tol=1e-5, stepsize=1e-1, maxiter=10_000, history_size=5,
                        use_gamma=True):
    @jax.jit
    def grad_inverse_cost(outer_y, init_x):
        def closure(x, inner_y):
            loss = h(x) - jnp.sum(x * inner_y)
            return loss

        
        lbfgs = jaxopt.LBFGS(fun=closure, tol=tol, stepsize=stepsize, maxiter=maxiter, 
                      history_size=history_size, use_gamma=use_gamma)
        out, _ = lbfgs.run(init_x, inner_y=outer_y)
        return out
    return grad_inverse_cost
