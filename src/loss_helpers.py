import jax
import jax.numpy as jnp
import jax.scipy as jsp
from typing import Callable
import functools
import jaxopt

def all_pairs_pairwise(cost_fn, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Compute matrix of all pairwise costs, excluding the :attr:`norms <norm>`.

    Args:
      cost_fn: cost_fn x,y -> R
      x: Array of shape ``[n, ...]``.
      y: Array of shape ``[m, ...]``.

    Returns:
      Array of shape ``[n, m]`` of cost evaluations.
    """
    return jax.vmap(lambda x_: jax.vmap(lambda y_: cost_fn(x_-y_))(y))(x)

def get_h_cost_fn(model, params):
    def func(x):
        cost = model.apply(params, x)
        return jnp.sum(cost)
    return func

def potential_fn(input_x: jnp.ndarray, 
                 potential_g: jnp.ndarray, 
                 y: jnp.ndarray, 
                 weights_b: jnp.ndarray, 
                 epsilon: float,
                 cost_fn):

    input_x = jnp.atleast_2d(input_x)
    assert input_x.shape[-1] == y.shape[-1], (input_x.shape, y.shape)
    cost_matrix = all_pairs_pairwise(cost_fn, input_x, y)
    z = (potential_g - cost_matrix) / epsilon
    lse = -epsilon * jsp.special.logsumexp(z, b=weights_b, axis=-1)
    return jnp.squeeze(lse)

def get_grad_h_inverse_fn_lbfgs(cost_model):
    @jax.jit
    def grad_inverse_cost(outer_y, init_x, params):
        def closure(x, inner_y, inner_params):
            cost_fn = get_h_cost_fn(model=cost_model, params=inner_params)
            loss = cost_fn(x) - jnp.sum(x * inner_y)
            return loss
        
        lbfgs = jaxopt.LBFGS(fun=closure, tol=1e-6, stepsize=1e-3, maxiter=10000, history_size=5,
                        use_gamma=True)

        out, _ = lbfgs.run(init_x, inner_y=outer_y, inner_params=params)
        return out
    return grad_inverse_cost

def get_grad_h_inverse_fn(cost_model, tol=1e-5):
    @jax.jit
    def grad_inverse_cost(outer_y, init_x, params):
        def closure(x, inner_y, inner_params):
            cost_fn = get_h_cost_fn(model=cost_model, params=inner_params)
            loss = cost_fn(x) - jnp.sum(x * inner_y)
            return loss
    
        gd_opt = jaxopt.GradientDescent(fun=closure, tol=tol)
        out, state = gd_opt.run(init_x, inner_y=outer_y, inner_params=params)
        return out, state
    return grad_inverse_cost

def get_grad_f(dual_f) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Vectorized gradient of the potential function :attr:`f`."""
    return jax.vmap(jax.grad(dual_f, argnums=0))