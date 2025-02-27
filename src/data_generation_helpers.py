import jax
import jax.numpy as jnp
from jaxopt import LBFGS, GradientDescent

jax.config.update('jax_platform_name', 'gpu')
jax.config.update("jax_enable_x64", True)

def get_implicit_x_layer(true_h, concave_func):
  '''
  Get the argmin function for the c-transform
  '''
  @jax.jit
  def implicit_layer(outer_y):
    def closure(x, inner_y):
      vec = x - inner_y
      loss = true_h(vec) - concave_func(x)
      return loss

    init_x = outer_y
    lbfgs = LBFGS(fun=closure, tol=1e-8, stepsize=1e-3, maxiter=100000, history_size=5,
                    use_gamma=True)
    
    # lbfgs = GradientDescent(fun=closure, tol=1e-8, maxiter=100000)

    out, state = lbfgs.run(init_x, inner_y=outer_y)
    return out, state
  return implicit_layer


def get_implicit_h_layer(true_h):
  '''
  Returns the (\nabla h)^{-1} function
  '''
  @jax.jit
  def implicit_layer(outer_y):
    def closure(x, inner_y):
      loss = true_h(x) - jnp.sum(x * inner_y)
      return loss

    init_x = outer_y
    lbfgs = LBFGS(fun=closure, tol=1e-8, stepsize=1e-3, maxiter=100000, history_size=5,
                    use_gamma=True)

    # lbfgs = GradientDescent(fun=closure, tol=1e-8, maxiter=100000)
    out, state = lbfgs.run(init_x, inner_y=outer_y)
    return out, state
  return implicit_layer


def get_mapping(true_h, concave_f):
    '''
    Given convex h and concave f, return a function for the induced mapping
    '''
    @jax.jit
    def T_mapping(x):
        
        x_tilde = get_implicit_x_layer(true_h, concave_f)
        x_tilde_val, x_state = jax.vmap(x_tilde)(x)
        grad_f_h = -jax.vmap(jax.grad(true_h))(x_tilde_val - x)

        h_tilde = get_implicit_h_layer(true_h)
        inv, h_state = jax.vmap(h_tilde)(grad_f_h)
        return x - inv, x_state, h_state
    
    return T_mapping


##########################################################

'''
Datasets code from Somnath et al. 2023 (https://arxiv.org/abs/2302.11419, https://github.com/vsomnath/aligned_diffusion_bridges)
'''

def rotate2d(x, radians):
    """Build a rotation matrix in 2D, take the dot product, and rotate."""
    c, s = jnp.cos(radians), jnp.sin(radians)
    j = jnp.array([[c, s], [-s, c]])
    m = jnp.dot(j, x)

    return m

def get_data_generator(problem):
    generator_cls = {
            "spiral": Spiral,
            "moon": Moon,
            "t_shape": T_shape,
            "cross": Cross
        }.get(problem)    
    return generator_cls()


class CheckerBoard:

    def sample(self, key, n_samples):
        n = n_samples
        n_points = 3 * n
        n_classes = 2
        freq = 5
        x = jax.random.uniform(key,
            minval=-(freq // 2) * jnp.pi, maxval=(freq // 2) * jnp.pi, shape=(n_points, n_classes)
        )
        mask = jnp.logical_or(
            jnp.logical_and(jnp.sin(x[:, 0]) > 0.0, jnp.sin(x[:, 1]) > 0.0),
            jnp.logical_and(jnp.sin(x[:, 0]) < 0.0, jnp.sin(x[:, 1]) < 0.0),
        )
        y = jnp.eye(n_classes)[1 * mask]
        x0 = x[:, 0] * y[:, 0]
        x1 = x[:, 1] * y[:, 0]
        sample = jnp.concatenate([x0[..., None], x1[..., None]], axis=-1)
        sqr = jnp.sum(jnp.square(sample), axis=-1)
        idxs = jnp.where(sqr == 0)[0]
        samples = jnp.delete(sample, idxs, axis=0)
        samples = samples[0:n, :]

        # transform dataset by adding constant shift
        samples_t = samples + jnp.array([0, 3])

        return {"mu": samples_t,
                "nu": samples}
    

class Spiral:

    def sample(self, key, n_samples):
        n = n_samples
        subkey1, subkey2 = jax.random.split(key)
        theta = (
            jnp.sqrt(jax.random.uniform(subkey1, shape=(n,))) * 3 * jnp.pi - 0.5 * jnp.pi
        )

        r_a = theta + jnp.pi
        data_a = jnp.array([jnp.cos(theta) * r_a, jnp.sin(theta) * r_a]).T
        x_a = data_a + 0.25 * jax.random.normal(subkey2, shape=(n,2))
        samples = jnp.append(x_a, jnp.zeros((n, 1)), axis=1)
        samples = samples[:, 0:2]

        # rotate and shrink samples
        samples_t = jnp.zeros(samples.shape)
        for i, v in enumerate(samples):
            samples_t = samples_t.at[i].set(rotate2d(v, 180))
        samples = samples * 0.8

        return {"mu": samples_t,
                "nu": samples}


class Moon:

    def sample(self, key, n_samples):
        n = n_samples
        subkey1, subkey2 = jax.random.split(key)

        x = jnp.linspace(0, jnp.pi, n // 2)
        u = jnp.stack([jnp.cos(x) + 0.5, -jnp.sin(x) + 0.2], axis=1) * 10.0
        u += 0.5 * jax.random.normal(subkey1, shape=u.shape)
        u /= 3
        v = jnp.stack([jnp.cos(x) - 0.5, jnp.sin(x) - 0.2], axis=1) * 10.0
        v += 0.5 * jax.random.normal(subkey2, shape=v.shape)
        v /= 3
        samples = jnp.concatenate([u, v], axis=0)

        # rotate and shrink samples
        samples_t = jnp.zeros(samples.shape)
        for i, v in enumerate(samples):
            samples_t = samples_t.at[i].set(rotate2d(v, 180))

        return {"mu": samples_t,
                "nu": samples}


class T_shape:

    def sample(self, key, n_samples):
        n = n_samples//2
        subkey1, subkey2, subkey3, subkey4, subkey5 = jax.random.split(key, 5)

        left_square = jnp.stack([jax.random.uniform(subkey1, minval=-1.2, maxval=-1., shape=(n,))*2-5, jnp.linspace(-.1, .5, n)*4+3], axis=1)
        right_square = left_square + jnp.array([14.4,0.0])

        top_square = jnp.stack([jnp.linspace(-.3, .3, n)*4, jax.random.uniform(subkey3, minval=.8, maxval=1., shape=(n,))*2+3], axis=1)
        bottom_square = top_square - jnp.array([0.0,10.1])

        rand_shuffling = jax.random.permutation(subkey5, n_samples)

        samples_t = jnp.concatenate([left_square, top_square], axis=0)[:n_samples][rand_shuffling]
        samples = jnp.concatenate([right_square, bottom_square], axis=0)[:n_samples][rand_shuffling]
    
        return {"mu": samples_t,
                "nu": samples}


class Cross:

    def sample(self, key, n_samples):
        n = n_samples//2
        subkey1, subkey2, subkey3, subkey4, subkey5 = jax.random.split(key, 5)

        left_square = jnp.stack([jax.random.uniform(subkey1, minval=-1.1, maxval=-0.9, shape=(n,))*2-5, jnp.linspace(-.3, .3, n)*4], axis=1)
        right_square = left_square + jnp.array([14.0,0.0])

        top_square = jnp.stack([jnp.linspace(-.3, .3, n)*4, jax.random.uniform(subkey3, minval=.9, maxval=1.1, shape=(n,))*2+3], axis=1)
        bottom_square = top_square - jnp.array([0.0,10.0])

        rand_shuffling = jax.random.permutation(subkey5, n_samples)

        samples_t = jnp.concatenate([left_square, top_square], axis=0)[:n_samples][rand_shuffling]
        samples = jnp.concatenate([right_square, bottom_square], axis=0)[:n_samples][rand_shuffling]
    
        return {"mu": samples_t,
                "nu": samples}