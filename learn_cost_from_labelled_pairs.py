import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from jax.example_libraries import stax
import flows

jax.config.update('jax_platform_name', 'gpu')
jax.config.update("jax_enable_x64", True)

import ott
from ott.geometry import pointcloud, geometry
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn
from ott.neural.networks import icnn
import optax
import jaxopt

import json

import pickle as pkl

from functools import partial

from tqdm import trange

from src.loss_helpers import all_pairs_pairwise, potential_fn, get_grad_f

######################## FLOWS #######################
'''
From jax-flows library (https://github.com/ChrisWaites/jax-flows)
'''

def get_masks(input_dim, hidden_dim=64, num_hidden=1):
    masks = []
    input_degrees = jnp.arange(input_dim)
    degrees = [input_degrees]

    for n_h in range(num_hidden + 1):
        degrees += [jnp.arange(hidden_dim) % (input_dim - 1)]
    degrees += [input_degrees % input_dim - 1]

    for (d0, d1) in zip(degrees[:-1], degrees[1:]):
        masks += [jnp.transpose(jnp.expand_dims(d1, -1) >= jnp.expand_dims(d0, 0)).astype(jnp.float32)]
    return masks

def masked_transform(rng, input_dim):
    masks = get_masks(input_dim, hidden_dim=64, num_hidden=1)
    act = stax.Relu
    init_fun, apply_fun = stax.serial(
        flows.MaskedDense(masks[0]),
        act,
        flows.MaskedDense(masks[1]),
        act,
        flows.MaskedDense(jnp.tile(masks[2], 2)),
    )
    _, params = init_fun(rng, (input_dim,))
    return params, apply_fun

##################### COST LEARNING ####################

class h_flow():
    '''
    Learns from limited labelled pairs by optimizing the entropic mapping to agree with known data pairings.
    Loss also uses the reversed transport map - implemented for symmetric h
    '''

    def __init__(self, init_key, config, d,
                 flow_type='shared',    # can be 'shared', 'separate', 'none'. NOTE: flows should be regularised suitably for the specific problem, to ensure the learned cost is meaningful
                 symmetric_h=True,
                 flow_depth=4,
                 num_steps=1000,
                 epsilon=0.01,
                 end_epsilon=None,
                 gamma = 1.0,
                 metric_gamma = 1e-2,
                 flow_gamma=1e-2,
                 max_gamma=1e-4,
                 alpha=0.01,
                 lr=1e-3,
                 flow_lr=1e-3,
                 cost_hidden_dims=[64,64,64],
                 cost_act_fn='celu',
                 inner_opt_solver='lbfgs',
                 inner_opt_tol=1e-5):

        self.config = config

        self.d = d
        self.flow_type = flow_type
        self.symmetric_h = symmetric_h
        self.flow_depth = flow_depth
        self.num_steps = num_steps
        self.epsilon = epsilon
        self.gamma = gamma
        self.metric_gamma = metric_gamma
        self.flow_gamma = flow_gamma
        self.max_gamma = max_gamma  
        self.alpha = alpha
        self.lr = lr
        self.flow_lr = flow_lr
        self.inner_opt_tol = inner_opt_tol

        if inner_opt_solver=='lbfgs':
            self.inner_opt_solver = jaxopt.LBFGS
        elif inner_opt_solver=='gd':
            self.inner_opt_solver = jaxopt.GradientDescent
        else:
            raise ValueError("Invalid inner optimizer: Implemented types are 'lbfgs', 'gd'")
        
        if symmetric_h==False:
            raise NotImplementedError("Asymmetric h not yet implemented")


        if end_epsilon is not None:
            self.decay_epsilon = True
            self.decay_epsilon_r = jnp.power(end_epsilon / epsilon, 1/self.num_steps)
        else:
            self.decay_epsilon = False

        activation_fns = {
            'celu': jax.nn.celu,
            'relu': jax.nn.relu,
            'sigmoid': jax.nn.sigmoid,
        }

        cost_init_key, flow_init_key = jax.random.split(init_key)

        self.h_model = icnn.ICNN(dim_hidden=cost_hidden_dims, dim_data=self.d, act_fn=activation_fns[cost_act_fn], pos_weights=True)

        cost_params = self.h_model.init(cost_init_key, jnp.zeros(self.d))

        init_fun = flows.Serial(*(flows.MADE(masked_transform), flows.Reverse()) * self.flow_depth)
        
        init_flow_params, self.direct_fun, self.inverse_fun = init_fun(flow_init_key, input_dim=self.d)

        if self.flow_type=='shared':
            flow_params = init_flow_params
        elif self.flow_type=='separate':
            flow_params = {'mu':init_flow_params, 'nu': init_flow_params}
        elif self.flow_type=='none':
            flow_params = 0.0
        else:
            raise ValueError("Invalid flow type: Implemented types are 'shared', 'separate', 'none'")

        self.params = (cost_params, flow_params)
        params_labels = ('cost_params', 'flow_params')

        self.optimizer = optax.multi_transform(
        {'cost_params': optax.adam(self.lr), 'flow_params': optax.adam(self.flow_lr)}, params_labels)


    def get_Phi(self, flow_params):
        def Phi(x):
            return self.direct_fun(flow_params, x)
        return Phi

    def get_Phi_inv(self, flow_params):
        def Phi_inv(x):
            return self.inverse_fun(flow_params, x)
        return Phi_inv
    
    def get_identity_flow(self):
        def identity_flow(x):
            return (x, jnp.zeros(x.shape[0]))
        return identity_flow
    
    def get_flows(self, flow_params):
        '''
        returns Phi_mu, Phi_mu_inv, Phi_nu, Phi_nu_inv
        '''
        if self.flow_type=='shared':
            return self.get_Phi(flow_params), self.get_Phi_inv(flow_params), self.get_Phi(flow_params), self.get_Phi_inv(flow_params)
        if self.flow_type=='separate':
            return self.get_Phi(flow_params['mu']), self.get_Phi_inv(flow_params['mu']), self.get_Phi(flow_params['nu']), self.get_Phi_inv(flow_params['nu'])
        if self.flow_type=='none':
            return self.get_identity_flow(), self.get_identity_flow(), self.get_identity_flow(), self.get_identity_flow()
        
    def get_h_from_params(self, h_params):
        def h_fn(x):
            cost = self.h_model.apply(h_params, x)
            return jnp.sum(cost) + (self.alpha / 2) * jnp.sum(x*x)
        
        def symmetric_h_fn(x):
            cost = 0.5*self.h_model.apply(h_params, x) + 0.5*self.h_model.apply(h_params, -x)
            return jnp.sum(cost) + (self.alpha / 2) * jnp.sum(x*x)
        
        if self.symmetric_h:
            return symmetric_h_fn
        else:
            return h_fn
    
    def flow_reg_loss(self, flow_params, x, y):
        '''
        Regularisation loss for the flows
        '''
        Phi_mu, _, Phi_nu, _ = self.get_flows(flow_params)

        _, log_det_x = Phi_mu(x)
        _, log_det_y = Phi_nu(y)

        reg_loss = jnp.sum(log_det_x*log_det_x) / x.shape[0] + jnp.sum(log_det_y*log_det_y) / y.shape[0]
        return reg_loss
    
    def metric_reg_loss(self, flow_params, x, y):
        '''
        Penalise the difference between flows (if using separate flows)
        '''
        Phi_mu, _, Phi_nu, _ = self.get_flows(flow_params)

        mu_x, _ = Phi_mu(x)
        mu_y, _ = Phi_mu(y)
        nu_x, _ = Phi_nu(x)
        nu_y, _ = Phi_nu(y)

        error_x = mu_x - nu_x
        error_y = mu_y - nu_y

        reg_loss = jnp.sum(error_x*error_x) / x.shape[0] + jnp.sum(error_y*error_y) / y.shape[0]

        return reg_loss

    def calc_reg_ot(self, x_dist, y_dist, init, epsilon=0.1):
        '''
        calculate the regularised OT cost using Sinkhorn, using a warmstart initialisation for the potentials
        '''
        geom = pointcloud.PointCloud(x_dist, y_dist, epsilon=epsilon)
        ot_prob = linear_problem.LinearProblem(geom)
        solver = sinkhorn.Sinkhorn(implicit_diff=None, recenter_potentials=True)
        ot = solver(ot_prob, init=init)

        next_inits = ot.f, ot.g
        return ot.reg_ot_cost, next_inits, ot

    def calc_sink_div(self, x_dist, y_dist, sinkhorn_inits, epsilon=0.1):
        '''
        Calculate the Sinkhorn divergence using a warmstart initialisation for the potentials
        '''
        xy_init = sinkhorn_inits['xy']
        xx_init = sinkhorn_inits['xx']
        yy_init = sinkhorn_inits['yy']

        xy_div, sinkhorn_inits['xy'], ot_xy = self.calc_reg_ot(x_dist, y_dist, init=xy_init, epsilon=epsilon)
        xx_div, sinkhorn_inits['xx'], ot_xx = self.calc_reg_ot(x_dist, x_dist, init=xx_init, epsilon=epsilon)
        yy_div, sinkhorn_inits['yy'], ot_yy = self.calc_reg_ot(y_dist, y_dist, init=yy_init, epsilon=epsilon)

        return xy_div - 0.5*(xx_div + yy_div), sinkhorn_inits

    def get_grad_h_inverse_fn(self, tol=1e-5):
        @jax.jit
        def grad_inverse_cost(outer_y, init_x, params):
            '''
            Calculates (\nabla h_theta)^{-1} for the current parameterised ICNN cost h
            '''
            @jax.jit
            def closure(x, inner_y, inner_params):
                cost_fn = self.get_h_from_params(h_params=inner_params)
                loss = cost_fn(x) - jnp.sum(x * inner_y)
                return loss
        
            opt_solver = self.inner_opt_solver(fun=closure, tol=tol)
            out, state = opt_solver.run(init_x, inner_y=outer_y, inner_params=params)
            return out, state
        return grad_inverse_cost

    def get_approx_transport_map(self, samples_x, samples_y, params, flow_params, sinkhorn_inits, epsilon):
        '''
        returns the generalised entropic mapping estimators for the current parameterised cost
        '''
        map_init = sinkhorn_inits['map']

        Phi_mu, Phi_mu_inv, Phi_nu, Phi_nu_inv = self.get_flows(flow_params)

        flow_samples_x, _ = Phi_mu(samples_x)
        flow_samples_y, _ = Phi_nu(samples_y)

        h_fn = self.get_h_from_params(params)
        cost_matrix = all_pairs_pairwise(h_fn, flow_samples_x, flow_samples_y)

        geom = geometry.Geometry(cost_matrix, epsilon=epsilon, relative_epsilon=True)
        ot_prob = linear_problem.LinearProblem(geom)
        # solver = sinkhorn.Sinkhorn(implicit_diff=None, recenter_potentials=True, threshold=1e-6)
        solver = sinkhorn.Sinkhorn(recenter_potentials=True, threshold=1e-6)
        ot = solver(ot_prob, init=map_init)

        sinkhorn_inits['map'] = ot.f, ot.g     # for initialisation of next iteration
        
        # get grad potential
        entropic_f = partial(potential_fn, 
                    potential_g=ot.g, 
                    y = flow_samples_y, 
                    weights_b = ot_prob.b, 
                    epsilon = geom.epsilon,
                    cost_fn=h_fn)
        
        grad_f = get_grad_f(entropic_f)

        grad_h_inverse = self.get_grad_h_inverse_fn(tol=self.inner_opt_tol)
        partial_grad_h_inv = partial(grad_h_inverse, params=params)
        
        def transport_map(x, inv_inits):
            x, _ = Phi_mu(x)
            g = grad_f(x)
            grad_inv, state = jax.vmap(partial_grad_h_inv)(g, init_x=inv_inits)
            transported_x = x - grad_inv
            transported_x, _ = Phi_nu_inv(transported_x)
            return transported_x, state, grad_inv
        

        # get grad potential
        entropic_g = partial(potential_fn, 
                    potential_g=ot.f, 
                    y = flow_samples_x, 
                    weights_b = ot_prob.a, 
                    epsilon = geom.epsilon,
                    cost_fn=h_fn)   # assuming h symmetric
        
        grad_g = get_grad_f(entropic_g)


        # TODO: generalise to asymmetric h
        def reverse_transport_map(y, inv_inits):
            y, _ = Phi_nu(y)
            g = grad_g(y)
            grad_inv, state = jax.vmap(partial_grad_h_inv)(g, init_x=inv_inits)
            transported_y = y - grad_inv
            transported_y, _ = Phi_mu_inv(transported_y)
            return transported_y, state, grad_inv
        
        return transport_map, reverse_transport_map, sinkhorn_inits, cost_matrix


    @partial(jax.jit, static_argnums=(0,))
    def loss_fn(self, total_params, mu_train, nu_train, aligned_idxs, inits, epsilon=0.1, gamma=1.0):
        '''
        Calculate L2 loss between target samples and entropic map predictions, with regularisation terms
        '''

        (params, flow_params) = total_params
        transport_map, reverse_transport_map, inits['sinkhorn'], cost_matrix = self.get_approx_transport_map(mu_train, nu_train, params=params, flow_params=flow_params, sinkhorn_inits=inits['sinkhorn'], epsilon=epsilon)

        cost_matrix_max_reg = logsumexp(jnp.abs(cost_matrix))

        (x_idxs, y_idxs) = aligned_idxs

        transported_x, state, new_inits = transport_map(mu_train[x_idxs], inits['inv'][x_idxs])
        inits['inv'] = inits['inv'].at[x_idxs].set(new_inits)

        transported_y, state, new_inits = reverse_transport_map(nu_train[y_idxs], inits['inv_reverse'][y_idxs])
        inits['inv_reverse'] = inits['inv_reverse'].at[y_idxs].set(new_inits)

        y_train = nu_train[y_idxs]
        error = y_train - transported_x
        forward_alignment_loss = jnp.sum(error*error) / error.shape[0]

        x_train = mu_train[x_idxs]
        error = x_train - transported_y
        reverse_alignment_loss = jnp.sum(error*error) / error.shape[0]

        alignment_loss = 0.5*forward_alignment_loss + 0.5*reverse_alignment_loss

        # fitting_loss, inits['sinkhorn'] = self.calc_sink_div(transported_mu, nu_train, sinkhorn_inits=inits['sinkhorn'], epsilon=epsilon)
        fitting_loss = 0.0  # not used (could be added to ensure a good fit to the data, but usually not required)

        metric_reg_loss = self.metric_reg_loss(flow_params, mu_train, nu_train)

        flow_reg_loss = self.flow_reg_loss(flow_params, mu_train, nu_train)

        loss = fitting_loss + gamma * alignment_loss + self.metric_gamma * metric_reg_loss + self.flow_gamma * flow_reg_loss + self.max_gamma * cost_matrix_max_reg

        return loss, (transported_x, fitting_loss, alignment_loss, metric_reg_loss, flow_reg_loss, cost_matrix_max_reg, inits)

    @partial(jax.jit, static_argnums=(0,))
    def step(self, params, mu_train, nu_train, aligned_idxs, inits, opt_state, epsilon=0.1, gamma=1.0):

        (loss, (transported_x, fitting_loss, alignment_loss, metric_reg_loss, flow_reg_loss, cost_matrix_max_reg, inits)), grads = jax.value_and_grad(self.loss_fn, has_aux=True)(params, mu_train, nu_train, aligned_idxs, inits, epsilon, gamma)

        updates, opt_state = self.optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        return loss, fitting_loss, alignment_loss, metric_reg_loss, flow_reg_loss, cost_matrix_max_reg, inits, params, opt_state
    
    
    def train(self, mu_train, nu_train, aligned_idxs=None):
        '''
        train according to labelled pairs.

        Args:
            mu_train: source samples
            nu_train: target samples
            aligned_idxs: indices of samples to align (if None, then assume we are in Inverse OT setting and all samples are paired in the given order)
        '''

        if aligned_idxs is None:
            assert mu_train.shape == nu_train.shape
            x_idxs = jnp.arange(mu_train.shape[0])
            y_idxs = jnp.arange(nu_train.shape[0])
            aligned_idxs = (x_idxs, y_idxs)

        loss_lst = []
        fitting_loss_lst = []   # not used
        alignment_loss_lst = []
        metric_reg_loss_lst = []
        flow_reg_loss_lst = []
        params_lst = []

        params = self.params
        params_lst.append(params)

        tbar = trange(self.num_steps, leave=True)

        opt_state = self.optimizer.init(self.params)
        self.opt_state = opt_state

        map_init = jnp.zeros(mu_train.shape[0]), jnp.zeros(nu_train.shape[0])
        xy_init = jnp.zeros(mu_train.shape[0]), jnp.zeros(nu_train.shape[0])
        xx_init = jnp.zeros(mu_train.shape[0]), jnp.zeros(mu_train.shape[0])
        yy_init = jnp.zeros(nu_train.shape[0]), jnp.zeros(nu_train.shape[0])
        sinkhorn_inits = {'map': map_init, 'xy': xy_init, 'xx': xx_init, 'yy': yy_init}

        inv_inits = jnp.zeros(mu_train.shape)
        inv_inits_reverse = jnp.zeros(nu_train.shape)

        inits = {'sinkhorn': sinkhorn_inits, 'inv': inv_inits, 'inv_reverse': inv_inits_reverse}

        params_step = self.num_steps / 50
        
        for step in tbar:

            if self.decay_epsilon==True:
                self.epsilon = self.epsilon*self.decay_epsilon_r

            loss, fitting_loss, alignment_loss, metric_reg_loss, flow_reg_loss, cost_matrix_max_reg, inits, params, opt_state = self.step(params, mu_train, nu_train, aligned_idxs, inits, opt_state, self.epsilon, self.gamma)

            self.params = params
            self.opt_state = opt_state

            loss_lst.append(loss)
            fitting_loss_lst.append(fitting_loss)
            alignment_loss_lst.append(alignment_loss)
            metric_reg_loss_lst.append(metric_reg_loss)
            flow_reg_loss_lst.append(flow_reg_loss)

            if step % params_step == 0:
                params_lst.append(params)

            postfix_str = (f'Loss: {loss}, Alignment Loss: {alignment_loss}, Fitting Loss: {fitting_loss}, Metric reg Loss: {metric_reg_loss} Flow reg Loss: {flow_reg_loss}, max_cost_matrix: {cost_matrix_max_reg}')
            tbar.set_postfix_str(postfix_str)
            
        return params_lst, loss_lst  

    def get_h_fn(self):
        h_params, _ = self.params
        h_fn = self.get_h_from_params(h_params)
        return h_fn
    
    def get_flow_cost(self):
        h_fn = self.get_h_fn()
        _, flow_params = self.params
        Phi_mu, _, Phi_nu, _ = self.get_flows(flow_params)

        def flow_cost_fn(x, y):
            flow_x, _ = Phi_mu(jnp.expand_dims(x, axis=0))
            flow_y, _ = Phi_nu(jnp.expand_dims(y, axis=0))
            return h_fn(flow_x - flow_y)

        return flow_cost_fn

    def get_transport_map(self, mu_train, nu_train):    
        # Solve sample OT problem
        h_params, flow_params = self.params
        h_fn = self.get_h_fn()

        Phi_mu, Phi_mu_inv, Phi_nu, Phi_nu_inv = self.get_flows(flow_params)
        flow_samples_x, _ = Phi_mu(mu_train)
        flow_samples_y, _ = Phi_nu(nu_train)

        cost_matrix = all_pairs_pairwise(h_fn, flow_samples_x, flow_samples_y)
        geom = geometry.Geometry(cost_matrix, epsilon=self.epsilon, relative_epsilon=True)
        ot_prob = linear_problem.LinearProblem(geom)
        solver = sinkhorn.Sinkhorn(threshold=1e-6)
        ot = solver(ot_prob)

        # get grad potential
        entropic_f = partial(potential_fn, 
                    potential_g=ot.g, 
                    y = flow_samples_y, 
                    weights_b = ot_prob.b, 
                    epsilon = geom.epsilon,
                    cost_fn=h_fn)
        grad_f = jax.grad(entropic_f)
        grad_h_inverse_fn = self.get_grad_h_inverse_fn(tol=self.inner_opt_tol)
        partial_grad_h_inv = partial(grad_h_inverse_fn, params=h_params)
        
        @jax.jit
        def transport_map(x):
            x, _ = Phi_mu(x)
            g = jax.vmap(grad_f)(x)
            inv, _ = jax.vmap(partial_grad_h_inv)(g, init_x=jnp.zeros_like(g))
            transported_x = x - inv
            transported_x, _ = Phi_nu_inv(transported_x)
            return transported_x
        
        return transport_map

    def save(self, save_dir):
        # save config as json
        json.dump(self.config, open(save_dir+'config.json', 'wb'))
        # save params
        pkl.dump(self.params, open(save_dir+'params', 'wb'))

    def load_params(self, params):
        self.params = params
