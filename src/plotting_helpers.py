# Define plotting functions

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from numpy import exp,arange
from pylab import meshgrid, cm,imshow,contour,clabel,colorbar,axis,title,show

def plot_cost_fn(cost_fn, x=arange(-3.0,3.0,0.1), y = arange(-3.0,3.0,0.1),argmin=None, save_dir=None):
    '''
    Creates a 3d plot of the cost function
    '''
    def func(x,y):
        z = jnp.array([x,y])
        return cost_fn(z)

    X,Y = meshgrid(x, y) # grid of point
    Z = jnp.vectorize(func)(X,Y) # evaluation of the function on the grid

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,cmap='jet', edgecolor = 'none')

    if argmin is not None:
        min_val = cost_fn(argmin)
        ax.scatter(argmin[0], argmin[1], min_val, c='red', marker='o', s=100)

    ax.view_init(azim=180)  # Set both elev and azim to 180

    if save_dir is not None:
        plt.savefig(save_dir)

def plot_cost_fn_planar(cost_fn, x=arange(-3.0,3.0,0.1), y=arange(-3.0,3.0,0.1), displacements=None, save_dir=None):
    '''
    Creates a planar heatmap plot of the cost function
    '''
    def func(x,y):
        z = jnp.array([x,y])
        return cost_fn(z)
    X,Y = meshgrid(x, y) # grid of point
    Z = jnp.vectorize(func)(X,Y) # evaluation of the function on the grid
    fig = plt.figure(figsize=(8,6))
    plt.pcolormesh(X, Y, Z, cmap='jet', shading='auto')
    plt.colorbar(label='Cost Function Value')

    if displacements is not None:
        plt.scatter(displacements[:,0], displacements[:,1], label='discrete displacements');

    if save_dir is not None:
        plt.savefig(save_dir)


def plot_points(x_samples, y_samples, transported_samples=None, save_dir = None, show=False, grid=False, RMSE=None):
    '''
    Plots the source, target and transported points
    '''
    plt.scatter(x_samples[:,0],x_samples[:,1], label='$source x$');
    plt.scatter(y_samples[:,0], y_samples[:,1], label='$target y$');
    plt.plot([x_samples[:,0],y_samples[:,0]], [x_samples[:,1],y_samples[:,1]], '-k', lw=0.5);

    if transported_samples is not None:
        plt.scatter(transported_samples[:,0],transported_samples[:,1], label='$T_\epsilon^h(x)$');
        plt.plot([x_samples[:,0],transported_samples[:,0]], [x_samples[:,1],transported_samples[:,1]], '-r', lw=0.5);
    plt.legend()

    if RMSE is not None:
        plt.title(f'RMSE {RMSE}')

    if grid:
        plt.grid(True)

    if save_dir is not None:
        plt.savefig(save_dir)

    if show:
        plt.show()

    plt.close()


def plot_limited_pairs(train_mu, train_nu, x_idxs, y_idxs, transported_x = None, save_dir = None, show=False, RMSE=None):
    '''
    Plots the source, target and transported points, and the arrows between the indexed pairs
    '''
    plt.scatter(train_mu[:,0],train_mu[:,1], label='source $x$', marker='x');
    plt.scatter(train_nu[:,0], train_nu[:,1], label='target $y$', marker='x');
    # Create arrows from init_samples to final_samples
    for i, j in zip(x_idxs, y_idxs):
        plt.annotate("", xy=(train_nu[j, 0], train_nu[j, 1]), xytext=(train_mu[i, 0], train_mu[i, 1]),
                arrowprops=dict(arrowstyle="->", color='k'))
        
    if transported_x is not None:
        plt.scatter(transported_x[:,0],transported_x[:,1], label='OT map estimator $\hat{T}(x)$', marker='x');
        for i in x_idxs:
            plt.annotate("", xy=(transported_x[i, 0], transported_x[i, 1]), xytext=(train_mu[i, 0], train_mu[i, 1]),
                    arrowprops=dict(arrowstyle="->", color='r'))
            
    plt.legend()

    if RMSE is not None:
        plt.title(f'RMSE {RMSE}')
        
    if save_dir is not None:
        plt.savefig(save_dir)

    if show:
        plt.show()

    plt.close()