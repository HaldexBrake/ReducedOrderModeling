import numpy as np
import matplotlib.pyplot as plt
from assimulo.solvers import CVode
from assimulo.problem import Explicit_Problem
import sys
sys.path.append('../')
from dmd import dmd
from sympy import symbols, lambdify
from numpy.linalg import solve, norm, inv
from scipy.linalg import eig


def simulate_prob(t_start, t_stop, x0, p, ncp, with_plots=False):
    """Simulates the problem using Assimulo.

    Args:
        t_start (double): Simulation start time.
        t_stop (double): Simulation stop time.
        x0 (list): Initial value.
        p (list): Problem specific parameters.
        ncp (int): Number of communication points.
        with_plots (bool): Plots the solution.

    Returns:
        tuple: (t,y). Time vector and solution at each time.

    """
    # Assimulo
    # Define the right-hand side
    def f(t, y):
        xd_1  = p[0]*y[0]
        xd_2  = p[1]*(y[1]-y[0]**2)
        return np.array([xd_1,xd_2])

    # Define an Assimulo problem
    exp_mod = Explicit_Problem(f, y0=x0, name='Planar ODE')

    # Define an explicit solver
    exp_sim = CVode(exp_mod)

    # Sets the solver parameters
    exp_sim.atol = 1e-12
    exp_sim.rtol = 1e-11

    # Simulate
    t, y = exp_sim.simulate(tfinal=t_stop, ncp=ncp)

    # Plot
    if with_plots:
        x1 = y[:,0]
        x2 = y[:,1]
        plt.figure()
        plt.title('Planar ODE')
        plt.plot(t,x1,'b')
        plt.plot(t,x2,'k')
        plt.legend(['x1', 'x2'])
        plt.xlim(t_start,t_stop)
        plt.xlabel('Time (s)')
        plt.ylabel('x')
        plt.grid(True)

    return t,y.T


def koopman_prob(data, observables, ncp):
    """Creates a snapshot matrix from the `observables`.

    Args:
        data (ndarray): Simulation result.
        observables (list of SymPy expressions): The observable functions used
            to extract states.
        ncp (int): Number of communication points.

    Returns:
        ndarray: New snapshot matrix.

    """
    def _wrapper(func, args):
        """
        Wrapper function to be able to call the function `func` with
        arguments inside a list.
        """
        return func(*args)

    # Structure data as we want to
    data = {'x1': data[0,:], 'x2': data[1,:]}

    # Extract interesting simulation results
    snapshots = np.zeros((len(observables),ncp+1))
    for i, obs in enumerate(observables):
        syms = obs.free_symbols                             # Get args in this observable
        states = [sym.name for  sym in list(syms)]          # Get the names of the args, i.e. our states
        f = lambdify(syms, obs, 'numpy')                    # Vectorize the observable function
        values = [data[state] for state in states]          # Get simulation result for each state
        val = _wrapper(f, values)                           # Computes g_i(x)
        snapshots[i,:] = val

    return snapshots


def analytical_prob(t_start, t_stop, x0, ncp, p):
    """Computes the analytical solution of the planar ODE.

    Args:
        t_start (double): Simulation start time.
        t_stop (double): Simulation stop time.
        x0 (list): Initial value.
        ncp (int): Number of communication points.
        p (list): Problem specific parameters.

    Returns:
        ndarray: The solution at each time step.

    """
    x0 = np.concatenate((x0, np.array([x0[0]**2])))
    n = len(x0)
    mu,lamb = p[0],p[1]

    # System matrix
    A = np.array([[mu,0,0],[0,lamb,-lamb],[0,0,2*mu]])

    # Eigendecomposition of `A`
    # Eigenvalues as elements of `lam_A`, eigenvecs as columns in `V`
    lam_A,V = eig(A)

    # Exponential matrix of A for the given time step
    dt = (t_stop-t_start)/ncp       # Step size
    expAdt = V@np.diag(np.exp(dt*lam_A))@inv(V)
    expAdt = np.real(expAdt)

    # Setup for time-stepping
    X = np.zeros((n,ncp+1))                 # Construct matrix for storage
    X[:,0] = x0                             # Set initial values
    B = solve(A, expAdt-np.eye(n))          # Help matrix for more efficient calculations
    b = B[:,-1]                             # Extract the needed column

    # Iterate solution forward in time
    for k in range(1,ncp+1):
        X[:,k] = expAdt@X[:,k-1]

    return X[:2,:]


if __name__=='__main__':
    # Parameters
    t_start = 0.0               # Start time
    t_stop = 12.0               # Stop time
    t_stop_train = 6.0          # Only train with data up to this time
    ncp = 1200                  # Number of communication points
    x0 = [1e-4, 1e-4**2]        # Initial value
    q = 0                       # Time-delay embeddings
    koopman = False             # If set to false then DMD is performed

    # Problem specific parameters
    mu = 0.3
    lamb = 1.1
    p = [mu,lamb]

    # Simulated data using Assimulo
    t,data_sim = simulate_prob(t_start, t_stop, x0, p, ncp, with_plots=False)
    t = np.array(t)

    # Analytic data
    data_an = analytical_prob(t_start, t_stop, x0, ncp, p)

    # Sets method and data depending on `koopman`
    if koopman:
        # Define observable functions and get new data
        method = 'Koopman'
        x1, x2 = symbols('x1 x2')
        g = [x1, x2, x1**2]
        data = koopman_prob(data_an, g, ncp)
    else:
        method = 'DMD'
        data = data_an

    # Calculate stop index for training
    if not t_stop_train:
        m_stop = ncp+1
    elif t_stop_train>t_stop:
        raise ValueError('t_stop_train must be <= t_stop.')
    else:
        m_stop = np.argmin(t<t_stop_train)

    # Construct X,Y from data
    X,Y = dmd.get_data_matrices(data, m_stop=m_stop, u=None, q=q)

    # Calculate DMD modes and eigenvalues
    lam,w,v,_ = dmd.get_dmd_modes(X,Y)

    # Predict the system
    Yhat = dmd.predict(lam, w, v, X[:,0], ncp, u=None, q=q)

    # Extract results
    x1_dmd = Yhat[0,:]
    x2_dmd = Yhat[1,:]
    x1_sim = data_sim[0,:]
    x2_sim = data_sim[1,:]
    x1_an  = data_an[0,:]
    x2_an  = data_an[1,:]

    # Step size
    dt = (t_stop-t_start)/ncp

    # Print errors
    print('Error ({} vs simulation)'.format(method))
    print('   x1: ', norm(x1_dmd-x1_sim,2))
    print('   x2: ', norm(x2_dmd-x2_sim,2))
    print('Error ({} vs analytical)'.format(method))
    print('   x1: ', norm(x1_dmd-x1_an,2))
    print('   x2: ', norm(x2_dmd-x2_an,2))
    print('Error (simulation vs analytical)')
    print('   x1: ', norm(x1_sim-x1_an,2))
    print('   x2: ', norm(x2_sim-x2_an,2))

    # Plot
    plt.figure()
    plt.plot(t,x1_an,'b')
    plt.plot(t,x1_dmd,'--r')
    plt.plot(t,x2_an,'k')
    plt.plot(t,x2_dmd,'--g')
    plt.legend(('Analytical $x_1$','{} $x_1$'.format(method),'Analytical $x_2$','{} $x_2$'.format(method)),
               bbox_to_anchor=(0,1.02,1,0.2), loc='lower left',mode='expand', borderaxespad=0, ncol=2)
    plt.xlim(0, dt*(ncp))
    plt.ylim(-0.006960648530924138,0.004165560206256513)
    plt.xlabel('Time [s]')
    plt.ylabel('x')
    plt.grid(True)
    if m_stop < ncp:
        plt.axvline(x=t_stop_train, color='k', linestyle='-',linewidth=1)
    plt.show()
