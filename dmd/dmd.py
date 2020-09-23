"""

"""
from pyfmi import load_fmu
import numpy as np
from scipy.linalg import eig
from numpy.linalg import svd, solve, inv, norm
import matplotlib.pyplot as plt
from sympy import symbols, lambdify


def create_input_vec(time_vec, inp_type='sin', amp=10.0, freq=1.0, delta_time=1.0, duration=1):
    """Constructs an input vector either as a sine wave, Dirac pulse or chirp signal.

    Args:
        time_vec (ndarray): Time vector.
        inp_type (str): What kind input is wanted. Must be `sin`, `delta` or `inc_freq`.
        amp (double): Amplitude of the sine wave or the Dirac pulse.
        freq (double): Frequency of the sine wave.
        delta_time (double): Time at which the pulse starts.
        duration (int): Duration in time steps of the Dirac pulse.

    Returns:
        ndarray: The input vector.

    Raises:
        ValueError: If specified input type is non-existing.

    """
    if amp == 0:
        u = None
    elif inp_type == 'sin':
        u = amp*np.sin((freq*2*np.pi)*time_vec)
    elif inp_type == 'delta':
        u = np.zeros_like(time_vec)
        idx = np.argmax(time_vec>delta_time)
        u[idx:idx+duration] = np.array(duration*[amp])
    elif inp_type == 'inc_freq':
        freq = np.linspace(0,1,len(time_vec))
        u = amp*np.sin((freq*2*np.pi)*time_vec)
    else:
        raise ValueError('inp must be either \'sin\', \'inc_freq\' or \'delta\'.')
    return u


def get_snapshots_damped_dual_mass(t_start, t_stop, ncp, input_force=None, time_vec=None, states=['mass1.s','mass1.v','mass2.s','mass2.v']):
    """Simulates the FMU of the damped dual mass system and returns the
    snapshots as a matrix.

    Args:
        t_start (double): Simulation start time.
        t_stop (double): Simulation stop time.
        ncp (int): Number of communication points, i.e. number of time steps
            excluding the initial conndition.
        input_force (ndarray): Input signal of same length as `time_vec`.
        time_vec (ndarray): Time vector of same length as `input_force`.
        states (list): The states of the fmu that should be included in the
            snapshots.

    Returns:
        ndarray: The matrix of snapshots, where row i corresponds to state i
            in states and time evovles along the columns.

    Raises:
        ValueError: If `input_force` is given without time vector.

    """
    # Load the FMU
    model = load_fmu('../fmu/DampedDualMassSystem.fmu')

    # Specify number of comuncation points (ncp)
    opts = model.simulate_options()
    opts['ncp'] = ncp

    # Create input object
    if input_force is not None:
        if time_vec is None:
            raise ValueError('Specify time vector plz.')
        input_object = ('F', np.transpose(np.vstack((time_vec,input_force))))
    else:
        input_object = None

    # Simulate the FMU
    res = model.simulate(start_time=t_start, final_time=t_stop, input=input_object, options=opts)

    # If no states are given, return all
    if states is None or len(states) == 0:
        states = res.keys()

    # Extract simulation result
    snapshots = np.zeros((len(states),ncp+1))
    for i, state in enumerate(states):
        snapshots[i,:] = res[state]

    return snapshots


def get_snapshots_stop_friction(t_start, t_stop, ncp, input_force=None, time_vec=None, states=['mass1.s','mass1.v','mass2.s','mass2.v']):
    """Simulates the FMU of the damped dual mass system with stop and friction
    and returns the snapshots as a matrix.

    Args:
        t_start (double): Simulation start time.
        t_stop (double): Simulation stop time.
        ncp (int): Number of communication points, i.e. number of time steps
            excluding the initial conndition.
        input_force (ndarray): Input signal of same length as `time_vec`.
        time_vec (ndarray): Time vector of same length as `input_force`.
        states (list): The states of the FMU that should be included in the
            snapshots.

    Returns:
        ndarray: The matrix of snapshots, where row i corresponds to state i
            in states and time evovles along the columns.

    Raises:
        ValueError: If `input_force` is given without time vector.

    """
    # Load the FMU
    model = load_fmu('../fmu/DampedDualMassSystemStopFriction.fmu')

    # Specify number of comuncation points (ncp)
    opts = model.simulate_options()
    opts['ncp'] = ncp

    # Create input object
    if input_force is not None:
        if time_vec is None:
            raise ValueError('Specify time vector plz.')
        input_object = ('F', np.transpose(np.vstack((time_vec,input_force))))
    else:
        input_object = None

    # Simulate the FMU
    res = model.simulate(start_time=t_start, final_time=t_stop, input=input_object, options=opts)

    # Find mask to extract the solution at the specified communication points, i.e. not at
    # the additional state events points.
    mask = len(res['time'])*[True]
    i = 0
    for el in time_vec:
        while abs(el-res['time'][i])>1e-12:
            mask[i] = False
            i += 1
            if i == len(res['time']):
                break
        i += 1

    # If no states are given, return all
    if states is None or len(states)==0:
        states = res.keys()

    # Extract simulation result
    snapshots = np.zeros((len(states),ncp+1))
    for i, state in enumerate(states):
        snapshots[i,:] = res[state][mask]

    return snapshots


def get_koopman_snapshots_stop_friction(t_start, t_stop, ncp, observables, input_force=None, time_vec=None):
    """Simulates the FMU of the damped dual mass system with stop and friction
    and returns the snapshots as a matrix where observables have been applied
    to the result.

    Args:
        t_start (double): Simulation start time.
        t_stop (double): Simulation stop time.
        ncp (int): Number of communication points, i.e. number of time steps
            excluding the initial conndition.
        observables (list of SymPy expressions): The observable functions used
            to extract states.
        input_force (ndarray): Input signal of same length as `time_vec`.
        time_vec (ndarray): Time vector of same length as `input_force`.

    Returns:
        ndarray: The matrix of snapshots, where row i corresponds to state i
            in states and time evovles along the columns.

    Raises:
        ValueError: If `input_force` is given without time vector.

    """
    # Wrapper function to be able to call the function `func` with
    # arguments inside a list
    def _wrapper(func, args):
        return func(*args)

    # Load the FMU
    model = load_fmu('../fmu/DampedDualMassSystemStopFriction.fmu')

    # Specify number of comuncation points (ncp)
    opts = model.simulate_options()
    opts['ncp'] = ncp

    # Create input object
    if input_force is not None:
        if time_vec is None:
            raise ValueError('Please specify time vector.')
        input_object = ('F', np.transpose(np.vstack((time_vec,input_force))))
    else:
        input_object = None

    # Simulate the FMU
    res = model.simulate(start_time=t_start, final_time=t_stop, input=input_object, options=opts)

    # Find mask for extracting the solution at the correct points, i.e. not at
    # the additional state events
    mask = len(res['time'])*[True]
    i = 0
    for t in time_vec:
        while abs(t-res['time'][i])>1e-12:
            mask[i] = False
            i += 1
            if i == len(res['time']):
                break
        i += 1

    # Extract simulation result
    snapshots = np.zeros((len(observables),ncp+1))
    for i, obs in enumerate(observables):
        syms = obs.free_symbols                             # Get args in this observable
        states = [sym.name for  sym in list(syms)]          # Get the names of the args, i.e. our states
        f = lambdify(syms, obs, 'numpy')                    # Vectorize the observable function
        values = [res[state][mask] for state in states]     # Get simulation result for each state
        val = _wrapper(f, values)                           # Computes g_i(x)
        snapshots[i,:] = val

    return snapshots


def get_analytical_snapshots_damped_dual_mass(t_start, t_stop, ncp, input_force=None):
    """Generates snapshots for the damped dual mass system from the analytical
    solution to the system.

    Args:
        t_start (double): Simulation start time.
        t_stop (double): Simulation stop time.
        ncp (int): Number of communication points, i.e. number of time steps
            excluding the initial conndition.
        input_force (ndarray): Input signal of same length as `time_vec`.

    Returns:
        ndarray: The matrix of snapshots, where row i corresponds to state i
            in states and time evovles along the columns. The states are:
            position and veclocity of mass 1, position and veclocity of mass 2

    """
    # Analytical solution to the damped dual mass system
    # Set the values of constants
    k1,k2,m1,m2,c_damp = 250, 1000, 3, 2, np.sqrt(500)

    # Construct system matrix s.t. dot{x} = Ax
    A = np.array([[0,1,0,0],
                  [-(k1+k2)/m1,-c_damp/m1,k2/m1,c_damp/m1],
                  [0,0,0,1],
                  [k2/m2,c_damp/m2,-k2/m2,-c_damp/m2]])

    # Eigendecomposition of A
    lam_A,V = eig(A)      # eigenvals as elements of lam, eigenvecs as columns in V

    # Exponential matrix of a for the given time step
    dt = (t_stop-t_start)/ncp       # step size
    expAdt = V@np.diag(np.exp(dt*lam_A))@inv(V)
    expAdt = np.real(expAdt)

    # Setup for time-stepping
    X = np.zeros((4,ncp+1))                         # Construct matrix for storage
    X[:,0] = np.array([0, 0, 0.1, -0.2])            # Set initial values
    B = solve(A, expAdt-np.eye(4))                  # Help matrix for more efficient calculations
    b = B[:,-1]                                     # Extract the needed column

    # Iterate solution forward in time
    # States at time k (given states up to k-1) are given by
    # X[:,k] = expAdt@X[:,k-1] + A_inv@(expAdt - np.eye(4))@np.array([0,input_force[k-1]/m2,0,0])
    if input_force is None:
        for k in range(1,ncp+1):
            X[:,k] = expAdt@X[:,k-1]
    else:
        for k in range(1,ncp+1):
            X[:,k] = expAdt@X[:,k-1] + input_force[k-1]/m2*b

    return X


def get_data_matrices(data, m_stop=None, u=None, q=0):
    """Creates the X and Y data matrices.

    Args:
        data (ndarray): Simulation result.
        m_stop (int): Number of columns of the data matrices.
        u (ndarray): Input vector.
        q (int): Number of time-delay embeddings.

    Returns:
        tuple: (X,Y). The X and Y data matrices.

    Raises:
        ValueError: If `m_stop` is not valid.

    """
    if m_stop is None:
        m_stop = data.shape[1]
    elif m_stop == 0:
        raise ValueError('m_stop must be greater than zero')
    elif m_stop <= q+1:
        raise ValueError('m_stop at least  = q+2')

    # Construct data matrices (If statement not required. More effective to create X directly and not stack vectors)
    if q>0:
        if u is None:
            X = data[:,:m_stop-(q+1)]
            for i in range(q,0,-1):
                X = np.vstack([X,data[:,(q+1-i):m_stop-i]])
            Y = np.vstack([X[data.shape[0]:,:],data[:,(q+1):m_stop]])
        else:
            zero_vec = np.zeros(m_stop-(q+1))
            X = np.vstack([data[:,:m_stop-(q+1)],u[:m_stop-(q+1)]])
            Y = np.vstack([data[:,1:m_stop-q],zero_vec])
            for i in range(q,0,-1):
                X = np.vstack([X,data[:,(q+1-i):m_stop-i],u[(q+1-i):m_stop-i]])
                Y = np.vstack([Y,data[:,(q+1-(i-1)):m_stop-(i-1)],zero_vec])
    else:
        if u is None:
            X = data[:,:m_stop-1]
            Y = data[:,1:m_stop]
        else:
            X = np.vstack([data[:,:m_stop-1],u[:m_stop-1]])
            Y = np.vstack([data[:,1:m_stop],np.zeros((m_stop-1,))])

    return X.astype(np.float64),Y.astype(np.float64)


def get_dmd_modes(X, Y, n_trunc=None, plot=False):
    """Computes the DMD modes `v` and eigenvalues `lam` of the data matrices `X`,`Y`.

    Args:
        X (ndarray): First data matrix.
        Y (ndarray): Second data matrix.
        n_trunc (int): Truncates `X` to rank `n_trunc`.
        plot (bool): Plots the singular values, eigenvalues and some columns of V.

    Returns:
        tuple: (lam,w,v,A). Eigenvalues, left eigenvectors, right eigenvectors and the matrix A or A_tilde.

    """
    U,S,VH = svd(X,full_matrices=False)

    if n_trunc is None:
        # Compute A_hat and make eigendecomposition
        A = Y@VH.T@np.diag(1/S)@U.T
        # Add some dust to the diagonal (singular values) if needed
        # A = Y@VH.T@np.diag(S/(S*S + 1e-2))@U.T
        lam,w,v = eig(A,left=True,right=True)
    else:
        # Truncate
        U = U[:,0:n_trunc]
        S_trunc = S[0:n_trunc]
        VH = VH[0:n_trunc,:]

        # Similarity transform to matrix A_tilde and eigendecomposition
        A = np.conj(U.T)@Y@np.conj(VH.T)@np.diag(1/S_trunc)
        lam,w_tilde,v_tilde = eig(A,left=True,right=True)

        # Project eigenvectors so we get correct DMD modes
        w = (w_tilde.T@np.conj(U.T)).T
        v = U@v_tilde

    if plot:
        # Singular values
        plt.figure()
        nnz_s = S > 1e-12
        x = np.arange(len(S))
        plt.semilogy(x[nnz_s],S[nnz_s], 'bx')
        plt.semilogy(x[nnz_s==False],S[nnz_s==False], 'rx')
        plt.xlim([1,len(S)])
        if n_trunc is not None:
            plt.axvline(x=n_trunc, color='k', linestyle='-',linewidth=1)
        plt.grid(True)
        plt.xlabel('Index')
        plt.ylabel('Singular value')

        # Eigenvalues in complex plane
        plt.figure()
        mask = np.abs(lam) > 1
        plt.plot(np.real(lam[mask==False]),np.imag(lam[mask==False]),'bx')
        plt.plot(np.real(lam[mask]),np.imag(lam[mask]),'rx')
        plt.plot(np.cos(np.linspace(0,2*np.pi)),np.sin(np.linspace(0,2*np.pi)),'--k')
        plt.xlabel('Re($\lambda$)')
        plt.ylabel('Im($\lambda$)')
        plt.grid(True)
        plt.axis('equal')

        # Columns in V
        plt.figure()
        plt.plot(np.conj(VH.T)[:,0:6])
        plt.xlim([0,VH.shape[1]])
        plt.title('Columns in V')
        plt.grid(True)

        plt.plot()

    return lam.astype(np.complex128),w.astype(np.complex128),v.astype(np.complex128),A


def one_step_pred(xk, lam, wH, v, norm_vec):
    """Performs a one-step prediction of the system represented by its DMD.

    Args:
        xk (ndarray): Solution (states) at time step k.
        lam (ndarray): DMD eigenvalues.
        wH (ndarray): Complex conjugated left eigenvectors from DMD.
        v (ndarray): DMD modes.
        norm_vec (ndarray): Normalization vector.

    Returns:
        ndarray: Prediction of the states of the system at time step k+1.

    """
    return np.real(np.sum(((lam*(wH@xk))*norm_vec)*v,axis=1))


def predict(lam, w, v, X0, N, u=None, q=0):
    """Predict the future dynamics of the system given an initial value `X0`. Result is returned
    as a matrix where rows correspond to states and columns to time.

    Args:
        lam (ndarray): DMD eigenvalues.
        w (ndarray): Left eigenvectors from DMD.
        v (ndarray): DMD modes.
        X0 (ndarray): Initial value of the system.
        N (int): Number of time steps to predict.
        u (ndarray): Input signal.
        q (int): Number of time-delay embeddings.

    Returns:
        ndarray: Prediction of the states of the system for N time steps into the future.

    """
    # Construct matrix for predictions and set initial values
    n = X0.shape[0]
    Yhat = np.zeros((n,N+1-q),dtype=np.float64)
    Yhat[:,0] = X0

    # Add input in the correct rows and construct a mask for prediction
    n_x = n//(q+1)
    if u is not None:
        if q>0:
            mask = np.array((q+1)*((n_x-1)*[True]+[False]))
            len_u = len(u)
            Yhat[mask==False,:] = np.vstack([u[i:len_u-(q-i)] for i in range(0,q+1)])
        else:
            mask = (n-1)*[True] + [False]
            Yhat[-1,:] = u                    # Add input
    else:
        mask = n*[True]

    # For efficient calculations
    wH = np.conj(w).T
    norm_vec = 1/(np.diag(wH@v))

    # Prediction
    for i in range(1,N+1-q):
        yhat = one_step_pred(Yhat[:,i-1],lam,wH,v,norm_vec)
        Yhat[mask,i] = yhat[mask]

    # Extract predictions
    res = np.zeros((n_x,N+1),dtype=np.float64)
    res[:,:N+1-q] = Yhat[:n_x,:]
    for i in range(q):
        res[:,N+1-q+i] = Yhat[(i+1)*n_x:(i+2)*n_x,-1]

    return res
