import sys
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
sys.path.append('../')
from dmd import dmd


if __name__ == '__main__':
    # Parameters
    t_start = 0.0               # start simulation time
    t_stop = 20.0               # stop simulations time
    t_stop_train = 10.0         # only train with data up to this time
    ncp = 501                   # number of communication points
    amp =  10.0                 # amplitude of input signal
    freq = 1.0                  # frequency of input signal
    inp = 'sin'                 # input type
    analytical_data = False     # use data generated through simulation or analytically
    q = 0                       # time-delay embeddings

    # Construct time vector
    t = np.linspace(t_start,t_stop, ncp+1)

    # Calculate stop index for training
    if not t_stop_train:
        m_stop = ncp+1
    elif t_stop_train>t_stop:
        raise ValueError('t_stop_train must be <= t_stop.')
    else:
        m_stop = np.sum(t<t_stop_train)

    # Get input vector
    u = dmd.create_input_vec(t, inp_type=inp, amp=amp, freq=freq)

    # Get snapshots, both simulated an analytical
    data_sim = dmd.get_snapshots_damped_dual_mass(t_start, t_stop, ncp, input_force=u, time_vec=t,
                                                  states=['mass1.s','mass1.v','mass2.s','mass2.v'])
    data_an = dmd.get_analytical_snapshots_damped_dual_mass(t_start, t_stop, ncp, input_force=u)

    # Get data matrices
    if analytical_data:
        X,Y = dmd.get_data_matrices(data_an, m_stop=m_stop, u=u, q=q)
    else:
        X,Y = dmd.get_data_matrices(data_sim, m_stop=m_stop, u=u, q=q)

    # Calculate DMD modes and eigenvalues
    lam,w,v,Ahat = dmd.get_dmd_modes(X,Y)

    # Predict the system
    Yhat = dmd.predict(lam, w, v, X[:,0], ncp, u=u, q=q)

    # Plots
    # Extract results
    x1_dmd = Yhat[0,:]
    x2_dmd = Yhat[2,:]
    x1_sim = data_sim[0,:]
    x2_sim = data_sim[2,:]
    x1_an = data_an[0,:]
    x2_an = data_an[2,:]

    # Step size
    dt = (t_stop-t_start)/ncp

    # Position of mass 2 (left mass)
    if analytical_data:
        data = 'Analytical'
    else:
        data ='Simulation'

    plt.figure()
    if analytical_data:
        plt.plot(t,x2_an,'-b')
    else:
        plt.plot(t,x2_sim,'-b')
    plt.plot(t,x2_dmd,'--r')
    plt.legend((data,'DMD'),bbox_to_anchor=(0,1.02,1,0.2), loc='lower left',
               mode='expand', borderaxespad=0, ncol=2)
    plt.ylabel('Position of mass 2 [m]')
    plt.xlabel('Time [s]')
    plt.xlim(0, dt*(ncp))
    plt.ylim(-0.65,0.65)
    plt.grid(True)
    if m_stop < ncp:
        plt.axvline(x=t_stop_train, color='k', linestyle='-',linewidth=1)
    plt.show()

    # Print errors
    print('Error (dmd vs simulation)')
    print('   x1: ', norm(x1_dmd-x1_sim,2))
    print('   x2: ', norm(x2_dmd-x2_sim,2))
    print('Error (dmd vs analytical)')
    print('   x1: ', norm(x1_dmd-x1_an,2))
    print('   x2: ', norm(x2_dmd-x2_an,2))
    print('Error (simulation vs analytical)')
    print('   x1: ', norm(x1_sim-x1_an,2))
    print('   x2: ', norm(x2_sim-x2_an,2))
