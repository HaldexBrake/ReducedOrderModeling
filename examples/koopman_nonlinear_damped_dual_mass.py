# TensorFlow
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU') # Force CPU
tf.keras.backend.set_floatx('float64')
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam

import sys
sys.path.append('../')
import numpy as np
from numpy.linalg import norm
np.set_printoptions(precision=5,linewidth=1000)
import matplotlib.pyplot as plt
from dmd import dmd
from koopman import koopman
from koopman.koopman import Net


if __name__ == '__main__':
    # Simulation params
    t_start = 0.0               # start simulation time
    t_stop = 40.0               # stop simulations time
    t_stop_train = 20.0         # only train with data up to this time
    ncp = 4000                  # number of communication points
    amp = 30.0                  # amplitude of input signal
    freq = 0.1                  # frequency of input signal
    inp = 'sin'                 # input type: delta or sin

    # NN and training params
    lr = 8e-5
    epochs = 400
    pre_lr = 5e-3
    pre_epochs = 300
    n = 21
    w_hidden = 60

    ############################################################################
    # Simulate and generate data
    ############################################################################
    # Construct time vector
    t = np.linspace(t_start,t_stop, ncp+1)
    dt = (t_stop-t_start)/ncp   # Step size

    # Calc stop idx for training
    if not t_stop_train:
        m_stop = ncp+1
    elif t_stop_train>t_stop:
        raise ValueError('t_stop_train must be <= t_stop.')
    else:
        m_stop = np.argmin(t<t_stop_train)

    # Compute input from time vector t
    u = dmd.create_input_vec(t, inp_type=inp, amp=amp, freq=freq)

    # Get snapshots, both simulated and analytical
    data_sim = dmd.get_snapshots_stop_friction(
        t_start, t_stop, ncp , input_force=u, time_vec=t, states=['mass1.s', 'mass1.v', 'mass2.s', 'mass2.v'])
    x1_target = data_sim[0,:]
    x2_target = data_sim[2,:]

    # Get data matrices
    X,Y = dmd.get_data_matrices(data_sim, m_stop=m_stop, u=u)

    ############################################################################
    # Construct and train NN
    ############################################################################
    # Convert arrays to tensors
    X_tensor = tf.constant(X.T, dtype=tf.float64)
    Y_tensor = tf.constant(Y.T, dtype=tf.float64)
    if u is not None:
        u_tensor = tf.constant(u, dtype=tf.float64)
    else:
        u_tensor = None

    # NN and training params
    m_stop, n_x = X_tensor.shape

    # Define networks
    g = Net(n_x, n, w_hidden=w_hidden, activation='relu')
    h = Net(n, n_x, w_hidden=w_hidden, activation='relu')

    # Initiate weights
    g(tf.expand_dims(tf.ones(n_x, dtype=tf.dtypes.float64),0))
    h(tf.expand_dims(tf.ones(n, dtype=tf.dtypes.float64),0))

    # Create optimizer objects
    pre_opt_g = Adam(learning_rate=pre_lr)
    pre_opt_h = Adam(learning_rate=pre_lr)

    # Pre-train network
    print('Pre-training the network as an autoencoder...')
    pre_loss_history = koopman.pre_train_networks(pre_opt_g, pre_opt_h, g,
                                                  h, Y_tensor, EPOCHS=tf.constant(pre_epochs))
    print('FINISHED with final pre-train loss: ',pre_loss_history[-1].numpy())

    # Train networks with new optimizer objects
    opt_g = Adam(learning_rate=lr)
    opt_h = Adam(learning_rate=lr)

    print('Training the network with Koopman loss...')
    loss_history, rec_loss_hist, lin_loss_hist, pred_loss_hist = koopman.train_networks(
        opt_g, opt_h, g, h, X_tensor, Y_tensor, u_tensor, EPOCHS=tf.constant(epochs))
    print('FINISHED with final train loss: ',loss_history[-1].numpy())

    ############################################################################
    # Make new predictions
    ############################################################################
    gX = g(X_tensor)
    gY = g(Y_tensor)

    # Calc DMD modes and eigenvalues
    lam_tensor,w_tensor,v_tensor, _ = koopman.get_dmd_modes_tensors(gX,gY)

    # Predict the system
    Yhat_tensor,_ = koopman.predict_koopman(lam_tensor, w_tensor, v_tensor, X_tensor[0,:],
                                    tf.constant(ncp), g, h, u=u_tensor)
    Yhat = Yhat_tensor.numpy().T
    x1_koopman = Yhat[0,:]
    x2_koopman = Yhat[2,:]

    ############################################################################
    # Perform DMD for comparison
    ############################################################################
    lam_dmd,w_dmd,v_dmd,_ = dmd.get_dmd_modes(X,Y)
    Yhat_dmd = dmd.predict(lam_dmd, w_dmd, v_dmd, X[:,0], ncp, u=u)
    Yhat_dmd = Yhat_dmd
    x1_dmd = Yhat_dmd[0,:]
    x2_dmd = Yhat_dmd[2,:]

    ############################################################################
    # Errors, prints and plots
    ############################################################################
    print('Train error Koopman')
    print('   x1: ', 1/m_stop*norm(x1_koopman[:m_stop]-x1_target[:m_stop],2))
    print('   x2: ', 1/m_stop*norm(x2_koopman[:m_stop]-x2_target[:m_stop],2))
    print('Train error DMD')
    print('   x1: ', 1/m_stop*norm(x1_dmd[:m_stop]-x1_target[:m_stop],2))
    print('   x2: ', 1/m_stop*norm(x2_dmd[:m_stop]-x2_target[:m_stop],2))
    if m_stop<ncp:
        print('Test error Koopman')
        print('   x1: ', 1/(ncp-m_stop)*norm(x1_koopman[m_stop:]-x1_target[m_stop:],2))
        print('   x2: ', 1/(ncp-m_stop)*norm(x2_koopman[m_stop:]-x2_target[m_stop:],2))

        print('Test error DMD')
        print('   x1: ', 1/(ncp-m_stop)*norm(x1_dmd[m_stop:]-x1_target[m_stop:],2))
        print('   x2: ', 1/(ncp-m_stop)*norm(x2_dmd[m_stop:]-x2_target[m_stop:],2))

    plt.figure()
    plt.plot(t,x1_target,'-b')
    plt.plot(t,x1_dmd,'--r')
    plt.plot(t,x1_koopman,'-.k')
    plt.xlim(0,dt*ncp)
    plt.xlabel('Time (s)')
    plt.ylabel('Possition of mass 1 (m)')
    plt.grid(True)
    plt.legend(('Simulation','DMD','Koopman'),bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=2)

    plt.figure()
    plt.plot(t,x2_target,'-b')
    plt.plot(t,x2_dmd,'--r')
    plt.plot(t,x2_koopman,'-.k')
    plt.xlim(0,dt*ncp)
    plt.xlabel('Time (s)')
    plt.ylabel('Possition of mass 2 (m)')
    plt.grid(True)
    plt.legend(('Simulation','DMD','Koopman'),bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=2)
    plt.show()
