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


class SimpleNet(Model):
    """
    Very simple neural network consisting of a dense layers.
    """
    def __init__(self, n_in, n_out, weight_decay=0.0):
        """Constructs a neural network made of a dense layer.

        Args:
            n_in (int): Input dimension.
            n_out (int): Output dimension.
            weight_decay (double): Regularisation strength for all layers.

        Returns:
            Net: The neural network.

        """
        super(SimpleNet, self).__init__()
        self.out_layer = Dense(
            n_out, input_shape=(n_in,), use_bias=False, activation=None,
            kernel_initializer=keras.initializers.RandomUniform(minval=-np.sqrt(1/n_in), maxval=np.sqrt(1/n_in), seed=37),
            kernel_regularizer=keras.regularizers.l2(weight_decay))

    def call(self, x):
        """Forward pass of the neural network.

        Args:
            x (tf.Tensor): Input to the network.

        Returns:
            tf.Tensor: The output from the network.

        """
        x = self.out_layer(x)
        return x


if __name__ == '__main__':
    # Simulation params
    t_start = 0.0               # start simulation time
    t_stop = 20.0               # stop simulations time
    t_stop_train = 10.0         # only train with data up to this time
    ncp = 1000                  # number of communication points
    amp = 10.0                  # amplitude of input signal
    freq = 0.1                  # frequency of input signal
    inp = 'sin'                 # input type: delta or sin

    # NN and training params
    lr = 2.5e-4
    epochs = 2000
    pre_lr = 1e-2
    pre_epochs = 160
    w_hidden = 30               # Dimension of hidden layer

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
    data_sim = dmd.get_snapshots_damped_dual_mass(
        t_start, t_stop, ncp , input_force=u, time_vec=t, states=['mass1.s', 'mass1.v','mass2.s','mass2.v'])
    x1_target = data_sim[0,:]
    x2_target = data_sim[2,:]

    # Get data matrices
    X,Y = dmd.get_data_matrices(data_sim, m_stop=m_stop, u=u)

    ############################################################################
    # Construct and train NN
    ############################################################################
    n_x, _ = X.shape            # input dim
    n = n_x                     # output dim

    # Convert arrays to tensors
    X_tensor = tf.constant(X.T, dtype=tf.float64)
    Y_tensor = tf.constant(Y.T, dtype=tf.float64)
    if u is not None:
        u_tensor = tf.constant(u, dtype=tf.float64)
    else:
        u_tensor = None

    # Define networks
    g = SimpleNet(n_x, n)
    h = SimpleNet(n, n_x)

    # Initiate weights
    g(tf.expand_dims(tf.ones(n_x, dtype=tf.dtypes.float64),0))
    h(tf.expand_dims(tf.ones(n, dtype=tf.dtypes.float64),0))

    # Create optimizer objects for pre-training
    pre_opt_g = Adam(learning_rate=pre_lr)
    pre_opt_h = Adam(learning_rate=pre_lr)

    # Pre-train network
    print('Pre-training the network as an autoencoder...')
    pre_loss_history = koopman.pre_train_networks(pre_opt_g, pre_opt_h, g,
                                                  h, Y_tensor, EPOCHS=tf.constant(pre_epochs))
    print('FINISHED with final pre-train loss: ',pre_loss_history[-1].numpy())

    # Train network
    opt_g = Adam(learning_rate=lr)
    opt_h = Adam(learning_rate=lr)

    print('Training the network with Koopman loss...')
    loss_history, rec_loss_hist, lin_loss_hist, pred_loss_hist = koopman.train_networks(
        opt_g, opt_h, g, h, X_tensor, Y_tensor, u_tensor, EPOCHS=tf.constant(epochs))
    print('FINISHED with final train loss: ',loss_history[-1].numpy())

    ############################################################################
    # Make new predictions
    ############################################################################
    print('Making new predictions...')
    gX = g(X_tensor)
    gY = g(Y_tensor)

    # Calc DMD modes and eigenvalues
    lam_tensor,w_tensor,v_tensor, Ahat = koopman.get_dmd_modes_tensors(gX,gY)

    # Predict the system
    Yhat_tensor,_ = koopman.predict_koopman(lam_tensor, w_tensor, v_tensor, X_tensor[0,:],
                                tf.constant(ncp), g, h, u=u_tensor)
    Yhat = Yhat_tensor.numpy().T
    x1_koopman = Yhat[0,:]
    x2_koopman = Yhat[2,:]
    print('FINISHED with predictions')

    ############################################################################
    # Perform DMD for comparison
    ############################################################################
    lam_dmd,w_dmd,v_dmd,_ = dmd.get_dmd_modes(X,Y)
    Yhat_dmd = dmd.predict(lam_dmd, w_dmd, v_dmd, X[:,0], ncp, u=u)
    x1_dmd = Yhat_dmd[0,:]
    x2_dmd = Yhat_dmd[2,:]

    ############################################################################
    # Plot
    ############################################################################
    print('PLOTS')
    plt.figure()
    plt.title('Simulated data and m={}'.format(m_stop))
    plt.plot(t,x1_target,'-b',linewidth=2)
    plt.plot(t,x1_dmd,'--r',linewidth=2)
    plt.plot(t,x1_koopman,'-.k',linewidth=2)
    plt.legend(('Simulated solution', 'DMD', 'Koopman'))
    plt.xlim(0, dt*(ncp))
    plt.xlabel('Time (s)')
    plt.ylabel('Position of mass 1 (m)')
    if m_stop < ncp:
        plt.axvline(x=t_stop_train, color='k', linestyle='-',linewidth=1)

    plt.figure()
    plt.title('Simulated data and m={}'.format(m_stop))
    plt.plot(t,x2_target,'-b',linewidth=2)
    plt.plot(t,x2_dmd,'--r',linewidth=2)
    plt.plot(t,x2_koopman,'-.k',linewidth=2)
    plt.legend(('Simulated solution', 'DMD', 'Koopman'))
    plt.xlim(0, dt*(ncp))
    plt.xlabel('Time (s)')
    plt.ylabel('Position of mass 2 (m)')
    if m_stop < ncp:
        plt.axvline(x=t_stop_train, color='k', linestyle='-',linewidth=1)

    plt.figure()
    plt.title('Pre-trained auto encoder result')
    plt.plot(t[:m_stop-1],Y[2,:],'-b',linewidth=2)
    plt.plot(t[:m_stop-1],h(g(Y_tensor)).numpy().T[2,:],'--r',linewidth=2)
    plt.legend(('Source','Reconstructed'))
    plt.xlim(0, dt*(ncp))
    plt.xlabel('Time (s)')
    plt.ylabel('Position of mass 2 (m)')

    plt.figure()
    plt.semilogy(pre_loss_history,'-xb')
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Pre-train loss')
    plt.xlim(0,pre_epochs)

    plt.figure()
    plt.title('Total loss')
    plt.semilogy(loss_history,'-xb')
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Train loss')
    plt.xlim(0,epochs)

    plt.figure()
    plt.title('Individual losses')
    plt.semilogy(rec_loss_hist,'-og')
    plt.semilogy(lin_loss_hist,'-or')
    plt.semilogy(pred_loss_hist,'-ok')
    plt.grid(True)
    plt.legend(('Rec loss', 'Lin loss', 'Pred loss'))
    plt.xlabel('Epoch')
    plt.ylabel('Train loss')
    plt.xlim(0,epochs)

    ############################################################################
    # Errors
    ############################################################################
    print('Train error Koopman')
    print('   x1: ', 1/m_stop*norm(x1_koopman[:m_stop]-x1_target[:m_stop],2))
    print('   x2: ', 1/m_stop*norm(x2_koopman[:m_stop]-x2_target[:m_stop],2))
    print('  tot: ', 1/4*1/m_stop*norm(Yhat[:4,:m_stop]-data_sim[:4,:m_stop],'fro'))
    print('Train error DMD')
    print('   x1: ', 1/m_stop*norm(x1_dmd[:m_stop]-x1_target[:m_stop],2))
    print('   x2: ', 1/m_stop*norm(x2_dmd[:m_stop]-x2_target[:m_stop],2))
    print('  tot: ', 1/4*1/m_stop*norm(Yhat_dmd[:4,:m_stop]-data_sim[:4,:m_stop],'fro'))

    if m_stop<ncp:
        print('Test error Koopman')
        print('   x1: ', 1/(ncp-m_stop)*norm(x1_koopman[m_stop:]-x1_target[m_stop:],2))
        print('   x2: ', 1/(ncp-m_stop)*norm(x2_koopman[m_stop:]-x2_target[m_stop:],2))
        print('  tot: ', 1/4*1/(ncp-m_stop)*norm(Yhat[:4,m_stop:]-data_sim[:4,m_stop:],'fro'))

        print('Test error DMD')
        print('   x1: ', 1/(ncp-m_stop)*norm(x1_dmd[m_stop:]-x1_target[m_stop:],2))
        print('   x2: ', 1/(ncp-m_stop)*norm(x2_dmd[m_stop:]-x2_target[m_stop:],2))
        print('  tot: ', 1/4*1/(ncp-m_stop)*norm(Yhat_dmd[:4,m_stop:]-data_sim[:4,m_stop:],'fro'))

    plt.show()
