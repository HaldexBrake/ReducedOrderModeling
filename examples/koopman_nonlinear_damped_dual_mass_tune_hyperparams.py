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


def hyper_tuning(X, Y, u, lr_list, epochs_list, pre_lr_list, pre_epochs_list, n_list, w_hidden_list, t_stop_train, dt, ncp, x1_target, x2_target, plots=True):
    """Tunes the hyperparameter by testing all combinations of the provided values.

    Args:
        X (ndarray): First data matrix.
        Y (ndarray): Second data matrix.
        u (ndarray): Input signal.
        lr_list (list): List of learning rates.
        epochs_list (list): List of the number of epochs the network should be trained.
        pre_lr_list (double): List of learning rates for pretraining.
        pre_epochs_list (list): List of different number of epochs the network should be pretrained with.
        n_list (list): List of the output dimensions of the encoder network to test.
        w_hidden_list (list): List of the widths of the hidden layers in the networks to test.
        t_stop_train (double): Only train with data up to this time.
        dt (double): Time step size.
        ncp (int): Number of prediction steps.
        x1_target (ndarray): Ground truth for x1.
        x2_target (ndarray): Ground truth for x2.
        plots (bool): Create plots for this run.

    Returns:
        tuple: (result_string, res). String presenting the findings in a fairly structured way, dict containing errors, losses and parameter values.

    """
    # Convert arrays to tensors
    X_tensor = tf.constant(X.T, dtype=tf.float64)
    Y_tensor = tf.constant(Y.T, dtype=tf.float64)
    if u is not None:
        u_tensor = tf.constant(u, dtype=tf.float64)
    else:
        u_tensor = None

    result_string = ''
    res = {}
    id_string = 'lr: {}, epochs: {}, pre_lr: {}, pre_epochs: {}, n: {}, w_hidden: {}'
    nbr_of_runs = len(lr_list)*len(epochs_list)*len(pre_lr_list)*len(pre_epochs_list)*len(n_list)*len(w_hidden_list)

    # Giant loop over all params
    run = 1
    for epochs in epochs_list:
        for lr in lr_list:
            for pre_lr in pre_lr_list:
                for pre_epochs in pre_epochs_list:
                    for n in n_list:
                        for w_hidden in w_hidden_list:
                            #String with all parameters to identify the run
                            params_string = id_string.format(lr, epochs, pre_lr,
                                                             pre_epochs, n, w_hidden)
                            print(('RUN {}/{}\n'+params_string).format(run,nbr_of_runs))

                            res['run{}'.format(run)] = {'params_string':params_string}
                            result_string += '\nRUN {}/{}   '.format(run,nbr_of_runs) + params_string + '\n'

                            try:
                                # Perform one iteration of the hyperparameter tuning
                                errors, final_loss, final_pre_loss = hyper_step(X_tensor, Y_tensor, u_tensor, lr, epochs,
                                                    pre_lr, pre_epochs, n, w_hidden, t_stop_train,
                                                    dt, ncp, x1_target, x2_target, run, plots=plots)

                                # Store result in dict
                                res['run{}'.format(run)]['x1_train'] = errors['x1_train']
                                res['run{}'.format(run)]['x2_train'] = errors['x2_train']
                                res['run{}'.format(run)]['x1_test'] = errors['x1_test']
                                res['run{}'.format(run)]['x2_test'] = errors['x2_test']
                                res['run{}'.format(run)]['loss'] = final_loss
                                res['run{}'.format(run)]['pre_loss'] = final_pre_loss
                                res['run{}'.format(run)]['n'] = n
                                res['run{}'.format(run)]['w'] = w_hidden
                                res['run{}'.format(run)]['lr'] = lr


                                # Add results to the results string
                                result_string += '   Final pre-loss: {}'.format(final_pre_loss)+'\n'
                                result_string += '   Final loss: {}'.format(final_loss)+'\n'
                                result_string += '   Train error'+'\n'
                                result_string += '      x1: {}'.format(errors['x1_train'])+'\n'
                                result_string += '      x2: {}'.format(errors['x2_train'])+'\n'
                                result_string += '   Test error'+'\n'
                                result_string += '      x1: {}'.format(errors['x1_test'])+'\n'
                                result_string += '      x2: {}'.format(errors['x2_test'])+'\n'
                            except:
                                res['run{}'.format(run)]['loss'] = 'FAIL'
                                result_string += '---FAIL---'
                            run += 1
    return result_string, res


def hyper_step(X_tensor, Y_tensor, u_tensor, lr, epochs, pre_lr, pre_epochs, n, w_hidden, t_stop_train, dt, ncp, x1_target, x2_target, run, plots=True):
    """Test one set of parameters in the hyperparameter tuning. Evaluate the model
    and save losses and prediction errors.

    Args:
        X_tensor (tf.Tensor): First data matrix.
        Y_tensor (tf.Tensor): Second data matrix.
        u_tensor (tf.Tensor): Input signal.
        lr (double): Learning rate.
        epochs (int): The number of epochs the network should be trained.
        pre_lr (double): Learning rate for pretraining.
        pre_epochs (int): The number of epochs the network should be pretrained.
        n (int): Output dimension of the encoder network.
        w_hidden (int): Width of the hidden layers in the networks.
        t_stop_train (double): Only train with data up to this time.
        dt (double): Time step size.
        ncp (int): Number of prediction steps.
        x1_target (ndarray): Ground truth for x1.
        x2_target (ndarray): Ground truth for x2.
        run (int): Keep track of which run in the tuning currently is running.
        plots (bool): Create plots for this run.

    Returns:
        tuple: (errors, final_loss, final_pre_loss).

    """
    ############################################################################
    # Construct and train NN
    ############################################################################
    # NN and training params
    m_stop, n_x = X_tensor.shape

    # Define networks
    g = Net(n_x, n, w_hidden=w_hidden, activation='relu')
    h = Net(n, n_x, w_hidden=w_hidden, activation='relu')

    # Initiate weights
    g(tf.expand_dims(tf.ones(n_x, dtype=tf.dtypes.float64),0))
    h(tf.expand_dims(tf.ones(n, dtype=tf.dtypes.float64),0))

    # Create optimizer objects
    opt_g = Adam(learning_rate=pre_lr)
    opt_h = Adam(learning_rate=pre_lr)

    # Pre-train network
    pre_loss_history = koopman.pre_train_networks(opt_g, opt_h, g, h, Y_tensor, EPOCHS=tf.constant(pre_epochs))
    print('Final pre-train loss: ',pre_loss_history[-1])
    if plots:
        plt.figure()
        plt.title('Run {}'.format(run))
        plt.semilogy(pre_loss_history,'-xb')
        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Pre-train loss')
        plt.xlim(0,pre_epochs)

        plt.figure()
        plt.title('Pre-trained auto encoder result')
        plt.plot(Y_tensor.numpy().T[2,:],'-b',linewidth=2)
        plt.plot(Y_tensor.numpy().T[0,:],'-k',linewidth=2)
        plt.plot(h(g(Y_tensor)).numpy().T[2,:],'--r',linewidth=2)
        plt.plot(h(g(Y_tensor)).numpy().T[0,:],'--m',linewidth=2)
        plt.legend(('Source x2','Source x1','Reconstructed x2','Reconstructed x1'))
        plt.ylabel('Position of mass 2 (m)')

    # Train networks with new optimizer objects
    opt_g = Adam(learning_rate=lr)
    opt_h = Adam(learning_rate=lr)

    loss_history, rec_loss_hist, lin_loss_hist, pred_loss_hist = koopman.train_networks(
        opt_g, opt_h, g, h, X_tensor, Y_tensor, u_tensor, EPOCHS=tf.constant(epochs))

    ############################################################################
    # Make new predictions
    ############################################################################
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

    # Save errors
    errors  = {'x1_train':0.0, 'x2_train':0.0, 'x1_test':0.0, 'x2_test':0.0}
    errors['x1_train'] = 1/m_stop*norm(x1_koopman[:m_stop]-x1_target[:m_stop],2)
    errors['x2_train'] = 1/m_stop*norm(x2_koopman[:m_stop]-x2_target[:m_stop],2)
    if m_stop<ncp:
        errors['x1_test'] = 1/(ncp-m_stop)*norm(x1_koopman[m_stop:]-x1_target[m_stop:],2)
        errors['x2_test'] = 1/(ncp-m_stop)*norm(x2_koopman[m_stop:]-x2_target[m_stop:],2)

    # Save final loss
    final_loss = loss_history[-1].numpy()
    final_pre_loss = pre_loss_history[-1].numpy()

    ############################################################################
    # Plots
    ############################################################################
    if plots:
        plt.figure()
        plt.title('Run {}'.format(run))
        plt.plot(t,x1_target,'-b',linewidth=2)
        plt.plot(t,x1_koopman,'-.k',linewidth=2)
        plt.legend(('Simulated solution','Koopman'))
        plt.xlim(0, dt*(ncp))
        plt.xlabel('Time (s)')
        plt.ylabel('Position of mass 1 (m)')
        if m_stop < ncp:
            plt.axvline(x=t_stop_train, color='k', linestyle='-',linewidth=1)

        plt.figure()
        plt.title('Run {}'.format(run))
        plt.plot(t,x2_target,'-b',linewidth=2)
        plt.plot(t,x2_koopman,'-.k',linewidth=2)
        plt.legend(('Simulated solution','Koopman'))
        plt.xlim(0, dt*(ncp))
        plt.xlabel('Time (s)')
        plt.ylabel('Position of mass 2 (m)')
        if m_stop < ncp:
            plt.axvline(x=t_stop_train, color='k', linestyle='-',linewidth=1)

        plt.figure()
        plt.title('Run {}'.format(run))
        plt.semilogy(loss_history,'-xb')
        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Total train loss')
        plt.xlim(0,epochs)

        plt.figure()
        plt.title('Run {}'.format(run))
        plt.semilogy(rec_loss_hist,'-og')
        plt.semilogy(lin_loss_hist,'-or')
        plt.semilogy(pred_loss_hist,'-ok')
        plt.grid(True)
        plt.legend(('Rec loss', 'Lin loss', 'Pred loss'))
        plt.xlabel('Epoch')
        plt.ylabel('Train loss')
        plt.xlim(0,epochs)

    return errors, final_loss, final_pre_loss


if __name__ == '__main__':
    # Simulation params
    t_start = 0.0               # start simulation time
    t_stop = 40.0               # stop simulations time
    t_stop_train = 20.0         # only train with data up to this time
    ncp = 4000                  # number of communication points
    amp = 30.0                  # amplitude of input signal
    freq = 0.1                  # frequency of input signal
    inp = 'sin'                 # input type: delta or sin

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
    # Run hyperparametertuning
    ############################################################################
    # parameter to tune
    lr = [6e-4,5e-4,4.5e-4,4e-3,3.5e-4,3e-4,8e-5] #8e-5 best
    pre_lr = [5e-3]
    epochs = [200,400,571,800,1000,1300,1700] #400 best
    pre_epochs = [300]
    n = [21]
    w_hidden = [60]
    plots = False

    result_string, res_dict = hyper_tuning(X, Y, u, lr, epochs, pre_lr,
                                           pre_epochs, n, w_hidden, t_stop_train, dt, ncp,
                                           x1_target, x2_target, plots=plots)

    ############################################################################
    # Errors, prints and plots
    ############################################################################
    print(result_string)

    if plots:
        plt.show()
