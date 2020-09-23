import sys
sys.path.append('../')
from dmd import dmd
import numpy as np

# Tensorflow and keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam


def predict_koopman(lam, w, v, x0, ncp, g, h, u=None):
    """Predict the future dynamics of the system given an initial value `x0`. Result is returned
    as a matrix where rows correspond to states and columns to time.

    Args:
        lam (tf.Tensor): Koopman eigenvalues.
        w (tf.Tensor): Left eigenvectors.
        v (tf.Tensor): Right eigenvectors.
        x0 (tf.Tensor): Initial value of the system.
        N (int): Number of time steps to predict.
        g (Net): Encoder network.
        h (Net): Decoder network.
        u (tf.Tensor): Input signal.

    Returns:
        tuple: Prediction of the states of the system for N time steps into the future,
               prediction of the observables of the system for N time steps into the future.

    """
    # Precompute some constants for more efficient computations
    wH = tf.linalg.adjoint(w)
    norm_vec = 1/tf.math.reduce_sum(tf.math.multiply(tf.math.conj(w),v), axis=0)

    # Store each time step in a list
    res_x = tf.TensorArray(x0.dtype,size=ncp+1)
    res_gx = tf.TensorArray(w.dtype,size=ncp+1)
    res_x = res_x.write(0,x0)
    res_gx = res_gx.write(0,tf.cast(tf.squeeze(g(tf.expand_dims(x0,0)),axis=[0]), w.dtype))

    # Initiate time stepping
    xk = x0
    if u is not None:
        for k in range(1,ncp+1):
            xk = tf.concat([tf.expand_dims(xk[:-1],0),tf.reshape(u[k-1],[1,-1])],axis=1)
            xk, gxk = one_step_pred(lam, wH, v, norm_vec, xk, g, h)
            res_x = res_x.write(k,xk)
            res_gx = res_gx.write(k,gxk)
    else:
        for k in range(1,ncp+1):
            xk = tf.expand_dims(xk,0)
            xk, gxk = one_step_pred(lam, wH, v, norm_vec, xk, g, h)
            res_x = res_x.write(k,xk)
            res_gx = res_gx.write(k,gxk)
    return res_x.stack(), res_gx.stack()


def one_step_pred(lam, wH, v, norm_vec, xk, g, h):
    """Performs a one-step prediction of the system.

    Args:
        xk (tf.Tensor): Solution (states) at time step k.
        lam (tf.Tensor): Koopman eigenvalues.
        wH (tf.Tensor): Complex conjugated left eigenvectors.
        v (tf.Tensor): Right eigenvectors.
        norm_vec (tf.Tensor): Normalization vector.
        g (Net): Encoder network.
        h (Net): Decoder network.

    Returns:
        ndarray: Prediction of the states of the system at time step k+1.
            Note that the returned tensor has one less dimension than `xk`.

    """
    # Apply observable function
    gx = tf.cast(g(xk),wH.dtype)

    # Advance one time step and reshape
    gxhat = tf.math.reduce_sum(lam*tf.linalg.matvec(wH,gx[0,:])*norm_vec*v, axis=1)
    gxhat = tf.expand_dims(gxhat,0)

    # Go back to states
    xhat = h(gxhat)

    # Return prediction of new state vector
    return tf.squeeze(xhat,axis=[0]), tf.squeeze(gxhat,axis=[0])


def get_dmd_modes_tensors(X,Y):
    """Computes the DMD modes `v` and eigenvalues `lam` of the data matrices `X`,`Y`.

    Args:
        X (tf.Tensor): First data matrix.
        Y (tf.Tensor): Second data matrix.

    Returns:
        tuple: (Lam,W,V,A). Eigenvalues, left eigenvectors, right eigenvectors and the matrix A.

    """
    S,U,V = tf.linalg.svd(tf.transpose(X),full_matrices=False,compute_uv=True)
    A = tf.linalg.matmul(tf.linalg.matmul(tf.linalg.matmul(tf.transpose(Y),V),tf.linalg.tensor_diag(1/S)),U,adjoint_b=True)

    # Compute left and right eigenvectors as well as eigenvalues of A
    Lam,V = tf.linalg.eig(A,name='eigvals_right')
    _,W = tf.linalg.eig(tf.linalg.adjoint(A),name='eigvals_left')

    # Reorder columns of W to form a biorthogonal base with the columns of V
    mask = tf.cast((abs(tf.linalg.matmul(W,V,adjoint_a=True)) > 1e-10),W.dtype)
    W = tf.linalg.matmul(W,mask)
    return Lam, W, V, A


def RMSloss(target, predicted, scale=tf.constant(1.0,dtype=tf.float32)):
    """Computes the RMS loss of the differance between `target` and `predicted`. The result can then be scaled with a factor `scale`.

    Args:
        target (tf.Tensor): Ground truth.
        predicted (tf.Tensor): Prediction.
        scale (tf.Tensor): Scaling factor.

    Returns:
        tf.Tensor: Loss.

    """
    return scale*tf.reduce_mean(tf.square(tf.math.abs(target - predicted)))


@tf.function
def koopman_loss(g, h, X, Y, u):
    """Computes the Koopman loss, which is a sum of linearity loss,
    reconstruction loss and prediction loss.

    Args:
        g (Net): Encoder network.
        h (Net): Decoder network.
        X (tf.Tensor): First data matrix.
        Y (tf.Tensor): Second data matrix.
        u (tf.Tensor): Input signal.

    Returns:
        tuple: total loss, reconstruction loss, linearity loss, prediction loss.

    """
    # Encode tensors
    gX = g(X)
    gY = g(Y)

    lam, w, v, Ahat = get_dmd_modes_tensors(gX,gY)

    # Multi step predictions
    Yhat, gYhat = predict_koopman(lam, w, v, X[0,:], tf.constant(X.shape[0]), g, h, u=u)

    # Decode tensors
    Y_rec = h(gY)

    # Exclude input state from loss if present
    if u is not None:
        stop_idx = -1
    else:
        stop_idx = X.shape[1]

    real_type = X.dtype
    gY = tf.cast(gY,w.dtype)
    l_lin = RMSloss(gY,gYhat[1:,:],scale=tf.constant(1.0,dtype=real_type))
    l_rec = RMSloss(Y[:,:stop_idx],Y_rec[:,:stop_idx],scale=tf.constant(1.0,dtype=real_type))
    l_pred = RMSloss(Y[:,:stop_idx],Yhat[1:,:stop_idx],scale=tf.constant(1.0,dtype=real_type))
    return l_lin+l_rec+l_pred, l_rec, l_lin, l_pred


def train_step(opt_g, opt_h, g, h, X, Y, u, train_loss_tracker, rec_loss_tracker, lin_loss_tracker, pred_loss_tracker):
    """Takes one training step with the Koopman loss.

    Args:
        opt_g (tf.keras.optimizers.Optimizer): Optimizer for encoder network.
        opt_h (tf.keras.optimizers.Optimizer): Optimizer for decoder network.
        g (Net): Encoder network.
        h (Net): Decoder network.
        X (tf.Tensor): First data matrix.
        Y (tf.Tensor): Second data matrix.
        u (tf.Tensor): Input signal.
        train_loss_tracker (tf.keras.metrics.Mean): Tracks the total training loss over time.
        rec_loss_tracker (tf.keras.metrics.Mean): Tracks the reconstruction loss over time.
        lin_loss_tracker (tf.keras.metrics.Mean): Tracks the linearity loss over time.
        pred_loss_tracker (tf.keras.metrics.Mean): Tracks the prediction loss over time.

    """
    # Compute grad w.r.t. `g`
    with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
        tape.watch(g.variables)
        tape.watch(h.variables)
        loss, loss_rec, loss_lin, loss_pred = koopman_loss(g, h, X, Y, u)
    grads_g = tape.gradient(loss, g.trainable_variables)
    grads_h = tape.gradient(loss, h.trainable_variables)
    del tape

    # Update weights
    opt_g.apply_gradients(zip(grads_g, g.trainable_variables))
    opt_h.apply_gradients(zip(grads_h, h.trainable_variables))

    train_loss_tracker(loss)
    rec_loss_tracker(loss_rec)
    lin_loss_tracker(loss_lin)
    pred_loss_tracker(loss_pred)


def train_networks(opt_g, opt_h, g, h, X, Y, u, EPOCHS=tf.constant(5)):
    """Train the networks for `EPOCHS` number of epochs with the Koopman loss.

    Args:
        opt_g (tf.keras.optimizers.Optimizer): Optimizer for encoder network.
        opt_h (tf.keras.optimizers.Optimizer): Optimizer for decoder network.
        g (Net): Encoder network.
        h (Net): Decoder network.
        X (tf.Tensor): First data matrix.
        Y (tf.Tensor): Second data matrix.
        u (tf.Tensor): Input signal.
        EPOCHS (tf.Tensor): Number of epochs.

    Returns:
        tuple: (loss_history, rec_loss_hist, rec_loss_hist, rec_loss_hist).

    """
    loss_history = []
    rec_loss_hist = []
    lin_loss_hist = []
    pred_loss_hist = []
    train_loss_tracker = tf.keras.metrics.Mean(name='train_loss')
    rec_loss_tracker = tf.keras.metrics.Mean(name='rec_loss')
    lin_loss_tracker = tf.keras.metrics.Mean(name='lin_loss')
    pred_loss_tracker = tf.keras.metrics.Mean(name='pred_loss')
    for epoch in range(EPOCHS):
        # Reset the metrics at the start of the next epoch
        train_loss_tracker.reset_states()
        rec_loss_tracker.reset_states()
        lin_loss_tracker.reset_states()
        pred_loss_tracker.reset_states()

        train_step(opt_g, opt_h, g, h, X, Y, u, train_loss_tracker, rec_loss_tracker, lin_loss_tracker, pred_loss_tracker)

        print('Epoch {}, Loss: {}'.format(epoch+1, train_loss_tracker.result()))

        loss_history.append(train_loss_tracker.result())
        rec_loss_hist.append(rec_loss_tracker.result())
        lin_loss_hist.append(lin_loss_tracker.result())
        pred_loss_hist.append(pred_loss_tracker.result())

    return loss_history, rec_loss_hist, lin_loss_hist, pred_loss_hist


@tf.function(experimental_relax_shapes=True)
def pre_train_loss(g, h, Y):
    """Computes the pre-training loss, which is just the reconstruction loss.

    Args:
        g (Net): Encoder network.
        h (Net): Decoder network.
        Y (tf.Tensor): Second data matrix.

    Returns:
        tf.Tensor: Reconstruction loss.

    """
    gY = g(Y)
    Y_rec = h(gY)
    return RMSloss(Y, Y_rec,scale=tf.constant(1.0,dtype=Y_rec.dtype))


def pre_train_step(opt_g, opt_h, g, h, Y, train_loss_tracker):
    """Takes one pre-training step.

    Args:
        opt_g (tf.keras.optimizers.Optimizer): Optimizer for encoder network.
        opt_h (tf.keras.optimizers.Optimizer): Optimizer for decoder network.
        g (Net): Encoder network.
        h (Net): Decoder network.
        Y (tf.Tensor): Second data matrix.
        train_loss_tracker (tf.keras.metrics.Mean): Tracks the total training loss over time.

    """
    # Compute grad w.r.t. `g`
    with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
        tape.watch(g.variables)
        tape.watch(h.variables)
        loss = pre_train_loss(g, h, Y)
    grads_g = tape.gradient(loss, g.trainable_variables)
    grads_h = tape.gradient(loss, h.trainable_variables)
    del tape

    opt_g.apply_gradients(zip(grads_g, g.trainable_variables))
    opt_h.apply_gradients(zip(grads_h, h.trainable_variables))

    train_loss_tracker(loss)


def pre_train_networks(opt_g, opt_h, g, h, Y, EPOCHS=tf.constant(5)):
    """Train the networks for `EPOCHS` number of epochs with the pre-training loss.

    Args:
        opt_g (tf.keras.optimizers.Optimizer): Optimizer for encoder network.
        opt_h (tf.keras.optimizers.Optimizer): Optimizer for decoder network.
        g (Net): Encoder network.
        h (Net): Decoder network.
        Y (tf.Tensor): Second data matrix.
        EPOCHS (tf.Tensor): Number of epochs.

    Returns:
        tf.Tensor: loss history.

    """
    loss_history = []
    train_loss_tracker = tf.keras.metrics.Mean(name='train_loss')
    for epoch in range(EPOCHS):
        # Reset the metrics at the start of the next epoch
        train_loss_tracker.reset_states()

        pre_train_step(opt_g, opt_h, g, h, Y, train_loss_tracker)

        print('Epoch {}, Loss: {}'.format(epoch+1, train_loss_tracker.result()))

        loss_history.append(train_loss_tracker.result())
    return loss_history


class Net(Model):
    """
    Neural network consisting of a number of dense layers.
    """
    def __init__(self, n_in, n_out, w_hidden=30, nbr_hidden=1, activation='relu', weight_decay=0.0):
        """Constructs a neural network made of dense layers.

        Args:
            n_in (int): Input dimension.
            n_out (int): Output dimension.
            w_hidden (int): Width of the hidden layers.
            nbr_hidden (int): NUmber of hidden layers.
            activation (str): Activation function for all layers.
            weight_decay (double): Regularisation strength for all layers.

        Returns:
            Net: The neural network.

        """
        super(Net, self).__init__()
        self.in_layer = Dense(w_hidden, input_shape=(n_in,), use_bias=True, activation=activation,
            kernel_initializer=keras.initializers.RandomUniform(minval=-np.sqrt(1/n_in), maxval=np.sqrt(1/n_in), seed=37),
            kernel_regularizer=keras.regularizers.l2(weight_decay))

        self.hidden_layers = []
        for i in range(nbr_hidden):
            self.hidden_layers.append(Dense(w_hidden, use_bias=True, activation=activation,
                kernel_initializer=keras.initializers.RandomUniform(minval=-np.sqrt(1/w_hidden), maxval=np.sqrt(1/w_hidden), seed=38),
                kernel_regularizer=keras.regularizers.l2(weight_decay)))

        self.out_layer = Dense(n_out, use_bias=True, activation=None,
            kernel_initializer=keras.initializers.RandomUniform(minval=-np.sqrt(1/w_hidden), maxval=np.sqrt(1/w_hidden), seed=39),
            kernel_regularizer=keras.regularizers.l2(weight_decay))

    def call(self, x):
        """Forward pass of the neural network.

        Args:
            x (tf.Tensor): Input to the network.

        Returns:
            tf.Tensor: The output from the network.

        """
        x = self.in_layer(x)

        for hidden_i in self.hidden_layers:
            x = hidden_i(x)

        x = self.out_layer(x)
        return x
