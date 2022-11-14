import numpy as np

from scipy.linalg import solve_continuous_lyapunov

import sklearn.linear_model as skllm

import tensorflow as tf

class SymmetricRidge(skllm.Ridge):
    def __init__(self, alpha=1.0):
        super().__init__(alpha,fit_intercept=False)

    def fit(self, X, y, sample_weight=None):
        X, y = self._validate_data(X, y,
                    accept_sparse=False,
                    dtype=[np.float64, np.float32],
                    multi_output=True, y_numeric=True)
        
        self.coef_ = self._solve_symmetric_ridge(X,y,self.alpha)
        self.intercept_ = 0.

    def _solve_symmetric_ridge(self,X,y,alpha=1.0):
        # https://mathoverflow.net/a/340001
        # A->X; H->A; K->Q; X->coef
        A = X.T @ X
        Q = X.T @ y + y.T @ X

        # L2 regulatization: add alpha to diagonal
        A.flat[::A.shape[-1] + 1] += alpha
 
        coef = solve_continuous_lyapunov(A,Q)

        return coef

force_unit = 1e-3
strain_unit = 10

class MultiSymmetricConstraint(tf.keras.constraints.Constraint):
    def __init__(self, square_side=None, n_squares=1, *args, **kwargs):
        self.square_side = square_side
        self.n_squares = n_squares
        super().__init__(*args,**kwargs)

    def __call__(self, w):
        self.square_side = w.shape[1] if self.square_side is None else self.square_side
        assert w.shape[1] == self.square_side, "Output shape is inconsistent"
        assert w.shape[0] // self.square_side >= self.n_squares, "Output shape is inconsistent"
        new_ws = []
        for i in range(self.n_squares):
            ww = w[i*self.square_side:(i+1)*self.square_side, :]
            new_ws.append(0.5 * (ww + tf.transpose(ww)))
        new_ws.append(w[self.n_squares*self.square_side:, :])
        return tf.concat(new_ws,axis=0)

def tfSymmetricConstraint_Regressor(pod_size,learning_rate=0.1):
    out = tf.keras.models.Sequential([
        #tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(units=pod_size,kernel_constraint=MultiSymmetricConstraint(),use_bias=False),
        tf.keras.layers.Rescaling(scale=force_unit)
    ])

    out.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),loss=tf.keras.losses.MeanSquaredError())
    return out

def tfNSymmetricConstraint_Regressor(pod_size,n_squares,learning_rate=0.1):
    out = tf.keras.models.Sequential([
        #tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(units=pod_size,kernel_constraint=MultiSymmetricConstraint(pod_size, n_squares),use_bias=False),
        tf.keras.layers.Rescaling(scale=force_unit)
    ])

    out.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),loss=tf.keras.losses.MeanSquaredError())
    return out

def tfNSymmetricConstraint_NormRegressor(pod_size,n_squares,learning_rate=0.1):
    out = tf.keras.models.Sequential([
        tf.keras.layers.BatchNormalization(center=False),
        tf.keras.layers.Dense(units=pod_size,kernel_constraint=MultiSymmetricConstraint(pod_size, n_squares),use_bias=False),
        tf.keras.layers.Rescaling(scale=force_unit)
    ])

    out.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),loss=tf.keras.losses.MeanSquaredError())
    return out



class SymmetricLinearLayer(tf.keras.layers.Layer):
    def __init__(self, units=32, n_squares=None,initializer=tf.keras.initializers.GlorotUniform(), **kwargs):
        """Layer with symmetrized weight matrix and no bias

        w = [S1; ...; SN; R], where S1,...,SN are symmetric matrices with size units

        Parameters
        ----------
        units : int, optional
            Output size of the layer, and of the square submatrices, by default 32
        n_squares : int, optional
            Number of square matrices to fit in, by default None which fits as many as possible
        """
        super().__init__(**kwargs)
        self.units = units
        self.n_squares = n_squares
        self.initializer = initializer
    
    def build(self,input_shape):
        self.flat_weights = []
        self.n_squares = input_shape[-1]//self.units if self.n_squares is None else self.n_squares
        for i in range(self.n_squares):
            flat_weight = self._make_flat(self.units)
            self.flat_weights.append(flat_weight)

        self.rect = self.add_weight(
                shape=(input_shape[-1] - self.n_squares*self.units, self.units),
                initializer=self.initializer,
                trainable=True
            )

    def _make_flat(self, N):
        x = self.add_weight(
                shape=(1,N*(N+1)//2),
                initializer=self.initializer,
                trainable=True
            )
        return x

    def _square_from_flat(self,x):
        #https://stackoverflow.com/a/71393652
        # x.shape = [1, n*(n+1)/2]
        n = self.units
        x_rev = tf.reverse(x[:, n:], [1])
        xc = tf.concat([x, x_rev], axis=1)
        x_res = tf.reshape(xc, [n, n])
        x_upper_triangular = tf.linalg.band_part(x_res, 0, -1)
        x_lower_triangular = tf.linalg.set_diag( tf.transpose(x_upper_triangular), tf.zeros([n]))
        return x_upper_triangular + x_lower_triangular

    def get_kernel(self):
        return tf.concat([self._square_from_flat(fw) for fw in self.flat_weights] + [self.rect], axis=0, name='kernel')

    def call(self, x):
        return tf.matmul(x, self.get_kernel())

def tfNSymmetricBuiltin_Regressor(pod_size,n_squares=None,learning_rate=0.1):
    out = tf.keras.models.Sequential([
        SymmetricLinearLayer(units=pod_size,n_squares=n_squares),
        tf.keras.layers.Rescaling(scale=force_unit)
    ])

    out.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),loss=tf.keras.losses.MeanSquaredError())
    return out

def tfNSymmetricBuiltin_NormRegressor(pod_size,n_squares=None,learning_rate=0.1):
    out = tf.keras.models.Sequential([
        tf.keras.layers.BatchNormalization(),
        SymmetricLinearLayer(units=pod_size,n_squares=n_squares),
        tf.keras.layers.Rescaling(scale=force_unit)
    ])

    out.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),loss=tf.keras.losses.MeanSquaredError())
    return out


class MeanSquaredPercentageError(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        return tf.reduce_mean(tf.math.square((y_true-y_pred)/(y_pred + 1e-6*tf.math.sign(y_pred))))

def tfDense(*sizes,learning_rate=0.1,activations=None):

    if activations is None:
        activation = [None] * len(sizes)

    out = tf.keras.models.Sequential([tf.keras.layers.BatchNormalization(center=False)] + [
                                      tf.keras.layers.Dense(units=size,activation=activation)
                                    for size, activation in zip(sizes, activations)
                                 ] + [tf.keras.layers.Rescaling(scale=force_unit)])
    
    out.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),loss=tf.keras.losses.MeanSquaredError())

    return out

def tfDenseSplit(*sizes,tail_size=0,learning_rate=0.1,activations=None):

    if activations is None:
        activation = [None] * len(sizes)

    head_size = sizes[-1] - tail_size

    out = tf.keras.models.Sequential([tf.keras.layers.Rescaling(scale=1/tf.constant([force_unit/10]*head_size+[strain_unit]*tail_size))] + [ 
                                      tf.keras.layers.Dense(units=size,activation=activation)
                                    for size, activation in zip(sizes, activations)
                                 ] + [tf.keras.layers.Rescaling(scale=tf.constant([force_unit/10]*head_size+[strain_unit]*tail_size))])
    
    out.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),loss=tf.keras.losses.MeanSquaredError())

    return out

def tfDenseRescale(*sizes,input_scale=1, input_offset=0, output_scale=1,learning_rate=0.1,activations=None):

    if activations is None:
        activation = [None] * len(sizes)


    out = tf.keras.models.Sequential([tf.keras.layers.Rescaling(scale=input_scale,offset=input_offset)] + [ 
                                      tf.keras.layers.Dense(units=size,activation=activation)
                                    for size, activation in zip(sizes, activations)
                                 ] + [tf.keras.layers.Rescaling(scale=output_scale)])
    
    out.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),loss=tf.keras.losses.MeanSquaredError())

    return out

class AbsThresholdLayer(tf.keras.layers.Layer):
    def __init__(self, thresholds):
      super(AbsThresholdLayer, self).__init__()
      self.thresholds = tf.constant(thresholds,dtype=tf.float32)
    
    def call(self, x):
        return x * tf.cast(tf.greater(tf.abs(x), self.thresholds),dtype=tf.float32)


def tfDenseThresholded(*sizes,thresholds=None,learning_rate=0.1,activations=None):

    if activations is None:
        activations = [None] * len(sizes)


    out = tf.keras.models.Sequential([ 
                                      tf.keras.layers.Dense(units=size,activation=activation)
                                    for size, activation in zip(sizes, activations)
                                 ] + [AbsThresholdLayer(thresholds)])
    
    out.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),loss=tf.keras.losses.MeanSquaredError())

    return out

def tfRNN(pod_size,rnn_sizes=[],rnn_activations=None,learning_rate=0.1,**kwargs):
    if rnn_activations is None:
        rnn_activations= [None] * len(rnn_sizes)
    
    
    out = tf.keras.models.Sequential([tf.keras.layers.SimpleRNN(units=size,activation=activation, return_sequences=True,**kwargs)
                                        for size, activation in zip(rnn_sizes[:-1], rnn_activations[:-1])] + [
                                      tf.keras.layers.SimpleRNN(units=rnn_sizes[-1],activation=rnn_activations[-1],**kwargs),
                                      tf.keras.layers.Dense(units=pod_size,activation='linear')])
    
    out.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),loss=tf.keras.losses.MeanSquaredError())

    return out

def tfLSTM(pod_size,rnn_sizes=[],rnn_activations=None,learning_rate=0.1,**kwargs):
    if rnn_activations is None:
        rnn_activations= [None] * len(rnn_sizes)
    
    
    out = tf.keras.models.Sequential([tf.keras.layers.LSTM(units=size,activation=activation, return_sequences=True,**kwargs)
                                        for size, activation in zip(rnn_sizes[:-1], rnn_activations[:-1])] + [
                                      tf.keras.layers.LSTM(units=rnn_sizes[-1],activation=rnn_activations[-1],**kwargs),
                                      tf.keras.layers.Dense(units=pod_size,activation='linear')])
    
    out.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),loss=tf.keras.losses.MeanSquaredError())

    return out