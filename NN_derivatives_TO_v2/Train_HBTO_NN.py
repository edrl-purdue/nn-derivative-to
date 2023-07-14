# by Joel C. Najmon
# Python 3.9
# IMPORT PACKAGES
import numpy as np  # version 1.24.3
import scipy as sp
import tensorflow as tf  # version 2.12.0
import time

# %% LOAD FEATURE SETS
# mat = sp.io.loadmat('Hom_based_CH_2D_N100_nel100.mat', squeeze_me=True)
# mat = sp.io.loadmat('Hom_based_CH_2D_N1000_nel100.mat', squeeze_me=True)
mat = sp.io.loadmat('Hom_based_CH_2D_N10000_nel100.mat', squeeze_me=True)
# mat = sp.io.loadmat('Hom_based_CH_2D_N100000_nel100.mat', squeeze_me=True)
# mat = sp.io.loadmat('Hom_based_CH_2D_N1000000_nel100.mat', squeeze_me=True)

SP_para = mat['SP_para']
SP_C = mat['SP_C']
SP_C_rot = mat['SP_C_rot']

X = SP_para[:, 0:2]
Y = SP_C[:, [0, 1, 2, 5]]
# X = SP_para
# Y = SP_C_rot
N = SP_para.shape[0]  # total number of feature sets for training (70%), testing (15%), and validation (15%)
ydim = Y.shape[1]  # y dimension
xdim = X.shape[1]  # x dimension

x_train = X[0:int(np.round(N * 0.85, decimals=0)), :]  # train (70%) and validation (15%) sets (85%)
y_train = Y[0:int(np.round(N * 0.85, decimals=0)), :]  # train (70%) and validation (15%) sets (85%)

x_test = X[int(np.round(N * 0.85, decimals=0)):N, :]  # test set (15%)
y_test = Y[int(np.round(N * 0.85, decimals=0)):N, :]  # test set (15%)

# %% TRAIN NN
reps = int(1)  # number of NNs to train
NL = int(1)  # number of hidden layers
NN = int(64)  # number of neurons per hidden layer
hidden_activation = 'sigmoid'  # activation function of hidden layer
output_activation = 'linear'  # activation function of output layer

models = []
yann = []
mse = []
t = time.time()
for n in range(reps):
    # Define NN
    tf.keras.backend.set_floatx(tf.float64.name)  # set precision
    nn_model = tf.keras.models.Sequential()  # create NN that has sequential layers
    nn_model.add(tf.keras.Input(shape=(xdim,)))  # create input layer
    norm_in_layer = tf.keras.layers.Normalization(axis=1)
    norm_in_layer.adapt(x_train)
    nn_model.add(norm_in_layer)  # create normalization layer
    loss = tf.keras.losses.MeanSquaredError()  # set loss function to mean squared error
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)  # set optimizer

    # Compile NN
    # kernel_reg = tf.keras.regularizers.l2(1e-2)  # define kernel regularizer
    kernel_reg = None  # define kernel regularizer
    for i in range(NL):
        nn_model.add(tf.keras.layers.Dense(NN, hidden_activation, kernel_regularizer=kernel_reg))  # create hidden layers
    nn_model.add(tf.keras.layers.Dense(ydim, output_activation, kernel_regularizer=kernel_reg))  # create output layer
    norm_out_layer = tf.keras.layers.Normalization(axis=1, invert=False)
    norm_out_layer.adapt(y_train)
    nn_model.add(norm_out_layer)  # create de-normalization layer
    nn_model.compile(loss=loss, optimizer=optimizer, metrics=['mse'])  # compile NN

    # Train NN
    # nn_model.fit(x_train, y_train, batch_size=256, epochs=1000, verbose=1)
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='mse', patience=5)
    nn_model.fit(x_train, y_train, batch_size=32, epochs=5000, verbose=1, validation_split=(0.15 / 0.85))
    models.append(nn_model)

    # Predict Test Dataset
    yann.append(nn_model.predict(x_test).reshape(int(np.round(N * 0.15, decimals=0)), ydim))

    # Calculate Performance of NN n on Test Dataset
    mse.append(tf.metrics.mean_squared_error(y_true=y_test.flatten(), y_pred=yann[n].flatten()).numpy())

# %% SAVE BEST PERFORMING NN
nnind = mse.index(min(mse))
nn_model = models[nnind]

print(' ')
print('NEURAL NETWORK PERFORMANCE')
print('MSE: ', mse[nnind])
print('Time:', time.time() - t, 'seconds')

# nn_model.save('ANN_HB_' + '{:.0e}'.format(N))