# Train Neural Network Material Model for Density-based Topology Optimization
# by Joel Najmon
# Python 3.9

# %% IMPORT PACKAGES
import numpy as np  # version 1.23.5
import scipy as sp  # version 1.10.1
import tensorflow as tf  # version 2.12.0
import matplotlib.pyplot as plt  # version 3.7.1
import matplotlib  # version 3.7.1
matplotlib.use('TkAgg')
import time

# %% DEFINE SIMP FUNCTION
ydim = 1  # y dimension
xdim = 1  # x dimension
x_lb = np.array([[0]])
x_ub = np.array([[1]])


def yx(x):
    return (1e-9 + (1 - 1e-9) * (x[:, 0] ** 3)).reshape(-1, 1)


def dyx(x):
    return (3 * (1 - 1e-9) * (x[:, 0] ** 2)).reshape(-1, 1, 1)


# %% GENERATE FEATURE SETS
N = int(1e4)  # total number of feature sets for training (70%), testing (15%), and validation (15%)
Nr = int(np.round(N * 0.85, decimals=0))  # number of training and validation feature sets
LHS = sp.stats.qmc.LatinHypercube(xdim)
x_train = LHS.random(Nr) * (x_ub - x_lb) + x_lb  # LHS random generation of feature sets
y_train = yx(x_train)

Nt = int(np.round(N * 0.15, decimals=0))  # number of testing feature sets
x_test = LHS.random(Nt) * (x_ub - x_lb) + x_lb  # LHS random generation of feature sets
y_test = yx(x_test)

# %% TRAIN NN
reps = int(1)  # number of NNs to train (for training more than 1 NN so that the best can be chosen from the lot)
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
    kernel_reg = None  # define kernel regularizer
    for i in range(NL):
        nn_model.add(tf.keras.layers.Dense(NN, hidden_activation, kernel_regularizer=kernel_reg))  # create hidden layers
    nn_model.add(tf.keras.layers.Dense(ydim, output_activation, kernel_regularizer=kernel_reg))  # create output layer
    norm_out_layer = tf.keras.layers.Normalization(axis=1, invert=False)
    norm_out_layer.adapt(y_train)
    nn_model.add(norm_out_layer)  # create de-normalization layer
    nn_model.compile(loss=loss, optimizer=optimizer, metrics=['mse'])  # compile NN

    # Train NN
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

nn_model.save('NN_model_DBTO_' + '{:.0e}'.format(N) + 'new_model')  # save model with 'new_model' flag so that it does
# not overwrite the original 'NN_model_DBTO_1e+04' models that were used in the paper.

# %% PLOT FUNCTIONS AND DERIVATIVES
# Generate uniform distribution of plotting points over xdim space
Np = 100
step_num = int(np.ceil(Np ** (1 / xdim)))
Np = step_num ** xdim
if xdim == 1:
    x_plot = np.transpose(np.mgrid[x_lb[0]:x_ub[0]:complex(0, step_num)].reshape(xdim, step_num ** xdim))
elif xdim == 2:
    x_plot = np.transpose(np.mgrid[x_lb[:, 0]:x_ub[:, 0]:complex(0, step_num),
                          x_lb[:, 1]:x_ub[:, 1]:complex(0, step_num)].reshape(xdim, step_num ** xdim))
elif xdim == 3:
    x_plot = np.transpose(np.mgrid[x_lb[:, 0]:x_ub[:, 0]:complex(0, step_num),
                          x_lb[:, 1]:x_ub[:, 1]:complex(0, step_num),
                          x_lb[:, 2]:x_ub[:, 2]:complex(0, step_num)].reshape(xdim, step_num ** xdim))

# Evaluate function and derivative at plotting points (via automatic differentiation)
y_plot = yx(x_plot)  # true function
dy_plot = dyx(x_plot)  # derivative of true function
y_nn = nn_model.predict(x_plot).reshape(Np, ydim)  # NN prediction of function
xn_tape = tf.Variable(x_plot, dtype=tf.float64)
with tf.GradientTape(persistent=True) as tape:  # NN prediction of the derivative of true function via AD
    yn_tape = xn_tape
    for layer in nn_model.layers:
        yn_tape = layer(yn_tape)
dy_nn = tf.reduce_sum(tape.jacobian(yn_tape, xn_tape), axis=[2]).numpy().reshape(Np, ydim, xdim)

# Plot Function
fig = plt.figure(1)
plt.plot(x_plot, y_plot, 'g', label='Ground-truth')
plt.plot(x_plot, y_nn, 'b', label='MLP prediction')
plt.xlabel('theta_e')
plt.ylabel('E(theta_e)')
plt.title('SIMP function; Eq. (14)')
plt.legend()

# Plot Derivatives
fig = plt.figure(2)
plt.plot(x_plot, dy_plot.reshape(Np, 1), 'g', label='Ground-truth derivative')
plt.plot(x_plot, dy_nn.reshape(Np, 1), 'c', label='Neural network-based derivative')
plt.xlabel('theta_e')
plt.ylabel('dE(theta_e)/dtheta_e')
plt.title('Derivative of SIMP function; Eq. (15)')
plt.legend()

# Show Figures
plt.show()