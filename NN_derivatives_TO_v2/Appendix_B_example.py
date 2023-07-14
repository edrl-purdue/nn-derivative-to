# NN Derivatives
# by Joel C. Najmon
# Python 3.9
# IMPORT PACKAGES
import numpy as np
from scipy import stats as stats
import matplotlib.pyplot as plt
import tensorflow as tf
import time

# %% Appendix B Example Function
ydim = 1  # y dimension
xdim = 1  # x dimension
x_lb = np.array([[-1]])
x_ub = np.array([[1]])


def yx(x):
    return (x[:, 0] ** 2).reshape(-1, 1)


def dyx(x):
    return (2 * x[:, 0]).reshape(-1, 1, 1)

# %% GENERATE FEATURE SETS
N = int(1e5)  # total number of feature sets for training (70%), testing (15%), and validation (15%)
Nr = int(np.round(N * 0.85, decimals=0))  # number of training and validation feature sets
tf.keras.utils.set_random_seed(10)  # make deterministic
LHS = stats.qmc.LatinHypercube(xdim, seed=10)
x_train = LHS.random(Nr) * (x_ub - x_lb) + x_lb  # LHS random generation of feature sets
y_train = yx(x_train)

Nt = int(np.round(N * 0.15, decimals=0))  # number of testing feature sets
x_test = LHS.random(Nt) * (x_ub - x_lb) + x_lb  # LHS random generation of feature sets

# %% TRAIN NN
NL = int(2)  # number of hidden layers
NN = int(3)  # number of neurons per hidden layer
hidden_activation = 'sigmoid'  # activation function of hidden layer
output_activation = 'linear'  # activation function of output layer

# Define NN
tf.keras.backend.set_floatx(tf.float64.name)  # set precision
nn_model = tf.keras.models.Sequential()  # create NN that has sequential layers
nn_model.add(tf.keras.Input(shape=(xdim,)))  # create input layer
loss = tf.keras.losses.MeanSquaredError()  # set loss function to mean squared error
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)  # set optimizer

# Compile NN
kernel_reg = None  # define kernel regularizer
for i in range(NL):
    nn_model.add(tf.keras.layers.Dense(NN, hidden_activation, kernel_regularizer=kernel_reg))  # create hidden layers
nn_model.add(tf.keras.layers.Dense(ydim, output_activation, kernel_regularizer=kernel_reg))  # create output layer
nn_model.compile(loss=loss, optimizer=optimizer, metrics=['mse'])  # compile NN

# Train NN
stop_early = tf.keras.callbacks.EarlyStopping(monitor='mse', patience=5)
t = time.time()
nn_model.fit(x_train, y_train, batch_size=32, epochs=500, verbose=1, validation_split=(0.15 / 0.85))
t_nn = time.time() - t  # training time


# %% DEFINE ACTIVATION FUNCTIONS AND THEIR DERIVATIVES
def act_fn(yn_1, wn, bn, fn):  # activation function yn
    yn = np.matmul(yn_1, wn) + bn
    if fn == 'sigmoid':
        return 1 / (1 + np.exp(-yn))
    if fn == 'linear':
        return yn
    if fn == 'relu':
        return np.maximum(yn, 0)


def dact_fn(yn_1, wn, bn, fn):  # derivative of activation function yn with respect to yn_1
    yn = np.matmul(yn_1, wn) + bn
    if fn == 'relu':
        return np.multiply(wn, np.divide(np.maximum(yn, 0), yn))
    if fn == 'linear':
        return wn
    if fn == 'sigmoid':
        return np.multiply(wn, np.exp(yn)) / ((1 + np.exp(yn)) ** 2)


# %% METHODS FOR y(x)
# 1) True Function
y_0 = yx(x_test)

# 2) Manual NN Prediction
# y_1 calculated below in Method 1 for dy(x)/dx

# 2) NN Prediction
y_2 = nn_model.predict(x_test).reshape(Nt, ydim)

# %% METHODS FOR dy(x)/dx
# 0) Analytical Derivative of true function
dy_0 = dyx(x_test)

# 1) Analytical Derivative of NN
t = time.time()
Wn = []
Bn = []
An = []
for L in np.arange(0, NL + 1):
    Wn.append(nn_model.layers[L].weights[0].numpy())  # store hidden layer weights
    Bn.append(nn_model.layers[L].bias.numpy().reshape(1, -1))  # store hidden layer biases
    An.append(nn_model.layers[L].activation.__name__)  # store hidden layer activation function names

y_1 = np.zeros((Nt, ydim))  # Manual NN prediction
dy_1 = np.zeros((Nt, ydim, xdim))
for s in np.arange(0, Nt):  # loop through test feature sets
    Y0 = (x_test[s, :]).reshape(xdim, 1)  # input
    Yn = [act_fn(Y0.T, Wn[0], Bn[0], An[0])]  # initialize 1st layer output
    dYn = [dact_fn(Y0.T, Wn[0], Bn[0], An[0])]  # initialize 1st layer output's derivative
    dy_product = dYn[0]

    for L in np.arange(1, NL + 1):  # manually loop through layers to calculate analytical derivative with chain rule
        Yn.append(act_fn(Yn[L - 1], Wn[L], Bn[L], An[L]))  # hidden layer L output
        dYn.append(dact_fn(Yn[L - 1], Wn[L], Bn[L], An[L]))  # derivative of hidden layer L output
        dy_product = np.matmul(dy_product, dYn[L])  # derivative of hidden layer L output with respect NN input
    y_1[s, :] = Yn[-1]  # manual NN prediction
    dy_1[s, :, :] = dy_product.T.reshape(1, ydim, xdim)  # derivative via Analytical Derivative of NN
t_1 = time.time() - t  # prediction time

# 2) Central Finite Difference Approximation
t = time.time()
h_cfd = 1e-6  # step size
dy_2 = np.zeros((Nt, ydim, xdim))
for d in np.arange(0, xdim):
    h_mat = np.zeros((Nt, xdim))
    h_mat[:, [d]] = np.ones((Nt, 1)) * h_cfd  # perturbation array
    dy_2[:, :, d] = ((nn_model.predict(x_test + h_mat) - nn_model.predict(x_test - h_mat))
                     / (2 * h_cfd)).reshape(Nt, ydim)  # derivative via CFDA
t_2 = time.time() - t  # prediction time

# 3) Complex Step Derivative Approximation
t = time.time()
h_csm = 1e-12  # step size
dy_3 = np.zeros((Nt, ydim, xdim))
for d in np.arange(0, xdim):  # loop through xdim for partial derivatives
    h_mat = 0j * np.zeros((1, xdim), dtype='complex')
    h_mat[0, d] = 1j * h_csm  # perturbation array

    y_csm = np.zeros((Nt, ydim), dtype='complex')
    for s in np.arange(0, Nt):  # loop through test feature sets
        Y0 = (x_test[s, :] + h_mat).reshape(xdim, 1)  # perturbed input
        Yn_csm = [act_fn(Y0.T, Wn[0], Bn[0], An[0])]  # initialize 1st layer output

        for L in np.arange(1, NL + 1):  # manual NN prediction so imaginary numbers can be passed
            Yn_csm.append(act_fn(Yn_csm[L - 1], Wn[L], Bn[L], An[L]))  # hidden layer L output
        y_csm[s, :] = Yn_csm[-1]  # manual NN prediction
    dy_3[:, :, d] = (np.imag(y_csm) / h_csm)  # derivative via CSDA
t_3 = time.time() - t  # prediction time

# 4) Automatic Differentiation
t = time.time()
xn_tape = tf.Variable(x_test, dtype=tf.float64)
with tf.GradientTape(persistent=True) as tape:
    yn_tape = xn_tape
    for layer in nn_model.layers:  # loop through layers
        yn_tape = layer(yn_tape)
dy_4 = tf.reduce_sum(tape.jacobian(yn_tape, xn_tape), axis=[2]).numpy().reshape(Nt, ydim, xdim)  # derivative via AD
t_4 = time.time() - t  # prediction time

# %% ERROR CALCULATIONS
# Mean Squared Error of NN
mse = tf.metrics.mean_squared_error(y_true=y_0.squeeze(), y_pred=y_1.squeeze()).numpy()

# Mean Squared Errors of the NN-derived Derivative Methods
dy_1_er = np.float64(tf.metrics.mean_squared_error(y_true=dy_0.reshape(1, -1), y_pred=dy_1.reshape(1, -1)).numpy())
dy_2_er = np.float64(tf.metrics.mean_squared_error(y_true=dy_0.reshape(1, -1), y_pred=dy_2.reshape(1, -1)).numpy())
dy_3_er = np.float64(tf.metrics.mean_squared_error(y_true=dy_0.reshape(1, -1), y_pred=dy_3.reshape(1, -1)).numpy())
dy_4_er = np.float64(tf.metrics.mean_squared_error(y_true=dy_0.reshape(1, -1), y_pred=dy_4.reshape(1, -1)).numpy())

print(' ')
print('NEURAL NETWORK PERFORMANCE')
print('Mean Squared Error: ', mse)
print('Training time:      ', t_nn)
print(' ')
print('NEURAL NETWORK-DERIVED DERIVATIVES\'')
print('Analytical Derivative of NN:')
print('Error:           ', dy_1_er)
print('Prediction time: ', t_1)
print(' ')
print('Central Finite Difference Approximation:')
print('Error:           ', dy_2_er)
print('Prediction time: ', t_2)
print(' ')
print('Complex Step Derivative Approximation:')
print('Error:           ', dy_3_er)
print('Prediction time: ', t_3)
print(' ')
print('Automatic Differentiation:')
print('Error:           ', dy_4_er)
print('Prediction time: ', t_4)

# %% PLOT FUNCTIONS AND DERIVATIVES
if ydim == 1:
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

    #  Evaluate function and derivative at plotting points
    y_plot = yx(x_plot)  # true function
    dy_plot = dyx(x_plot)  # derivative of true function
    y_nn = nn_model.predict(x_plot).reshape(Np, ydim)  # NN prediction of function
    xn_tape = tf.Variable(x_plot, dtype=tf.float64)
    with tf.GradientTape(persistent=True) as tape:  # NN prediction of the derivative of true function via AD
        yn_tape = xn_tape
        for layer in nn_model.layers:
            yn_tape = layer(yn_tape)
    dy_nn = tf.reduce_sum(tape.jacobian(yn_tape, xn_tape), axis=[2]).numpy().reshape(Np, ydim, xdim)

    #  Plot Functions and Derivatives (WILL ADD SUBPLOTS AND TITLES LATER. MAYBE ALSO SUPPORT FOR MORE PLOT DIMENSIONS)
    if xdim == 1:
        # Plot Function
        fig = plt.figure()
        plt.plot(x_plot, y_plot, 'g')
        plt.plot(x_plot, y_nn, 'b')
        plt.show()

        # Plot Derivatives
        fig = plt.figure()
        plt.plot(x_plot, dy_plot.reshape(Np, 1), 'g')
        plt.plot(x_plot, dy_nn.reshape(Np, 1), 'c')
        plt.show()
    elif xdim == 2:
        # Plot True Function
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x_plot[:, 0].reshape(step_num, step_num),
                        x_plot[:, 1].reshape(step_num, step_num),
                        y_plot.reshape(step_num, step_num))
        plt.show()

        # Plot Predicted Function
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x_plot[:, 0].reshape(step_num, step_num),
                        x_plot[:, 1].reshape(step_num, step_num),
                        y_nn.reshape(step_num, step_num))
        plt.show()
