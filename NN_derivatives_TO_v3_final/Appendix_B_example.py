# Appendix B: Example of Analytical Derivative Calculations through a Neural Network
# by Joel Najmon
# Python 3.9

# %% IMPORT PACKAGES
import numpy as np  # version 1.23.5
from scipy import stats as stats  # version 1.10.1
import tensorflow as tf  # version 2.12.0
import matplotlib.pyplot as plt  # version 3.7.1
import matplotlib  # version 3.7.1
matplotlib.use('TkAgg')
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

# 1) Neural Network Jacobian
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
    dy_1[s, :, :] = dy_product.T.reshape(1, ydim, xdim)  # derivative via NNJ
t_1 = time.time() - t  # prediction time

# %% ERROR CALCULATIONS
# Mean Squared Error of NN
mse = tf.metrics.mean_squared_error(y_true=y_0.squeeze(), y_pred=y_1.squeeze()).numpy()

# Mean Squared Errors of the NN-derived Derivative Methods
dy_1_er = np.float64(tf.metrics.mean_squared_error(y_true=dy_0.reshape(1, -1), y_pred=dy_1.reshape(1, -1)).numpy())

print(' ')
print('NEURAL NETWORK PERFORMANCE')
print('Mean Squared Error: ', mse)
print('Training time:      ', t_nn)
print(' ')
print('NEURAL NETWORK-DERIVED DERIVATIVES\'')
print('Analytical Derivative of NN:')
print('Error:           ', dy_1_er)
print('Prediction time: ', t_1)

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

# Evaluate function and derivative at plotting points (via automatic differentiation)
# Plot Function
fig = plt.figure(1)
plt.plot(x_plot, y_plot, 'g', label='Ground-truth')
plt.plot(x_plot, y_nn, 'b', label='NN prediction')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Example Function')
plt.legend()

# Plot Derivatives
fig = plt.figure(2)
plt.plot(x_plot, dy_plot.reshape(Np, 1), 'g', label='Ground-truth derivative')
plt.plot(x_plot, dy_nn.reshape(Np, 1), 'c', label='Neural network-based derivative')
plt.xlabel('x')
plt.ylabel('dy/dx')
plt.title('Derivative of Example function')
plt.legend()

# Show Figures
plt.show()