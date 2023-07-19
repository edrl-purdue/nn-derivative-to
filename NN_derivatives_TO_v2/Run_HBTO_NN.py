# Perform Homogenization-based Topology Optimization with a Neural network material model
# by Joel Najmon and Andres Tovar
# Python 3.9

# %% IMPORT PACKAGES
import numpy as np  # version 1.23.5
import numpy.matlib as npm
import scipy as sp  # version 1.10.1
from scipy.sparse import csc_matrix, coo_matrix
from scipy.sparse.linalg import spsolve, norm
import tensorflow as tf  # version 2.12.0
import matplotlib.pyplot as plt  # version 3.7.1
import matplotlib  # version 3.7.1
matplotlib.use('TkAgg')
from math import ceil


# Element stiffness matrix for mechanical analysis
def kefun_C(C):
    C1_1 = C[0, 0]
    C2_1 = C[1, 0]
    C2_2 = C[1, 1]
    C3_3 = C[2, 2]
    KE = np.zeros((8, 8))
    KE[0, 0] = (216345702438951137307716154342489*C1_1)/649037107316853453566312041152512 + (216345702438951137307716154342489*C3_3)/649037107316853453566312041152512
    KE[1, 0] = (324518553658426690754359001612289*C2_1)/1298074214633706907132624082305024 + (324518553658426690754359001612289*C3_3)/1298074214633706907132624082305024
    KE[2, 0] = (13521606402434444180830355908725*C3_3)/81129638414606681695789005144064 - (216345702438951137307716154342489*C1_1)/649037107316853453566312041152512
    KE[3, 0] = (324518553658426690754359001612289*C2_1)/1298074214633706907132624082305024 - (324518553658426690754359001612289*C3_3)/1298074214633706907132624082305024
    KE[4, 0] = - (13521606402434444180830355908725*C1_1)/81129638414606681695789005144064 - (13521606402434444180830355908725*C3_3)/81129638414606681695789005144064
    KE[5, 0] = - (324518553658426690754359001612289*C2_1)/1298074214633706907132624082305024 - (324518553658426690754359001612289*C3_3)/1298074214633706907132624082305024
    KE[6, 0] = (13521606402434444180830355908725*C1_1)/81129638414606681695789005144064 - (216345702438951137307716154342489*C3_3)/649037107316853453566312041152512
    KE[7, 0] = (324518553658426690754359001612289*C3_3)/1298074214633706907132624082305024 - (324518553658426690754359001612289*C2_1)/1298074214633706907132624082305024
    KE[0, 1] = (324518553658426690754359001612289*C2_1)/1298074214633706907132624082305024 + (324518553658426690754359001612289*C3_3)/1298074214633706907132624082305024
    KE[1, 1] = (216345702438951137307716154342489*C2_2)/649037107316853453566312041152512 + (216345702438951137307716154342489*C3_3)/649037107316853453566312041152512
    KE[2, 1] = (324518553658426690754359001612289*C3_3)/1298074214633706907132624082305024 - (324518553658426690754359001612289*C2_1)/1298074214633706907132624082305024
    KE[3, 1] = (13521606402434444180830355908725*C2_2)/81129638414606681695789005144064 - (216345702438951137307716154342489*C3_3)/649037107316853453566312041152512
    KE[4, 1] = - (324518553658426690754359001612289*C2_1)/1298074214633706907132624082305024 - (324518553658426690754359001612289*C3_3)/1298074214633706907132624082305024
    KE[5, 1] = - (13521606402434444180830355908725*C2_2)/81129638414606681695789005144064 - (13521606402434444180830355908725*C3_3)/81129638414606681695789005144064
    KE[6, 1] = (324518553658426690754359001612289*C2_1)/1298074214633706907132624082305024 - (324518553658426690754359001612289*C3_3)/1298074214633706907132624082305024
    KE[7, 1] = (13521606402434444180830355908725*C3_3)/81129638414606681695789005144064 - (216345702438951137307716154342489*C2_2)/649037107316853453566312041152512
    KE[0, 2] = (13521606402434444180830355908725*C3_3)/81129638414606681695789005144064 - (216345702438951137307716154342489*C1_1)/649037107316853453566312041152512
    KE[1, 2] = (324518553658426690754359001612289*C3_3)/1298074214633706907132624082305024 - (324518553658426690754359001612289*C2_1)/1298074214633706907132624082305024
    KE[2, 2] = (216345702438951137307716154342489*C1_1)/649037107316853453566312041152512 + (216345702438951137307716154342489*C3_3)/649037107316853453566312041152512
    KE[3, 2] = - (324518553658426690754359001612289*C2_1)/1298074214633706907132624082305024 - (324518553658426690754359001612289*C3_3)/1298074214633706907132624082305024
    KE[4, 2] = (13521606402434444180830355908725*C1_1)/81129638414606681695789005144064 - (216345702438951137307716154342489*C3_3)/649037107316853453566312041152512
    KE[5, 2] = (324518553658426690754359001612289*C2_1)/1298074214633706907132624082305024 - (324518553658426690754359001612289*C3_3)/1298074214633706907132624082305024
    KE[6, 2] = - (13521606402434444180830355908725*C1_1)/81129638414606681695789005144064 - (13521606402434444180830355908725*C3_3)/81129638414606681695789005144064
    KE[7, 2] = (324518553658426690754359001612289*C2_1)/1298074214633706907132624082305024 + (324518553658426690754359001612289*C3_3)/1298074214633706907132624082305024
    KE[0, 3] = (324518553658426690754359001612289*C2_1)/1298074214633706907132624082305024 - (324518553658426690754359001612289*C3_3)/1298074214633706907132624082305024
    KE[1, 3] = (13521606402434444180830355908725*C2_2)/81129638414606681695789005144064 - (216345702438951137307716154342489*C3_3)/649037107316853453566312041152512
    KE[2, 3] = - (324518553658426690754359001612289*C2_1)/1298074214633706907132624082305024 - (324518553658426690754359001612289*C3_3)/1298074214633706907132624082305024
    KE[3, 3] = (216345702438951137307716154342489*C2_2)/649037107316853453566312041152512 + (216345702438951137307716154342489*C3_3)/649037107316853453566312041152512
    KE[4, 3] = (324518553658426690754359001612289*C3_3)/1298074214633706907132624082305024 - (324518553658426690754359001612289*C2_1)/1298074214633706907132624082305024
    KE[5, 3] = (13521606402434444180830355908725*C3_3)/81129638414606681695789005144064 - (216345702438951137307716154342489*C2_2)/649037107316853453566312041152512
    KE[6, 3] = (324518553658426690754359001612289*C2_1)/1298074214633706907132624082305024 + (324518553658426690754359001612289*C3_3)/1298074214633706907132624082305024
    KE[7, 3] = - (13521606402434444180830355908725*C2_2)/81129638414606681695789005144064 - (13521606402434444180830355908725*C3_3)/81129638414606681695789005144064
    KE[0, 4] = - (13521606402434444180830355908725*C1_1)/81129638414606681695789005144064 - (13521606402434444180830355908725*C3_3)/81129638414606681695789005144064
    KE[1, 4] = - (324518553658426690754359001612289*C2_1)/1298074214633706907132624082305024 - (324518553658426690754359001612289*C3_3)/1298074214633706907132624082305024
    KE[2, 4] = (13521606402434444180830355908725*C1_1)/81129638414606681695789005144064 - (216345702438951137307716154342489*C3_3)/649037107316853453566312041152512
    KE[3, 4] = (324518553658426690754359001612289*C3_3)/1298074214633706907132624082305024 - (324518553658426690754359001612289*C2_1)/1298074214633706907132624082305024
    KE[4, 4] = (216345702438951137307716154342489*C1_1)/649037107316853453566312041152512 + (216345702438951137307716154342489*C3_3)/649037107316853453566312041152512
    KE[5, 4] = (324518553658426690754359001612289*C2_1)/1298074214633706907132624082305024 + (324518553658426690754359001612289*C3_3)/1298074214633706907132624082305024
    KE[6, 4] = (13521606402434444180830355908725*C3_3)/81129638414606681695789005144064 - (216345702438951137307716154342489*C1_1)/649037107316853453566312041152512
    KE[7, 4] = (324518553658426690754359001612289*C2_1)/1298074214633706907132624082305024 - (324518553658426690754359001612289*C3_3)/1298074214633706907132624082305024
    KE[0, 5] = - (324518553658426690754359001612289*C2_1)/1298074214633706907132624082305024 - (324518553658426690754359001612289*C3_3)/1298074214633706907132624082305024
    KE[1, 5] = - (13521606402434444180830355908725*C2_2)/81129638414606681695789005144064 - (13521606402434444180830355908725*C3_3)/81129638414606681695789005144064
    KE[2, 5] = (324518553658426690754359001612289*C2_1)/1298074214633706907132624082305024 - (324518553658426690754359001612289*C3_3)/1298074214633706907132624082305024
    KE[3, 5] = (13521606402434444180830355908725*C3_3)/81129638414606681695789005144064 - (216345702438951137307716154342489*C2_2)/649037107316853453566312041152512
    KE[4, 5] = (324518553658426690754359001612289*C2_1)/1298074214633706907132624082305024 + (324518553658426690754359001612289*C3_3)/1298074214633706907132624082305024
    KE[5, 5] = (216345702438951137307716154342489*C2_2)/649037107316853453566312041152512 + (216345702438951137307716154342489*C3_3)/649037107316853453566312041152512
    KE[6, 5] = (324518553658426690754359001612289*C3_3)/1298074214633706907132624082305024 - (324518553658426690754359001612289*C2_1)/1298074214633706907132624082305024
    KE[7, 5] = (13521606402434444180830355908725*C2_2)/81129638414606681695789005144064 - (216345702438951137307716154342489*C3_3)/649037107316853453566312041152512
    KE[0, 6] = (13521606402434444180830355908725*C1_1)/81129638414606681695789005144064 - (216345702438951137307716154342489*C3_3)/649037107316853453566312041152512
    KE[1, 6] = (324518553658426690754359001612289*C2_1)/1298074214633706907132624082305024 - (324518553658426690754359001612289*C3_3)/1298074214633706907132624082305024
    KE[2, 6] = - (13521606402434444180830355908725*C1_1)/81129638414606681695789005144064 - (13521606402434444180830355908725*C3_3)/81129638414606681695789005144064
    KE[3, 6] = (324518553658426690754359001612289*C2_1)/1298074214633706907132624082305024 + (324518553658426690754359001612289*C3_3)/1298074214633706907132624082305024
    KE[4, 6] = (13521606402434444180830355908725*C3_3)/81129638414606681695789005144064 - (216345702438951137307716154342489*C1_1)/649037107316853453566312041152512
    KE[5, 6] = (324518553658426690754359001612289*C3_3)/1298074214633706907132624082305024 - (324518553658426690754359001612289*C2_1)/1298074214633706907132624082305024
    KE[6, 6] = (216345702438951137307716154342489*C1_1)/649037107316853453566312041152512 + (216345702438951137307716154342489*C3_3)/649037107316853453566312041152512
    KE[7, 6] = - (324518553658426690754359001612289*C2_1)/1298074214633706907132624082305024 - (324518553658426690754359001612289*C3_3)/1298074214633706907132624082305024
    KE[0, 7] = (324518553658426690754359001612289*C3_3)/1298074214633706907132624082305024 - (324518553658426690754359001612289*C2_1)/1298074214633706907132624082305024
    KE[1, 7] = (13521606402434444180830355908725*C3_3)/81129638414606681695789005144064 - (216345702438951137307716154342489*C2_2)/649037107316853453566312041152512
    KE[2, 7] = (324518553658426690754359001612289*C2_1)/1298074214633706907132624082305024 + (324518553658426690754359001612289*C3_3)/1298074214633706907132624082305024
    KE[3, 7] = - (13521606402434444180830355908725*C2_2)/81129638414606681695789005144064 - (13521606402434444180830355908725*C3_3)/81129638414606681695789005144064
    KE[4, 7] = (324518553658426690754359001612289*C2_1)/1298074214633706907132624082305024 - (324518553658426690754359001612289*C3_3)/1298074214633706907132624082305024
    KE[5, 7] = (13521606402434444180830355908725*C2_2)/81129638414606681695789005144064 - (216345702438951137307716154342489*C3_3)/649037107316853453566312041152512
    KE[6, 7] = - (324518553658426690754359001612289*C2_1)/1298074214633706907132624082305024 - (324518553658426690754359001612289*C3_3)/1298074214633706907132624082305024
    KE[7, 7] = (216345702438951137307716154342489*C2_2)/649037107316853453566312041152512 + (216345702438951137307716154342489*C3_3)/649037107316853453566312041152512
    return KE


# Preparation for mechanical analysis
def feaprep(model):
    # Preparing edofMat: element dof matrix
    nodenrs = np.arange(0, (model.nelx + 1) * (model.nely + 1), 1).reshape(model.nely + 1, model.nelx + 1, order='F') + 1
    edofVec = np.asarray(2 * nodenrs[0:-1, 0:-1] + 1).reshape(-1, 1, order='F')
    Cvec = np.array([2, 3, 0, 1])
    Cmat = np.block([[0, 1, 2 * model.nely + Cvec, -2, -1]])
    edofMat = npm.repmat(edofVec, 1, 8) + npm.repmat(Cmat, model.nelx * model.nely, 1)
    edofMat = edofMat - 1
    # Preparing iK and jK
    iKmat = np.kron(edofMat, np.ones((8, 1)))
    jKmat = np.kron(edofMat, np.ones((1, 8)))
    iK = np.reshape(iKmat.transpose(), (-1, 1), order='F')
    jK = np.reshape(jKmat.transpose(), (-1, 1), order='F')
    return edofMat, iK, jK


# Filter preparation
def filterprep(model):
    rminc = ceil(model.rmin)
    iH = np.ones((model.nelx * model.nely * (2 * (rminc - 1) + 1) ** 2, 1))
    jH = np.ones(iH.shape)
    sH = np.zeros(iH.shape)
    k = -1
    for i1 in np.arange(1, model.nelx + 1):
        for j1 in np.arange(1, model.nely + 1):
            e1 = (i1 - 1) * model.nely + j1
            for i2 in np.arange(np.amax([i1 - rminc + 1, 1]), np.amin([i1 + rminc - 1, model.nelx]) + 1):
                for j2 in np.arange(np.amax([j1 - rminc + 1, 1]), np.amin([j1 + rminc - 1, model.nely]) + 1):
                    e2 = (i2 - 1) * model.nely + j2
                    k = k + 1
                    iH[k] = e1
                    jH[k] = e2
                    sH[k] = np.amax([0, model.rmin - np.sqrt((i1 - i2) ** 2 + (j1 - j2) ** 2)])
    # Hs matrix
    iHint = iH.astype(int).flatten() - 1
    jHint = jH.astype(int).flatten() - 1
    sHshp = sH.flatten()
    H = coo_matrix((sHshp, (iHint, jHint)))
    H = csc_matrix(H)
    Hs = H.sum(axis=1)
    return H, Hs


# Filter function
def filterfun(x, dc, dv, H, Hs, ft):
    nelx = x.shape[1]
    nely = x.shape[0]
    dcf = dc
    dvf = dv
    if ft == 1:  # sensitivity filter
        xdc = x * dc
        xdc = xdc.reshape(-1, 1, order='F')
        tmat = npm.repmat(1e-3, nely, nelx)
        tmax = np.maximum(tmat, x).reshape(-1, 1, order='F')
        dcf = H * xdc / Hs / tmax
        dcf = dcf.reshape(nely, nelx, order='F')
        dvf = dv
    elif ft == 2:  # density filter
        xdc = dc.reshape(-1, 1, order='F')
        xdv = dv.reshape(-1, 1, order='F')
        dcf = H * (xdc / Hs)
        dcf = dcf.reshape(nely, nelx, order='F')
        dvf = H * (xdv / Hs)
        dvf = dvf.reshape(nely, nelx, order='F')
    return dcf, dvf


# Finite element analysis
def feafun(xPhys, model):
    xPhys = xPhys.reshape(-1, 2, order='F')

    if model.ft == 1:
        xPhys = xPhys
    elif model.ft == 2:
        xPhys[:, 0] = np.asarray(model.H * xPhys[:, 0].reshape(-1, 1, order='F') / model.Hs).flatten('F')
        xPhys[:, 1] = np.asarray(model.H * xPhys[:, 1].reshape(-1, 1, order='F') / model.Hs).flatten('F')

    # sK vector
    Cmax = np.array([[1.3462, 0.5769, 0], [0.5769, 1.3462, 0], [0, 0, 0.3846]])  # minimum stiffness tensor
    Cmin = np.array([[1.3462, 0.5769, 0], [0.5769, 1.3462, 0], [0, 0, 0.3846]]) * 1e-9  # maximum stiffness tensor
    x_test = xPhys
    C_pred = model.nn_model.predict(x_test, verbose=0)  # predict stiffness tensor

    # FEA
    sK_mat = np.zeros((64, xPhys.shape[0]))
    for xx in np.arange(0, xPhys.shape[0]):
        C = np.array([[C_pred[xx, 0], C_pred[xx, 1], 0], [C_pred[xx, 1], C_pred[xx, 2], 0], [0, 0, C_pred[xx, 3]]])
        C = np.minimum(np.maximum(C, Cmin), Cmax)
        KEvec = kefun_C(C).reshape(-1, 1, order='F')
        sK_mat[:, xx] = KEvec.squeeze()
    sK_vec = sK_mat.reshape(-1, 1, order='F')

    # Stiffness matrix
    iKint = model.iK.astype(int).flatten()
    jKint = model.jK.astype(int).flatten()
    sKshp = sK_vec.flatten()
    K = coo_matrix((sKshp, (iKint, jKint)))
    K = 0.5 * (K.transpose() + K)

    # Solve finite element equation
    U = np.zeros((model.dofs, 1))
    K = csc_matrix(K)
    F = csc_matrix(model.F)
    U[model.freedofs, :] = spsolve(K[model.freedofs, model.freedofs.reshape(-1, 1, order='F')],
                             F[model.freedofs, :]).reshape(-1, 1, order='F')
    U[model.fixeddofs, :] = 0

    # Objective function
    ceU = U[model.edofMat][:, :, 0]  # Element compliance matrix
    ceUKU = np.zeros(xPhys.shape[0])
    for xx in np.arange(0, xPhys.shape[0]):
        KE = sK_mat[:, xx].reshape(8, 8, order='F')
        ceUKU[xx] = np.matmul(np.matmul(ceU[xx, :], KE), ceU[xx, :])

    return sum(ceUKU)


# Finite element analysis with sensitivity coefficients
def dfeafun(xPhys, model):
    xPhys = xPhys.reshape(-1, 2, order='F')

    if model.ft == 1:
        xPhys = xPhys
    elif model.ft == 2:
        xPhys[:, 0] = np.asarray(model.H * xPhys[:, 0].reshape(-1, 1, order='F') / model.Hs).flatten('F')
        xPhys[:, 1] = np.asarray(model.H * xPhys[:, 1].reshape(-1, 1, order='F') / model.Hs).flatten('F')

    # sK vector
    Cmax = np.array([[1.3462, 0.5769, 0], [0.5769, 1.3462, 0], [0, 0, 0.3846]])  # minimum stiffness tensor
    Cmin = np.array([[1.3462, 0.5769, 0], [0.5769, 1.3462, 0], [0, 0, 0.3846]]) * 1e-9  # maximum stiffness tensor
    x_test = xPhys
    C_pred = model.nn_model.predict(x_test, verbose=0)  # predict stiffness tensor
    dC_pred = dcfun(x_test, model.nn_model)  # neural network-based sensitivity coefficients

    # FEA
    sK_mat = np.zeros((64, xPhys.shape[0]))
    for xx in np.arange(0, xPhys.shape[0]):
        C = np.array([[C_pred[xx, 0], C_pred[xx, 1], 0], [C_pred[xx, 1], C_pred[xx, 2], 0], [0, 0, C_pred[xx, 3]]])
        C = np.minimum(np.maximum(C, Cmin), Cmax)
        KEvec = kefun_C(C).reshape(-1, 1, order='F')
        sK_mat[:, xx] = KEvec.squeeze()
    sK_vec = sK_mat.reshape(-1, 1, order='F')

    sdK_mat = np.zeros((64, xPhys.shape[0], 2))
    for dd in np.arange(0, 2):
        for xx in np.arange(0, xPhys.shape[0]):
            dC = np.array([[dC_pred[xx, 0, dd], dC_pred[xx, 1, dd], 0], [dC_pred[xx, 1, dd], dC_pred[xx, 2, dd], 0], [0, 0, dC_pred[xx, 3, dd]]])
            dKEvec = kefun_C(dC).reshape(-1, 1, order='F')
            sdK_mat[:, xx, dd] = dKEvec.squeeze()

    # Stiffness matrix
    iKint = model.iK.astype(int).flatten()
    jKint = model.jK.astype(int).flatten()
    sKshp = sK_vec.flatten()
    K = coo_matrix((sKshp, (iKint, jKint)))
    K = 0.5 * (K.transpose() + K)

    # Solve finite element equation
    U = np.zeros((model.dofs, 1))
    K = csc_matrix(K)
    F = csc_matrix(model.F)
    U[model.freedofs, :] = spsolve(K[model.freedofs, model.freedofs.reshape(-1, 1, order='F')],
                             F[model.freedofs, :]).reshape(-1, 1, order='F')
    U[model.fixeddofs, :] = 0

    # Objective function
    ceU = U[model.edofMat][:, :, 0]  # Element compliance matrix
    ceUKU = np.zeros(xPhys.shape[0])
    for xx in np.arange(0, xPhys.shape[0]):
        KE = sK_mat[:, xx].reshape(8, 8, order='F')
        ceUKU[xx] = np.matmul(np.matmul(ceU[xx, :], KE), ceU[xx, :])

    # Derivative of objective function
    ceUdKU = np.zeros((xPhys.shape[0], 2))
    for dd in np.arange(0, 2):
        for xx in np.arange(0, xPhys.shape[0]):
            dKE = sdK_mat[:, xx, dd].reshape(8, 8, order='F')
            ceUdKU[xx, dd] = np.matmul(np.matmul(ceU[xx, :], dKE), ceU[xx, :])

    dc = -1 * ceUdKU
    dv = np.ones(dc.shape)

    # Apply filter
    dcf1, dvf1 = filterfun(xPhys[:, 0].reshape((model.nely, model.nelx), order='F'), dc[:, 0].reshape((model.nely, model.nelx), order='F'),
                         dv[:, 0].reshape((model.nely, model.nelx), order='F'), model.H, model.Hs, model.ft)
    dcf2, dvf2 = filterfun(xPhys[:, 1].reshape((model.nely, model.nelx), order='F'), dc[:, 1].reshape((model.nely, model.nelx), order='F'),
                           dv[:, 1].reshape((model.nely, model.nelx), order='F'), model.H, model.Hs, model.ft)
    dcf = np.zeros(dc.shape)
    dcf[:, 0] = dcf1.flatten('F')
    dcf[:, 1] = dcf2.flatten('F')

    return dcf.flatten('F')


# Neural network-based sensitivity coefficients
def dcfun(x_test, nn_model):
    Nt = x_test.shape[0]
    ydim = nn_model.output_shape[1]  # y dimension
    xdim = x_test.shape[1]  # x dimension

    NL = len(nn_model.layers) - 3  # number of hidden layers
    NN = nn_model.layers[1].output_shape[1]  # number of neurons per hidden layer

    norm_in_layer = nn_model.layers[0]
    norm_out_layer = nn_model.layers[-1]

    # DEFINE ACTIVATION FUNCTIONS
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

    # DEFINE NORMALIZATION FUNCTION AND PARAMETERS
    mean_in = norm_in_layer.mean.numpy()  # mean of input normalization
    std_in = norm_in_layer.variance.numpy() ** 0.5  # standard deviation of input normalization
    mean_out = norm_out_layer.mean.numpy()  # mean of output normalization
    std_out = norm_out_layer.variance.numpy() ** 0.5  # standard deviation of output normalization

    def norm_fn(x, mu, sigma):  # normalization function
        return np.divide((x - mu), sigma)

    # 1) Analytical Derivative of NN
    if model.der_type == 1:
        Wn = []
        Bn = []
        An = []
        for L in np.arange(0, NL + 1):
            Wn.append(nn_model.layers[L + 1].weights[0].numpy())  # store hidden layer weights
            Bn.append(nn_model.layers[L + 1].bias.numpy().reshape(1, -1))  # store hidden layer biases
            An.append(nn_model.layers[L + 1].activation.__name__)  # store hidden layer activation function names

        y_1 = np.zeros((Nt, ydim))  # Manual NN prediction
        dy_1 = np.zeros((Nt, ydim, xdim))
        for s in np.arange(0, Nt):  # loop through test feature sets
            Y0 = norm_in_layer(x_test[s, :]).numpy().reshape(xdim, 1)  # normalize input
            Yn = [act_fn(Y0.T, Wn[0], Bn[0], An[0])]  # initialize 1st layer output
            dY0 = dact_fn(Y0.T, Wn[0], Bn[0], An[0])  # initialize 1st layer output's derivative
            dYn = [
                np.divide(dY0, std_in.reshape(xdim, 1))]  # normalize derivative of input with standard deviation of layer
            dy_product = dYn[0]

            for L in np.arange(1,
                               NL + 1):  # manually loop through layers to calculate analytical derivative with chain rule
                Yn.append(act_fn(Yn[L - 1], Wn[L], Bn[L], An[L]))  # hidden layer L output
                dYn.append(dact_fn(Yn[L - 1], Wn[L], Bn[L], An[L]))  # derivative of hidden layer L output
                dy_product = np.matmul(dy_product, dYn[L])  # derivative of hidden layer L output with respect NN input
            y_1[s, :] = norm_out_layer(Yn[-1]).numpy()  # de-normalize to get manual NN prediction
            dy_product = np.divide(dy_product,
                                   std_out)  # de-normalize derivative of output with standard deviation of layer
            dy_1[s, :, :] = dy_product.T.reshape(1, ydim, xdim)  # derivative via Analytical Derivative of NN
        dC_pred = dy_1

    # 2) Central Finite Difference Approximation
    if model.der_type == 2:
        h_cfd = 1e-6  # step size
        dy_2 = np.zeros((Nt, ydim, xdim))
        for d in np.arange(0, xdim):
            h_mat = np.zeros((Nt, xdim))
            h_mat[:, [d]] = np.ones((Nt, 1)) * h_cfd  # perturbation array
            dy_2[:, :, d] = ((nn_model.predict(x_test + h_mat, verbose=0) - nn_model.predict(x_test - h_mat, verbose=0))
                             / (2 * h_cfd)).reshape(Nt, ydim)  # derivative via CFDA
        dC_pred = dy_2

    # 3) Complex Step Derivative Approximation
    if model.der_type == 3:
        Wn = []
        Bn = []
        An = []
        for L in np.arange(0, NL + 1):
            Wn.append(nn_model.layers[L + 1].weights[0].numpy())  # store hidden layer weights
            Bn.append(nn_model.layers[L + 1].bias.numpy().reshape(1, -1))  # store hidden layer biases
            An.append(nn_model.layers[L + 1].activation.__name__)  # store hidden layer activation function names

        h_csm = 1e-12  # step size
        dy_3 = np.zeros((Nt, ydim, xdim))
        for d in np.arange(0, xdim):  # loop through xdim for partial derivatives
            h_mat = 0j * np.zeros((1, xdim), dtype='complex')
            h_mat[0, d] = 1j * h_csm  # perturbation array

            y_csm = np.zeros((Nt, ydim), dtype='complex')
            for s in np.arange(0, Nt):  # loop through test feature sets
                Y0 = norm_fn(x_test[s, :] + h_mat, mean_in, std_in).reshape(xdim, 1)  # normalize perturbed input
                Yn_csm = [act_fn(Y0.T, Wn[0], Bn[0], An[0])]  # initialize 1st layer output

                for L in np.arange(1, NL + 1):  # manual NN prediction so imaginary numbers can be passed
                    Yn_csm.append(act_fn(Yn_csm[L - 1], Wn[L], Bn[L], An[L]))  # hidden layer L output
                y_csm[s, :] = norm_fn(Yn_csm[-1], mean_out, std_out)  # de-normalize to get manual NN prediction
            dy_3[:, :, d] = (np.imag(y_csm) / h_csm)  # derivative via CSDA
        dC_pred = dy_3

    # 4) Automatic Differentiation
    if model.der_type == 4:
        xn_tape = tf.Variable(x_test, dtype=tf.float64)
        with tf.GradientTape(persistent=True) as tape:
            yn_tape = xn_tape
            for layer in nn_model.layers:  # loop through layers
                yn_tape = layer(yn_tape)
        dy_4 = tf.reduce_sum(tape.jacobian(yn_tape, xn_tape), axis=[2]).numpy().reshape(Nt, ydim, xdim)  # derivative via AD
        dC_pred = dy_4

    return dC_pred


# Non-linear volume fraction constraint
def nlcon(xPhys):
    xPhys = xPhys.reshape(-1, 2, order='F')
    return ((model.nelx * model.nely - np.sum(np.multiply(xPhys[:, 0], xPhys[:, 1]))) / (model.nelx * model.nely)) - model.volfrac


# Derivative of non-linear volume fraction constraint
def dnlcon(xPhys):
    xPhys = xPhys.reshape(-1, 2, order='F')
    dc = np.ones(xPhys.shape)
    dv = np.ones(xPhys.shape)
    dv[:, 0] = -xPhys[:, 1] / (model.nelx * model.nely)
    dv[:, 1] = -xPhys[:, 0] / (model.nelx * model.nely)

    # Apply filter
    dcf1, dvf1 = filterfun(xPhys[:, 0].reshape((model.nely, model.nelx), order='F'), dc[:, 0].reshape((model.nely, model.nelx), order='F'),
                           dv[:, 0].reshape((model.nely, model.nelx), order='F'), model.H, model.Hs, model.ft)
    dcf2, dvf2 = filterfun(xPhys[:, 1].reshape((model.nely, model.nelx), order='F'), dc[:, 1].reshape((model.nely, model.nelx), order='F'),
                           dv[:, 1].reshape((model.nely, model.nelx), order='F'), model.H, model.Hs, model.ft)
    dvf = dv
    dvf[:, 0] = dvf1.flatten('F')
    dvf[:, 1] = dvf2.flatten('F')

    return dvf.flatten('F')


# Save select iteration information
def save_step(x, it):
    # Define which iterations get saved
    if it.niter == 1 or it.niter == 10 or it.niter == 100 or it.niter % 500 == 0:
        xopt1 = x[0:(model.nely * model.nelx)].reshape(model.nely, model.nelx, order='F')
        xopt2 = x[(model.nely * model.nelx):].reshape(model.nely, model.nelx, order='F')
        topt = it.execution_time
        comp = it.fun
        niter = it.niter
        dc = it.grad

        filename = 'HBTO_NN_result_dy' + str(model.der_type) + '/HBTO_NN_dy' + str(model.der_type) + '_it' + str(niter)
        np.savez(filename, xopt1=xopt1, xopt2=xopt2, topt=topt, comp=comp, dc=dc, niter=niter)  # save file


# Main function
def HBTO(model):
    bds = sp.optimize.Bounds(lb=0, ub=1, keep_feasible=True)  # define theta bounds
    con = sp.optimize.NonlinearConstraint(nlcon, lb=0.0, ub=0.0, jac=dnlcon, keep_feasible=True)  # define NL constraint
    res = sp.optimize.minimize(feafun,
                               model.x,
                               args=(model),
                               method='trust-constr',
                               jac=dfeafun,
                               bounds=bds,
                               constraints=con,
                               tol=model.tolx,
                               options={'maxiter': model.maxiter, 'disp': True, 'verbose': 3},
                               callback=save_step)
    return res


# Create microstructure with rectangular hole
def create_Micro(nel, theta1, theta2):
    ye, xe = np.meshgrid(np.linspace(-1 + 1/nel, +1 - 1/nel, nel), np.linspace(-1 + 1/nel, +1 - 1/nel, nel))
    ecoords_c_norm = np.concatenate((xe.reshape(-1, 1, order='F'), ye.reshape(-1, 1, order='F')), axis=1)
    xMicro = np.zeros((nel*nel, 1))
    for e in np.arange(0, nel*nel):
        if ecoords_c_norm[e, 0] <= -1 + (1 - theta1):  # right
            xMicro[e, 0] = 1
        if ecoords_c_norm[e, 1] <= -1 + (1 - theta2):  # bottom
            xMicro[e, 0] = 1
        if ecoords_c_norm[e, 0] >= 1 - (1 - theta1):  # left
            xMicro[e, 0] = 1
        if ecoords_c_norm[e, 1] >= 1 - (1 - theta2):  # top
            xMicro[e, 0] = 1
    return xMicro.reshape(nel, nel, order='F')

# Model class
class model:
    def __init__(self):
        self.x = None  # Densities
        self.xPhys = None  # Densities after filter
        self.F = None  # External forces
        self.dofs = None  # Number of dofs
        self.fixeddofs = None  # Supports
        self.freedofs = None  # Free dofs
        self.KE = None  # Element stiffness matrix
        self.nn_model = None  # NN model
        self.nelx = None  # Number of elements in x-direction
        self.nely = None  # Number of elements in y-direction
        self.H = None
        self.Hs = None
        self.ft = None
        self.edofMat = None
        self.iK = None
        self.jK = None
        self.volfrac = None
        self.rmin = None
        self.maxiter = None
        self.tolx = None
        self.E0 = None  # Void stiffness
        self.Emin = None  # Fluid stiffness
        self.nu = None  # Poissons ratio


# %% RUN HBTO MBB EXAMPLE
# Define model and optimization settings
model.nelx = 60  # number of elements in the x-direction
model.nely = 30  # number of elements in the y-direction
model.volfrac = 0.5  # volume fraction constraint
model.rmin = 2.7  # filter radius
model.ft = 1  # Filter option (1 = sensitivity filter or 2 = density filter)
model.maxiter = 50  # Number of iterations
model.tolx = 1e-3

# Initialize model details
x0 = (np.ones((model.nely * model.nelx, 2)) * np.sqrt(1-model.volfrac)).flatten('F')  # initial design
model.x = x0  # initial design
model.edofMat, model.iK, model.jK = feaprep(model)  # prepare fea
model.H, model.Hs = filterprep(model)  # prepare filter
model.nn_model = tf.keras.models.load_model("NN_model_HBTO_1e+04")  # load nn model
model.der_type = 2  # type of derivative method to use (1 = Analytical Der., 2 = CFD, 3 = CSM, 4 = Automatic Diff.)

# Force vector for the MBB problem
dofs = 2 * (model.nelx + 1) * (model.nely + 1)
fv = np.array([-1])  # force magnitude and direction
fi = np.array([1])  # x coordinate of the dof
fj = np.array([0])  # y coordinate of the dof
F = coo_matrix((fv, (fi, fj)), shape=(dofs, 1))
model.dofs = dofs
model.F = F

# Fixed dofs for the MBB problem
fxd1 = np.arange(0, 2 * (model.nely + 1), 2)
fxd2 = 2 * (model.nelx + 1) * (model.nely + 1) - 1
model.fixeddofs = np.block([fxd1, fxd2])
model.freedofs = np.setdiff1d(np.arange(0, dofs), model.fixeddofs)

# Solve topology optimization problem
dc0 = dfeafun(x0, model)  # compute initial sensitivity coefficients
res = HBTO(model)  # solve TO problem

# Record and save output data
xopt1 = res.x[0:(model.nely * model.nelx)].reshape(model.nely, model.nelx, order='F')
xopt2 = res.x[(model.nely * model.nelx):].reshape(model.nely, model.nelx, order='F')
topt = res.execution_time
comp = res.fun
niter = res.niter
dc = res.grad

filename = 'HBTO_NN_result_dy' + str(model.der_type) + '/HBTO_NN_dy' + str(model.der_type) + '_it' + str(niter) + 'final'
np.savez(filename, xopt1=xopt1, xopt2=xopt2, topt=topt, comp=comp, dc0=dc0, dc=dc, niter=niter)  # save final design

# Plot final topology
nel = 100
xfinal = np.zeros((model.nely*nel, model.nelx*nel))
xopt1 = xopt1.reshape(-1, 1, order='F')
xopt2 = xopt2.reshape(-1, 1, order='F')
yv, xv = np.meshgrid(np.linspace(0, model.nelx*nel, model.nelx+1), np.linspace(0, model.nely*nel, model.nely+1))
xv1 = xv[0:model.nely, 0:model.nelx].reshape(-1, 1, order='F')
xv2 = xv[1:model.nely+1, 1:model.nelx+1].reshape(-1, 1, order='F')
yv1 = yv[0:model.nely, 0:model.nelx].reshape(-1, 1, order='F')
yv2 = yv[1:model.nely+1, 1:model.nelx+1].reshape(-1, 1, order='F')
for e in np.arange(0, model.nelx*model.nely):
    xfinal[int(xv1[e, 0]):int(xv2[e, 0]), int(yv1[e, 0]):int(yv2[e, 0])] = create_Micro(nel, xopt1[e, 0], xopt2[e, 0])
plt.imshow(xfinal, extent=[0, model.nelx*nel, 0, model.nely*nel], cmap='Greys')
plt.show()