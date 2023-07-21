# Perform Density-based Topology Optimization with the SIMP model
# by Joel Najmon and Andres Tovar
# Python 3.9

# %% IMPORT PACKAGES
import numpy as np  # version 1.23.5
import numpy.matlib as npm
import scipy as sp  # version 1.10.1
from scipy.sparse import csc_matrix, coo_matrix
from scipy.sparse.linalg import spsolve
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
    if model.ft == 1:
        xPhys = xPhys
    elif model.ft == 2:
        xPhys = np.asarray(model.H * xPhys.reshape(-1, 1, order='F') / model.Hs).flatten('F')

    # sK vector
    Cmax = np.array([[1.3462, 0.5769, 0], [0.5769, 1.3462, 0], [0, 0, 0.3846]])  # minimum stiffness tensor
    Cmin = np.array([[1.3462, 0.5769, 0], [0.5769, 1.3462, 0], [0, 0, 0.3846]]) * 1e-9  # maximum stiffness tensor
    x_test = xPhys
    C0 = np.ones((x_test.shape[0], 4))
    C0[:, 0] = C0[:, 0] * Cmax[0, 0]
    C0[:, 1] = C0[:, 1] * Cmax[1, 0]
    C0[:, 2] = C0[:, 2] * Cmax[1, 1]
    C0[:, 3] = C0[:, 3] * Cmax[2, 2]
    C_simp = np.transpose(npm.repmat((1e-9 + x_test**model.penal*(1 - 1e-9)), 4, 1)) * C0  # SIMP model

    # FEA
    sK_mat = np.zeros((64, xPhys.shape[0]))
    for xx in np.arange(0, xPhys.shape[0]):
        C = np.array([[C_simp[xx, 0], C_simp[xx, 1], 0], [C_simp[xx, 1], C_simp[xx, 2], 0], [0, 0, C_simp[xx, 3]]])
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
    U = np.zeros((dofs, 1))
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
    if model.ft == 1:
        xPhys = xPhys
    elif model.ft == 2:
        xPhys = np.asarray(model.H * xPhys.reshape(-1, 1, order='F') / model.Hs).flatten('F')

    # sK vector
    Cmax = np.array([[1.3462, 0.5769, 0], [0.5769, 1.3462, 0], [0, 0, 0.3846]])  # minimum stiffness tensor
    Cmin = np.array([[1.3462, 0.5769, 0], [0.5769, 1.3462, 0], [0, 0, 0.3846]]) * 1e-9  # maximum stiffness tensor
    x_test = xPhys
    C0 = np.ones((x_test.shape[0], 4))
    C0[:, 0] = C0[:, 0] * Cmax[0, 0]
    C0[:, 1] = C0[:, 1] * Cmax[1, 0]
    C0[:, 2] = C0[:, 2] * Cmax[1, 1]
    C0[:, 3] = C0[:, 3] * Cmax[2, 2]
    C_simp = np.transpose(npm.repmat((1e-9 + x_test**model.penal*(1 - 1e-9)), 4, 1)) * C0  # SIMP stiffness tensor
    dC_simp = (np.transpose(npm.repmat((model.penal * (1 - 1e-9) * x_test**(model.penal - 1)), 4, 1)) * C0).reshape(-1, 4, 1, order='F')  # derivative of SIMP stiffness tensor

    # FEA
    sK_mat = np.zeros((64, xPhys.shape[0]))
    for xx in np.arange(0, xPhys.shape[0]):
        C = np.array([[C_simp[xx, 0], C_simp[xx, 1], 0], [C_simp[xx, 1], C_simp[xx, 2], 0], [0, 0, C_simp[xx, 3]]])
        C = np.minimum(np.maximum(C, Cmin), Cmax)
        KEvec = kefun_C(C).reshape(-1, 1, order='F')
        sK_mat[:, xx] = KEvec.squeeze()
    sK_vec = sK_mat.reshape(-1, 1, order='F')

    sdK_mat = np.zeros((64, xPhys.shape[0], 2))
    for dd in np.arange(0, 1):
        for xx in np.arange(0, xPhys.shape[0]):
            dC = np.array([[dC_simp[xx, 0, dd], dC_simp[xx, 1, dd], 0], [dC_simp[xx, 1, dd], dC_simp[xx, 2, dd], 0], [0, 0, dC_simp[xx, 3, dd]]])
            dKEvec = kefun_C(dC).reshape(-1, 1, order='F')
            sdK_mat[:, xx, dd] = dKEvec.squeeze()

    # Stiffness matrix
    iKint = model.iK.astype(int).flatten()
    jKint = model.jK.astype(int).flatten()
    sKshp = sK_vec.flatten()
    K = coo_matrix((sKshp, (iKint, jKint)))
    K = 0.5 * (K.transpose() + K)

    # Solve finite element equation
    U = np.zeros((dofs, 1))
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
    ceUdKU = np.zeros((xPhys.shape[0], 1))
    for dd in np.arange(0, 1):
        for xx in np.arange(0, xPhys.shape[0]):
            dKE = sdK_mat[:, xx, dd].reshape(8, 8, order='F')
            ceUdKU[xx, dd] = np.matmul(np.matmul(ceU[xx, :], dKE), ceU[xx, :])

    dc = -1 * ceUdKU
    dv = np.ones(dc.shape)

    # Apply filter
    dcf, dvf = filterfun(xPhys.reshape((model.nely, model.nelx), order='F'), dc.reshape((model.nely, model.nelx), order='F'),
                         dv.reshape((model.nely, model.nelx), order='F'), model.H, model.Hs, model.ft)

    return np.asarray(dcf).flatten('F')


# Non-linear volume fraction constraint
def nlcon(xPhys):
    one1 = np.ones(xPhys.shape)
    return (np.dot(one1, xPhys) / (model.nelx * model.nely)) - model.volfrac


# Derivative of non-linear volume fraction constraint
def dnlcon(xPhys):
    dc = np.ones(xPhys.shape)
    dv = np.ones(xPhys.shape)

    # Apply filter
    dcf, dvf = filterfun(xPhys.reshape((model.nely, model.nelx), order='F'), dc.reshape((model.nely, model.nelx), order='F'),
              dv.reshape((model.nely, model.nelx), order='F'), model.H, model.Hs, model.ft)
    return dvf.flatten('F')


# Save select iteration information
def save_step(x, it):
    # Define which iterations get saved
    if it.niter == 1 or it.niter == 10 or it.niter == 100 or it.niter % 500 == 0:
        xopt = x.reshape(model.nely, model.nelx, order='F')
        topt = it.execution_time
        comp = it.fun
        niter = it.niter
        dc = it.grad

        filename = 'DBTO_SIMP_result/DBTO_SIMP_it' + str(niter)
        np.savez(filename, xopt=xopt, topt=topt, comp=comp, dc=dc, niter=niter)  # save file


# Main function
def DBTO(model):
    bds = sp.optimize.Bounds(lb=0, ub=1, keep_feasible=True)
    con = sp.optimize.NonlinearConstraint(nlcon, lb=0.0, ub=0.0, jac=dnlcon, keep_feasible=True)
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
        self.penal = None


# %% RUN DBTO-SIMP MBB EXAMPLE
# Define model and optimization settings
model.nelx = 60  # number of elements in the x-direction
model.nely = 30  # number of elements in the y-direction
model.volfrac = 0.5  # volume fraction constraint
model.rmin = 2.7  # filter radius
model.penal = 3.0  # penalization parameter for SIMP
model.ft = 1  # Filter option (1 = sensitivity filter or 2 = density filter)
model.maxiter = 1000  # Number of iterations
model.tolx = 1e-3

# Initialize model details
x0 = (np.ones((model.nely * model.nelx, 1)) * model.volfrac).flatten()  # initial design
model.x = x0  # initial design
model.edofMat, model.iK, model.jK = feaprep(model)  # prepare fea
model.H, model.Hs = filterprep(model)  # prepare filter

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
res = DBTO(model)  # solve TO problem

# Record and save output data
xopt = res.x.reshape(model.nely, model.nelx, order='F')
topt = res.execution_time
comp = res.fun
niter = res.niter
dc = res.grad

filename = 'DBTO_SIMP_result/DBTO_SIMP_it' + str(niter) + 'final'
np.savez(filename, xopt=xopt, topt=topt, comp=comp, dc0=dc0, dc=dc, niter=niter)  # save final design

# Plot final topology
plt.imshow(xopt, extent=[0, model.nelx, 0, model.nely], cmap='Greys')
plt.show()