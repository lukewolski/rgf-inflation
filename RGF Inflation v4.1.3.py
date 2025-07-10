## This is the document that will contain the methods suitable for execution in Quark.
## The purpose of this document is to hone in on the relevant tools and methods,
## omit the unnecessary content, and satisfy that craving for elegance.

###############################################################################
## ------------------------------IMPORTATIONS------------------------------- ##
###############################################################################


import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import pandas as pd
import time
import itertools
import os
import shutil
import scipy.linalg
import scipy.signal
import scipy.interpolate
import scipy.integrate
import scipy.stats
import concurrent.futures
import copy
from scipy.interpolate import CubicSpline as CS
from datetime import date

###############################################################################
## ----------------------------GLOBAL CONSTANTS----------------------------- ##
###############################################################################



G = 6.674e-11
hbar = 1.055e-34
c = 2.998e8
useMmcUnits = True
plc = 1/3 if useMmcUnits else 8*np.pi/3
plancklabel = r"$(8 \pi G)^{-1} = 1$" if useMmcUnits else r"$G = 1$"
nEBack4Pivot = 55
kPivot = 5.25e-60 ## = 0.002 Mpc^-1
kStar = 1.3125e-58 ## 0.05 Mpc^-1
noise = 1e-12
maxCovLen = 1000
timelabel = r"Time [$t_{Pl}$]"
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'

Q = {'phi':"Inflaton Field",
     'dphi':"Time Derivative of Inflaton Field",
     'V':"Inflaton Potential",
     'dV':"Phi Gradient of Inflaton Potential",
     'ddV':"Hessian Matrix of Second Derivatives of Inflaton Potential",
     't':"Time",
     'nE':"Number of e-Folds of Inflation",
     'params':"Main Parameters (s, V0, N)",
     'H':"Hubble Parameter",
     'a':"Scale Factor",
     'epsilon':"Epsilon Slow Roll Parameter",
     'k':"Horizon-Crossing Wavenumber",
     'inw':"Iota Inwardness Metric",
     'vcov':"Covariance Matrix",
     'vcov_L':"Cholesky Decomposition of Covariance Matrix",
     'M' : "Compression Matrix",
     'beefCov':"Padded Covariance Matrix",
     'beefCov_L':"Padded Cholesky Decomposition of Covariance Matrix",
     'beefM' : "Padded Compression Matrix",
     'phiStuff':"Unfiltered Inflaton Field Values",
     'vStuff':"Inflaton Potential + Derivative Values",
     'vFull':"Unfiltered Inflaton Potential + Derivative Values",
     'infoVec':"Information Matrix",
     'infoFull':"Unfiltered Information Matrix",
     'fulltimes':"Unfiltered Time Values",
     'bowl':"Second Derivatives at the Origin",
     'N':"Dimension of V",
     'state':"Random State of Simulation",
     'nfev':"Number of Function Evaluations",
     'phiSol':"Solution Output of solve_ivp",
     'planck':"Planck Units Assumed",
     'upscale':"Wavenumber Normalization Factor",
     'simtime':"Simulation Time",
     'fails':"Dictionary of Failures Before and During Simulation",
     'success':"Dictionary of Successes After Simulation",
     'forgets':"Number of Forgetting Steps Executed",
     'name':"Name of Simulation",
     'coupler':"Coupling Matrix",
     'tensor_spec':"Power Spectrum of Tensor Perturbations",
     'A_t':"Tensor Spectrum Amplitude",
     'n_t':"Tensor Spectrum Index",
     'A_s':"Matter Spectrum Amplitude",
     'n_s':"Matter Spectrum Index",
     'A_iso':"Isocurvature Spectrum Amplitude",
     'n_iso':"Isocurvature Spectrum Index",
     'r':"Tensor-to-Scalar Ratio"}



###############################################################################
## ------------------------COVARIANCE MATRIX AND XTX------------------------ ##
###############################################################################


# Probabilist's Hermite polynomials

def He_0(x):
    return 1


def He_1(x):
    return x


def He_2(x):
    return x ** 2 - 1


def He_3(x):
    return x ** 3 - 3 * x


def He_4(x):
    return x ** 4 - 6 * x ** 2 + 3


def He(n, x):
    func = np.array([He_0, He_1, He_2, He_3, He_4]).take(n)
    return func(x)


# Covariance between potential values at two different points in space.
# (Fundamental property of Gaussian random fields)
def vv_covariance(phi_j, phi_k, s):
    return np.exp(-np.sum((phi_j - phi_k) ** 2, axis=-1) / (2 * s ** 2))

# New method for populating the covariance matrix.
def lightning_fast_vcov(phi, infoVec, s):
    """
    Populates covariance matrix in a vectorized way.

    Parameters
    -------------------
    phi : (N,m) np.ndarray
        phi values corresponding to constraints on the GRF
    infoVec: [3 x T array]
        information matrix of indices, derivatives
    s : float
        coherence scale
    """

    infoVec = infoVec.astype(int)
    d1 = infoVec[1]
    d2 = infoVec[2]

    # Get 'm', the number of constraints on V, and 'N', the dimension of phi
    m,N = np.shape(phi.T)
    m = len(d1)
    phi_dp = phi.T[infoVec[0,:m]]

    # Initialize the covariance matrix with zeros
    vcov = np.zeros((m, m))

    # Create grids using np.meshgrid
    # Information about constraints j,k are found in phi_j, phi_k, d1_j, d1_k, d2_j, d2_k
    # phi_j[j,k] and phi_k[j,k] yield the positions in (inflaton) space of constraints j,k
    # d1_j[j,k] and d1_k[j,k] yield the first differentiation variables of constraints j,k (-1 implies no derivative)
    # d2_j[j,k] and d2_k[j,k] yield the second differentiation variables of constraints j,k (-1 implies no second derivative)
    phi_j = np.zeros((m, m, N))
    phi_k = np.zeros((m, m, N))
    for alpha in range(N):
        phi_j[:, :, alpha], phi_k[:, :, alpha] = np.meshgrid(phi_dp[:, alpha], phi_dp[:, alpha])
    d1_j, d1_k = np.meshgrid(d1, d1)
    d2_j, d2_k = np.meshgrid(d2, d2)

    # Compute total number 'n' of derivatives with respect to each differentiation variable, 'alpha' (n[:,:,alpha])
    # J[:,:,alpha], K[:,:,alpha] are the total number of derivatives with respect to phi_j^{(alpha)} and phi_k^{(alpha)}
    # Both 'n' and 'K' are necessary to get the sign of the covariance correct.
    J = np.zeros((m, m, N))
    K = np.zeros((m, m, N))
    for alpha in range(N):
        J[:, :, alpha] = (d1_j == alpha).astype(int) + (d2_j == alpha).astype(int)
        K[:, :, alpha] = (d1_k == alpha).astype(int) + (d2_k == alpha).astype(int)

    n = J + K
    # Start by populating every element with the base covariance
    vcov = vv_covariance(phi_j, phi_k, s) * (-1) ** np.sum(n, axis=-1) * (-1) ** np.sum(K, axis=-1)
    M = vv_covariance(phi_j, phi_k, s*np.sqrt(2)) * (-1) ** np.sum(n, axis=-1) * (-1) ** np.sum(K, axis=-1)

    # Multiply on Hermite polynomials, one differentiation variable at a time
    for alpha in range(N):

        # Loop through all possible Hermite polynomials, multiplying where appropriate
        for nn in range(5):
            nn_mask = (n[:, :, alpha] == nn)
            vcov[nn_mask] *= He(nn, (phi_j[nn_mask, alpha] - phi_k[nn_mask, alpha]) / s ** 2)
            M[nn_mask] *= He(nn, (phi_j[nn_mask, alpha] - phi_k[nn_mask, alpha]) / (s*np.sqrt(2)) ** 2)
    return vcov, M


def lightning_fast_beefupdate(s, phiPoints, infoVec, beefCov, beefM, bar):
    phi = phiPoints.T
    N = len(phi[0])
    infoVec = infoVec.astype(int)
    d1 = infoVec[1]
    d2 = infoVec[2]
    T = len(infoVec[0])
    phi_dp = phi[infoVec[0,:T]]

    phi_j = np.zeros((T, T, N))
    phi_k = np.zeros((T, T, N))
    for alpha in range(N):
        phi_j[:, :, alpha], phi_k[:, :, alpha] = np.meshgrid(phi_dp[:, alpha], phi_dp[:, alpha])
    d1_j, d1_k = np.meshgrid(d1, d1)
    d2_j, d2_k = np.meshgrid(d2, d2)

    J = np.zeros((T, T, N))
    K = np.zeros((T, T, N))
    for alpha in range(N):
        J[:, :, alpha] = (d1_j == alpha).astype(int) + (d2_j == alpha).astype(int)
        K[:, :, alpha] = (d1_k == alpha).astype(int) + (d2_k == alpha).astype(int)

    n = J + K
    mask = np.full((maxCovLen,maxCovLen),False)
    mask[:T,:T] = True
    mask[:bar,:bar] = False
    beefCov[mask] = vv_covariance(phi_j[mask[:T,:T]],phi_k[mask[:T,:T]],s) * (-1) ** np.sum(n[mask[:T,:T]], axis=-1) * (-1) ** np.sum(K[mask[:T,:T]], axis=-1)
    beefM[mask] = vv_covariance(phi_j[mask[:T,:T]],phi_k[mask[:T,:T]],s*np.sqrt(2)) * (-1) ** np.sum(n[mask[:T,:T]], axis=-1) * (-1) ** np.sum(K[mask[:T,:T]], axis=-1)

    nn_mask = np.full((maxCovLen,maxCovLen),False)

    for alpha in range(N):

        for nn in range(5):
            nn_mask[:T,:T] = (n[:,:, alpha] == nn)
            nn_mask[:bar,:bar] = False
            beefCov[nn_mask] *= He(nn, (phi_j[nn_mask[:T,:T],alpha] - phi_k[nn_mask[:T,:T],alpha]) / s ** 2)
            beefM[nn_mask] *= He(nn, (phi_j[nn_mask[:T,:T],alpha] - phi_k[nn_mask[:T,:T],alpha]) / (s*np.sqrt(2)) ** 2)
    return beefCov, beefM




def updateCholesky(beefCov_L, vcov, bar):

    """
    Rank 1 update of Cholesky decomposed covariance matrix.
    - beefCov_L: [(bar + M)^2 matrix] large matrix of zeros with old Cholesky decomposed covariance matrix in upper left (bar)^2 section
    - vcov: [T^2 matrix] updated covariance matrix
    - bar: size of old Cholesky decomposed covariance matrix
    """

    lv = len(vcov)

    quartered = quarter(vcov, bar)
    ON, NN = quartered[1], quartered[3]

    n = sp.linalg.solve_triangular(beefCov_L[:bar,:bar], ON, lower = True, check_finite = False)
    toChol = NN - np.transpose(n) @ n
    l = sp.linalg.cholesky(toChol + noise*np.identity(len(toChol)), lower = True)

    beefCov_L[bar:lv, :bar] = np.transpose(n)
    beefCov_L[bar:lv, bar:lv] = l

    return beefCov_L



###############################################################################
## -----------------------------PATH GENERATION----------------------------- ##
###############################################################################



class BStuff():

    """
    Stores and organizes the outputs of a single "bushwhack" simulation.
    """

    def __init__(self, **kwargs):

        self.phi = kwargs.get('phi')
        self.dphi = kwargs.get('dphi')
        self.V = kwargs.get('V')
        self.dV = kwargs.get('dV')
        self.ddV = kwargs.get('ddV')
        self.t = kwargs.get('t')
        self.nE = kwargs.get('nE')
        self.params = kwargs.get('params')
        self.H = kwargs.get('H')
        self.a = kwargs.get('a')
        self.epsilon = kwargs.get('epsilon')
        self.k = kwargs.get('k')
        self.inw = kwargs.get('inw')
        self.vcov = kwargs.get('vcov')
        self.vcov_L = kwargs.get('vcov_L')
        self.beefCov = np.zeros((maxCovLen,maxCovLen)) if kwargs.get('beefCov') is None else kwargs.get('beefCov')
        self.beefCov_L = np.zeros((maxCovLen,maxCovLen)) if kwargs.get('beefCov_L') is None else kwargs.get('beefCov_L')
        self.M = kwargs.get('M')
        self.beefM = np.zeros((maxCovLen,maxCovLen)) if kwargs.get('beefM') is None else kwargs.get('beefM')
        self.A = kwargs.get('A')
        self.phiStuff = kwargs.get('phiStuff')
        self.vStuff = kwargs.get('vStuff')
        self.vFull = kwargs.get('vFull')
        self.infoVec = kwargs.get('infoVec')
        self.infoFull = kwargs.get('infoFull')
        self.fulltimes = [] if kwargs.get('fulltimes') is None else kwargs.get('fulltimes')
        self.bowl = kwargs.get('bowl')
        self.N = None if kwargs.get('N') is None else int(kwargs.get('N'))
        self.state = kwargs.get('state')
        self.nfev = 0 if kwargs.get('nfev') is None else kwargs.get('nfev')
        self.phiSol = kwargs.get('phiSol')
        self.planck = plancklabel if kwargs.get('planck') is None else kwargs.get('planck')
        self.upscale = kwargs.get('upscale')
        self.simtime = kwargs.get('simtime')
        self.fails = {"Inwardness": 0, "e-Folds": 0, "Total": 0, "Exception": False, "Overall": False} if kwargs.get('fails') is None else kwargs.get('fails')
        self.success = {"Inwardness": False, "Convergence": False, "e-Folds": False} if kwargs.get('success') is None else kwargs.get('success')
        self.forgets = 0 if kwargs.get('forgets') is None else kwargs.get('forgets')
        self.name = kwargs.get('name')

        self.coupler = kwargs.get('coupler')
        self.tensor_spec = kwargs.get('tensor_spec')
        self.A_t = kwargs.get('A_t')
        self.n_t = kwargs.get('n_t')
        self.A_s = kwargs.get('A_s')
        self.n_s = kwargs.get('n_s')
        self.A_iso = kwargs.get('A_iso')
        self.n_iso = kwargs.get('n_iso')
        self.r = kwargs.get('r')

        return

    def __str__(self, shapes=False):
        print(f"Parameters:\ns (Coherence scale) = {self.params[0]}\nV_0 (Inflationary Energy Scale) = {self.params[1]}\nN (# of inflaton fields) = {self.params[2]}\n")
        print(f"Planck units: {self.planck}")
        print(f"Simulation time: {self.simtime} seconds")
        print(f"Name: {self.name}\n")
        print(f"Quantities:\nA_t = {self.A_t}, n_t = {self.n_t}\nA_s = {self.A_s}, n_s = {self.n_s}\nr = {self.r}")
        if shapes:
            print(f"\nShape of phi: {np.shape(self.phi)}")
            print(f"Shape of d(phi)/dN_e: {np.shape(self.dphi)}")
            print(f"Shape of V: {np.shape(self.V)}")
            print(f"Shape of grad(V): {np.shape(self.dV)}")
            print(f"Shape of laplacian(V): {np.shape(self.ddV)}")
            print(f"Shape of t: {np.shape(self.t)}")
            print(f"Shape of nE: {np.shape(self.nE)}")
            print(f"Shape of H: {np.shape(self.H)}")
            print(f"Shape of a: {np.shape(self.a)}")
            print(f"Shape of epsilon: {np.shape(self.epsilon)}")
            print(f"Shape of k: {np.shape(self.k)}")
            print(f"Shape of V covariance matrix: {np.shape(self.vcov)}")
            print(f"Shape of coupling tensor: {np.shape(self.coupler)}")
            print(f"Shape of tensor spectrum: {np.shape(self.tensor_spec)}")
        return f"Random path {hex(id(self))}."

    def s(self): return self.params[0]
    def V0(self): return self.params[1]
    def strongMin(self): return self.V0()/self.s()**2
    def minStrength(self): return self.bowl/self.strongMin()*100
    def nEBounds(self): I = self.V[0]/np.linalg.norm(self.dV[:,0])*np.linalg.norm(self.phi[:,0]); return (I/4, 3*I/4)

    def phiFunc(self, nEVals, d=0):

        """
        Interpolated inflaton field values phi(N_e).
        - nEVals: times at which to sample phi
        - d: derivative number
        """

        funclist = np.empty(self.N, dtype=object)
        for dim in range(self.N):
            funclist[dim] = CS(self.nE, self.phi[dim]).derivative(d)
        phiVals = np.zeros((self.N, len(nEVals)))
        for i in range(len(nEVals)):
            for j in range(len(funclist)):
                phiVals[j,i] = funclist[j](nEVals[i])

        return phiVals

    def dphiFunc(self, nEVals, d=0):

        """
        Interpolated inflaton field derivatives d/dN_e(phi(N_e)).
        - nEVals: times at which to sample phi
        - d: derivative number
        """

        funclist = np.empty(self.N, dtype=object)
        for dim in range(self.N):
            funclist[dim] = CS(self.nE, self.dphi[dim]).derivative(d)
        dphiVals = np.zeros((self.N, len(nEVals)))
        for i in range(len(nEVals)):
            for j in range(len(funclist)):
                dphiVals[j,i] = funclist[j](nEVals[i])

        return dphiVals

    def VFunc(self, nEVals, strictlyInc=False, d=0):

        """
        Interpolated inflaton potential values V(N_e).
        - nEVals: times at which to sample phi
        - strictlyInc: self.nE is strictly increasing
        - d: derivative number
        """

        func = []
        if strictlyInc: func = CS(self.nE, self.V).derivative(d)
        else:
            func = sp.interpolate.interp1d(self.nE, self.V)
            if d>0: print("Time values must be strictly increasing to get derivatives.")
        VVals = np.zeros(len(nEVals))
        for i in range(len(nEVals)):
            VVals[i] = func(nEVals[i])

        return VVals

    def dVFunc(self, nEVals, strictlyInc=False, d=0):

        """
        Interpolated inflaton potential partial derivatives d/d(phi^(alpha)) (V(N_e)).
        - nEVals: times at which to sample phi
        - strictlyInc: self.nE is strictly increasing
        - d: derivative number
        """

        funclist = np.empty(self.N, dtype=object)
        for dim in range(self.N):
            if strictlyInc: funclist[dim] = CS(self.nE, self.dV[dim]).derivative(d)
            else: funclist[dim] = sp.interpolate.interp1d(self.nE, self.dV[dim])
        dVVals = np.zeros((self.N, len(nEVals)))
        for i in range(len(nEVals)):
            for j in range(len(funclist)):
                dVVals[j,i] = funclist[j](nEVals[i])

        return dVVals

    def traj_df(self):

        """
        Dataframe of basic information regarding the trajectory.
        """

        cols = ["N_e", "t", "H", "a", "epsilon", "V", "k", "inwardness"]
        cols = np.append(cols, [f"dV{i}" for i in range(self.N)])
        cols = np.append(cols, [f"phi{i}" for i in range(self.N)])
        cols = np.append(cols, [f"dphi{i}" for i in range(self.N)])
        df = pd.DataFrame(columns = cols)

        df['N_e'] = self.nE
        df['t'] = self.t
        df['H'] = self.H
        df['a'] = self.a
        df['epsilon'] = self.epsilon
        df['V'] = self.V
        df['k'] = self.k
        df['inwardness'] = self.inw
        for i in range(self.N):
            if self.phi is not None: df[f'phi{i}'] = self.phi[i]
            if self.dphi is not None: df[f'dphi{i}'] = self.dphi[i]
            if self.dV is not None: df[f'dV{i}'] = self.dV[i]

        return df

    def stats_df(self):

        """
        Dataframe of quantities and statistics derived from the trajectory.
        """

        cols = ['id', 's', 'V0', 'N', 'from0', 'N_e', 'A_t', 'n_t', 'A_s', 'n_s', 'A_iso', 'n_iso', 'r', 'sim_time', 'nfev', 'forgets', 'upscale', 'fails:inwardness', 'fails:e-folds', 'fails:total', 'fails:exception', 'fails:overall', 'success:inwardness', 'success:convergence', 'success:e-folds']
        cols = np.append(cols, [f'bowl{i}' for i in range(self.N)])
        df = pd.DataFrame(columns = cols)

        df['id'] = [hex(id(self))]
        df['s'] = [self.s()]
        df['V0'] = [self.V0()]
        df['N'] = [self.N]
        if self.phi is not None: df['from0'] = np.linalg.norm(self.phi[:,0])/self.s()
        if self.nE is not None: df['N_e'] = self.nE[-1]
        df['A_t'] = [self.A_t]
        df['n_t'] = [self.n_t]
        df['A_s'] = [self.A_s]
        df['n_s'] = [self.n_s]
        df['A_iso'] = [self.A_iso]
        df['n_iso'] = [self.n_iso]
        df['r'] = [self.r]
        df['sim_time'] = self.simtime
        df['nfev'] = self.nfev
        df['forgets'] = self.forgets
        df['upscale'] = self.upscale
        df['fails:inwardness'] = self.fails['Inwardness']
        df['fails:e-folds'] = self.fails['e-Folds']
        df['fails:total'] = self.fails['Total']
        df['fails:exception'] = self.fails['Exception']
        df['fails:overall'] = self.fails['Overall']
        df['success:inwardness'] = self.success['Inwardness']
        df['success:convergence'] = self.success['Convergence']
        df['success:e-folds'] = self.success['e-Folds']
        for i in range(self.N):
            df[f'bowl{i}'] = self.bowl[i]

        return df



def getVSkew(cov):

    """
    Uses covariance matrix to sample random deviations.
    - cov: covariance matrix
    """

    L = sp.linalg.cholesky(cov + noise*np.identity(len(cov)), lower=True)
    y = np.random.normal(0, 1, len(cov))
    mush = L @ y

    return mush



def quarter(m, definedCut):

    """
    Subdivides a matrix into four blocks.
    - m: matrix to be subdivided
    - definedCut: row/column at which to subdivide
    """

    return m[:definedCut, :definedCut], m[:definedCut, definedCut:], m[definedCut:, :definedCut], m[definedCut:, definedCut:]



def beefItUp(m, beef):

    """
    Pads ("beefs up") a matrix with rows and columns of zeros.
    """

    m = np.concatenate((m, np.zeros((beef, len(m)))), axis=0)
    m = np.concatenate((m, np.zeros((len(m), beef))), axis=1)

    return m



def sampleVData(b, phiPoint, d1=(-1,), d2=(-1,), include=True):

    """
    Given a BStuff object, samples V or its derivatives at some point in phi-space.
    - b: BStuff simulation object
    - phiPoint: point(s) at which to sample data
    - d1: [list of int(s) from -1 to N-1] first derivative index(s)
    - d2: [list of int(s) from -1 to N-1] second derivative index(s)
    - include: modify b to include sampled data point
    """

    d1, d2 = np.array(d1), np.array(d2)
    if len(d1) != len(d2): d2 = np.append( d2, [-1]*(len(d1)-len(d2)) ) if len(d1) > len(d2) else print("Supply first derivatives, too!")
    if max(np.append(d1, d2) > b.N-1): print("There aren't that many dimensions!")
    if (len(b.vcov) + len(d1) >= maxCovLen) and include:
        forgetEarlies(b,cut=1/2)
    elif len(b.vcov) + len(d1) >= maxCovLen: print("No more space - set include True to forget"); return

    phiList, vStuff, infoVec, s, V0, N = b.phiStuff[0], b.vStuff, b.infoVec, b.s(), b.V0(), b.N
    vcov, beefCov, vcov_L, beefCov_L = b.vcov, b.beefCov, b.vcov_L, b.beefCov_L
    M, beefM = b.M, b.beefM
    phiList = np.concatenate((phiList, np.transpose([phiPoint])), axis=1)
    infoPart = np.array([[infoVec[0,-1]+1]*len(d1), d1, d2])
    infoVec = np.concatenate((infoVec, infoPart), axis=1)

    bar = len(vcov)
    beefCov, beefM = lightning_fast_beefupdate(s, phiList, infoVec, beefCov, beefM, bar)
    vcov = beefCov[:len(infoVec[0]), :len(infoVec[0])]
    M = beefM[:len(infoVec[0]), :len(infoVec[0])]
    OO, ON, NO, NN = quarter(vcov, -len(d1))

    halfSandwich = sp.linalg.solve_triangular(vcov_L, ON, lower = True)
    gammaC = NN - (np.transpose(halfSandwich) @ halfSandwich) ## Cleverly, this is NO @ OOI @ ON
    skew = getVSkew(gammaC)*V0

    mu = np.transpose(halfSandwich) @ sp.linalg.solve_triangular(vcov_L, vStuff, lower = True) ## Cleverly again, this is NO @ OOI @ xO
    vNew = mu + skew

    if include:
        b.phiStuff = np.concatenate((b.phiStuff, np.array([phiPoint, np.zeros(N)]).reshape(2,N,1)), axis=2)
        b.vStuff = np.append(b.vStuff, mu + skew)
        b.vFull = np.append(b.vFull, mu + skew) if b.vFull is not None else b.vStuff
        b.infoVec = infoVec
        b.infoFull = np.concatenate((b.infoFull, infoPart), axis=1) if b.infoFull is not None else b.infoVec
        b.beefCov = beefCov
        b.vcov = vcov
        b.beefCov_L = updateCholesky(beefCov_L, vcov, bar)
        b.vcov_L = b.beefCov_L[:len(infoVec[0]), :len(infoVec[0])]
        b.nfev = b.nfev + 1
        b.M = M
        b.beefM = beefM

    return vNew



def sampleVConcavity(b, points=10, include=True, annotate=True):

    """
    Samples all second derivatives of V at evenly-spaced points along the trajectory.
    - b: BStuff simulation object
    - points: number of points at which to sample second derivatives
    - include: modify b to include sampled data point
    - annotate: print progress messages
    """

    nE, N = b.nE, b.N
    sparsenE = np.linspace(nE[0], nE[-1], points)
    sparsephi = b.phiFunc(sparsenE)

    ddV = np.zeros((N,N,points))
    offdiags = np.array(list(itertools.combinations(np.arange(N), r=2)))

    for p in range(points):
        if annotate: print(f"Simulating d2V/d(phi)2 at N_e = {np.round(sparsenE[p],3)}...")
        if N>1:
            sampleVData(b, sparsephi[:,p], d1=offdiags[:,0], d2=offdiags[:,1])
            ddV[offdiags[:,0], offdiags[:,1], p] = b.vStuff[-len(offdiags):]
            ddV[offdiags[:,1], offdiags[:,0], p] = ddV[offdiags[:,0], offdiags[:,1], p]
        for i in range(N):
            sampleVData(b, sparsephi[:,p], d1=(i,), d2=(i,))
            ddV[i,i,p] = b.vStuff[-1]

    ddVfunc = np.empty((N,N), dtype=object)
    ddVfull = np.zeros((N,N,len(nE)))
    for i in range(N):
        for j in range(N):
            ddVfunc[i,j] = CS(sparsenE, ddV[i,j])
            ddVfull[i,j] = ddVfunc[i,j](nE)

    if include: b.ddV = ddVfull

    return ddVfull



def unitize(b, redoVCOV=False):

    """
    Modifies a BStuff object with dimensionless inflaton field and potential values to be dimensionful.
    - b: BStuff simulation object
    - redoVCOV: reinitialize the covariance matrix
    """

    s, V0 = b.params[:2]

    if isinstance(b.phi, np.ndarray): b.phi = b.phi*s
    if isinstance(b.dphi, np.ndarray): b.dphi = b.dphi*s
    if isinstance(b.V, np.ndarray): b.V = b.V*V0
    if isinstance(b.dV, np.ndarray): b.dV = b.dV*V0/s
    if isinstance(b.ddV, np.ndarray): b.ddV = b.ddV*V0/s**2
    if isinstance(b.phiStuff, np.ndarray): b.phiStuff = b.phiStuff*s
    if isinstance(b.vStuff, np.ndarray):
        b.vStuff = b.vStuff*V0
        b.vStuff[b.infoVec[1] != -1] = b.vStuff[b.infoVec[1] != -1]/s
        b.vStuff[b.infoVec[2] != -1] = b.vStuff[b.infoVec[2] != -1]/s
    if isinstance(b.vFull, np.ndarray):
        b.vFull = b.vFull*V0
        b.vFull[b.infoFull[1] != -1] = b.vFull[b.infoFull[1] != -1]/s
        b.vFull[b.infoFull[2] != -1] = b.vFull[b.infoFull[2] != -1]/s
    if redoVCOV and isinstance(b.vcov, np.ndarray): b.vcov, b.M = lightning_fast_vcov((b.phiStuff.T[0]),b.infoVec,s)

    return b



def forceMin(s, V0, N):

    """
    Creates a BStuff object with a minimum at the origin.
    - s: coherence length
    - V0: inflationary energy scale
    - N: dimension of V
    """

    b = BStuff(params = (s,V0,N), N=N)

    ## Zeroeth and first derivatives forced to zero.
    b.phiStuff = np.zeros((2,N,1))
    b.infoVec = np.array([np.zeros((N+1)), np.arange(-1,N), -1*np.ones(N+1)], dtype=int)
    b.vStuff = np.zeros((N+1))
    b.vcov, b.M = lightning_fast_vcov((b.phiStuff.T[0]),b.infoVec,s)
    b.vcov_L = sp.linalg.cholesky(b.vcov + noise*np.identity(N+1), lower = True)
    b.beefCov[:N+1, :N+1] = b.vcov
    b.beefCov_L[:N+1, :N+1] = b.vcov_L
    b.beefM[:N+1, :N+1] = b.M

    ## Randomly simulate diagonal of second derivative matrix, then make them positive.
    sampleVData(b, np.zeros(N), d1=np.arange(N), d2=np.arange(N))
    b.bowl = np.abs(b.vStuff[-N:])
    b.vStuff[-N:] = b.bowl

    ## Set off-diagonals of second derivative matrix to zero.
    if N > 1:
        mixed = np.transpose(np.array(list(itertools.permutations(np.arange(N), 2))))
        infoPart = np.concatenate((np.array([np.ones(len(mixed[0]))]), mixed))
        b.infoVec = np.concatenate((b.infoVec, infoPart), axis=1)

        bar = len(b.vcov)
        b.beefCov, b.beefM = lightning_fast_beefupdate(s, b.phiStuff[0], b.infoVec, b.beefCov, b.beefM, bar)
        b.vcov = b.beefCov[:len(b.infoVec[0]), :len(b.infoVec[0])]
        b.M = b.beefM[:len(b.infoVec[0]), :len(b.infoVec[0])]
        b.beefCov_L = updateCholesky(b.beefCov_L, b.vcov, bar)
        b.vcov_L = b.beefCov_L[:len(b.infoVec[0]), :len(b.infoVec[0])]

        b.vStuff = np.append(b.vStuff, np.zeros(len(mixed[0])))

    b.nfev = 1

    return b



def testStart(b, phi0, s, nE_bounds):

    """
    Predicts whether an infant simulation has the potential to succeed (converge to the origin with enough inflation).
    - b: infant BStuff simulation object
    - phi0: starting location in phi-space
    - s: coherence length
    - nE_bounds: 2-tuple of allowable range of predicted number of e-folds
    """

    vNew = sampleVData(b, phi0, d1=np.arange(-1,b.N), include=True)
    V, dV = vNew[0], vNew[1:]
    phihat, dVhat = phi0/np.linalg.norm(phi0), dV/np.linalg.norm(dV)
    nEEst = 1/2*np.linalg.norm(phi0)*V/np.linalg.norm(dV)*s**2
    inwardness = np.dot(phihat, dVhat)
    enough_nE = (nEEst > nE_bounds[0] or nE_bounds[0] is None) and (nEEst < nE_bounds[1] or nE_bounds[1] is None)
    inward = inwardness > 0

    return enough_nE, inward, nEEst, inwardness



def adjustVData(b):

    """
    Organizes raw differential equation data into coherent lists.
    - b: BStuff simulation object
    """

    fullV = b.vFull[np.where(b.infoFull[1]==-1)[0]]
    fulldV = np.reshape(b.vFull[np.intersect1d(np.where(b.infoFull[1]!=-1)[0], np.where(b.infoFull[2]==-1)[0])], (b.N,len(fullV)), order='F')
    fullV, fulldV = fullV[1:], fulldV[:,1:]
    resorter = np.argsort(b.fulltimes)
    b.nE = b.fulltimes
    b.V, b.dV = fullV[resorter], fulldV[:,resorter] ## dummy step to use d/VFunc and sample at b.nE
    b.V, b.dV = b.VFunc(b.phiSol.t), b.dVFunc(b.phiSol.t)
    b.nE = b.phiSol.t

    return b



def salvageTraj(b):

    """
    Salvages quantities from differential equation if a critical failure occurs.
    - b: BStuff simulation object
    """

    if b.phiSol is None:
        if b.phiStuff is not None: b.phi, b.dphi = b.phiStuff[0], b.phiStuff[1]
        if b.fulltimes is not None: b.nE = b.fulltimes
        if b.vFull is not None and b.infoFull is not None:
            b.V = b.vFull[np.where(b.infoFull[1] == -1)[0]]
            b.dV = np.reshape(b.vFull[np.intersect1d(np.where(b.infoFull[1]!=-1)[0], np.where(b.infoFull[2]==-1)[0])], (b.N,len(b.V)), order='F')
            b.V, b.dV, b.phi, b.dphi = b.V[1:len(b.nE)+1], b.dV[:,1:len(b.nE)+1], b.phi[:,2:len(b.nE)+2], b.dphi[:,2:len(b.nE)+2]
    else:
        adjustVData(b)
        b.phi, b.dphi = b.phiSol.y[:b.N], b.phiSol.y[b.N:]

    try:
        unitize(b, redoVCOV=False)
        calculateQuantities(b, which='basic')
    except: pass

    return b



def calculateQuantities(b, which, **kwargs):

    """
    Computes various quantities from the inflaton field and potential.
    - b: BStuff simulation object
    - which: ('basic' or 'adv') which quantities to compute
    - psi_integrate (via kwargs): start and stop times for matter spectrum differential equation
    """

    if which=='basic':
        nEPivot = b.nE[-1] - nEBack4Pivot
        b.epsilon = np.linalg.norm(b.dphi, axis=0)**2/2
        b.H = np.sqrt(b.V/(3 - b.epsilon))
        b.a = np.exp(b.nE - nEPivot)
        Hfunc = CS(b.nE, b.H)
        b.upscale = Hfunc(nEPivot)/kStar
        b.k = b.a*b.H/b.upscale
        b.t = np.append([0], sp.integrate.cumulative_trapezoid(1/b.H, x = b.nE))
        b.inw = [np.dot(b.phi[:,i]/np.linalg.norm(b.phi[:,i]), b.dV[:,i]/np.linalg.norm(b.dV[:,i])) for i in range(len(b.dV[0]))]
    if which=='adv':
        psi_integrate = kwargs.get('psi_integrate')
        b.A_t, b.n_t, b.tensor_spec = getTensorSpectrum(b)
        b.coupler = getCouplingTensor(b)
        b.A_s, b.n_s, b.A_iso, b.n_iso = getMatterSpectrum(b, nE_integrate=psi_integrate)
        b.r = getSpectralRatio(b)

    return b

def forgetPrincipled(b):

    """
    Principled forgetting - forgets earlier data by compressing covariance matrix with matrix constructed from eigenvectors.
    - b: BStuff simulation object
    """

    ## Update vcov, vcov_L, beefCov, beefCov_L, vStuff, infoVec
    vcov = b.vcov
    M = b.M
    eigvals, eigvecs = sp.linalg.eigh(M, vcov+noise*np.identity(len(vcov)))
    minval = 0.999999*np.sum(eigvals)
    eigvals = np.flipud(eigvals)
    eigvecs = eigvecs.T
    eigvecs = np.flipud(eigvecs)
    A = np.array(np.matrix(eigvecs[0]))
    numvals = 1
    Avals = eigvals[0]
    while np.sum(Avals)<minval:
        A = np.vstack((A,eigvecs[numvals]))
        Avals += eigvals[numvals]
        numvals += 1
    newvcov = A@vcov@A.T
    newM = A@M@A.T

    b.vcov = newvcov
    b.M = newM
    b.vcov_L = sp.linalg.cholesky(b.vcov + 0.01*np.identity(len(b.vcov)),lower=True)
    b.beefCov, b.beefCov_L = np.zeros((2,maxCovLen,maxCovLen))
    b.beefCov[:len(b.vcov),:len(b.vcov)] = b.vcov
    b.beefCov_L[:len(b.vcov),:len(b.vcov)] = b.vcov_L
    b.beefM = np.zeros((maxCovLen,maxCovLen))
    b.beefM[:len(b.M),:len(b.M)] = b.M
    b.vStuff = A@b.vStuff
    b.infoVec = A@(b.infoVec.T)
    b.A = A

    b.forgets += 1

    return b

def forgetEarlies(b, cut=1/2):

    """
    Unprincipled forgetting - forgets earlier data by truncating covariance matrix.
    - b: BStuff simulation object
    - cut: fraction of covariance matrix to remove
    """

    ## Update vcov, vcov_L, beefCov, beefCov_L, phiStuff, vStuff, infoVec, NOT nfev
    vcov = b.vcov
    seedNum = b.N**2 + b.N + 1
    cut = int((len(vcov)-seedNum)*cut)
    newvcov = np.zeros((len(vcov)-cut, len(vcov)-cut))
    newvcov[:seedNum, :seedNum] = vcov[:seedNum, :seedNum]
    newvcov[seedNum:, seedNum:] = vcov[(seedNum+cut):, (seedNum+cut):]
    newvcov[:seedNum, seedNum:] = vcov[:seedNum, (seedNum+cut):]
    newvcov[seedNum:, :seedNum] = np.transpose(newvcov[:seedNum, seedNum:])

    b.vcov = newvcov
    b.vcov_L = sp.linalg.cholesky(b.vcov + noise*np.identity(len(b.vcov)), lower=True)
    b.beefCov, b.beefCov_L = np.zeros((2,maxCovLen,maxCovLen))
    b.beefCov[:len(b.vcov),:len(b.vcov)] = b.vcov
    b.beefCov_L[:len(b.vcov),:len(b.vcov)] = b.vcov_L
    b.vStuff = np.append(b.vStuff[:seedNum], b.vStuff[(seedNum+cut):])
    b.infoVec = np.concatenate((b.infoVec[:,:seedNum], b.infoVec[:,(seedNum+cut):]), axis=1)
    ## phiStuff shouldn't have to be changed since it is referenced properly by infoVec

    b.forgets += 1

    return b



def annotater(section, **kwargs):

    """
    Prints progress messages.
    - section: subsection of bushwhack code being processed
    """

    if section=='start':
        print("Finding minimum...\n")
    if section=='min found':
        from0 = kwargs.get('from0')
        good = kwargs.get('good')
        fails = kwargs.get('fails')
        nEEst = kwargs.get('nEEst')
        inwardness = kwargs.get('inwardness')
        print(f"Minimum found: {good}")
        print(f"Fails: Inwardness - {fails['Inwardness']}, e-Folds - {fails['e-Folds']}, Total - {fails['Total']}")
        if good:
            print(f"|phi_0| = {from0}")
            print(f"Est. # e-folds: {nEEst}")
            print(f"Inwardness: {inwardness}\n")
    if section=='during':
        nE = kwargs.get('nE')
        phi = kwargs.get('phi')
        print(f"Integrating eqs. of motion at N_e = {np.round(nE,3)}, |phi| = {np.round(np.linalg.norm(phi),3)}...")
    if section=='exception':
        e = kwargs.get('e')
        print(f"\nIntegration failure, exception raised:\n{repr(e)}\n")
    if section=='solved':
        success = kwargs.get('success')
        calculable = kwargs.get('calculable')
        print("Integration finished.\n")
        print(f"Heading inward: {success['Inwardness']}")
        print(f"Ended near origin: {success['Convergence']}")
        print(f"Enough inflation: {success['e-Folds']}")
        print(f"Mode equations solvable: {calculable}")
        if calculable: print("\nComputing d2V/d(phi)2 and reformatting...\n")
    if section=='modes':
        print("\nSolving mode equations...\n")
    if section=='failure':
        print("Maximum allowable failures reached with given parameters.")

    return



def bushwhack(s = 30, V0 = 5e-9, N = 2, from0_bounds = (0.2,0.8), nE_bounds = (45,500), psi_integrate = (-3,None),
              fails_coef = 100, maxJumpFrac = 1/20, nE_buffer = 2, state = None, method = 'RK45', name = None, annotate = False):

    """
    Solves Klein-Gordon equations for inflaton field in a seeded RGF, then extracts spectral quantities.
    - s: coherence length
    - V0: inflationary energy scale
    - N: dimension of V
    - from0_bounds: range of allowed starting distances from the origin (phi = 0)
    - nE_bounds: allowable range of predicted number of e-folds
    - psi_integrate: start and stop times for matter spectrum differential equation
    - fails_coef: number of allowable failed seeds per dimension
    - maxJumpFrac: maximum fraction of s that can be jumped in one time step
    - nE_buffer: complements psi_integrate to modify matter spectrum differential equation
    - state: random state
    - method: differential equation solution method
    - name: name of simulation
    - annotate: print progress messages
    """

    start = time.time()
    if annotate: annotater('start')
    np.random.seed()
    if state is not None: np.random.set_state(state)
    state = np.random.get_state()

    phi0, from0, enough_nE, inward, fails_nE, fails_inward, fails, nEEst, inwardness = None, None, False, False, -1, -1, -1, 0, 0
    while not (enough_nE and inward) and fails < fails_coef*2**(N-1):

        if not enough_nE: fails_nE += 1
        if not inward: fails_inward += 1
        if not (enough_nE and inward): fails += 1
        b = forceMin(1, 1, N)
        b.fails['e-Folds'], b.fails['Inwardness'], b.fails['Total'] = fails_nE, fails_inward, fails
        b.bowl *= V0/s**2

        phi0 = np.random.normal(size = N)
        from0 = np.random.uniform(from0_bounds[0]**N, from0_bounds[1]**N)**(1/N)
        phi0 = from0*phi0/np.linalg.norm(phi0)
        enough_nE, inward, nEEst, inwardness = testStart(b, phi0, s, nE_bounds)

    if annotate: annotater('min found', from0=from0, good=(enough_nE and inward), fails=b.fails, nEEst=nEEst, inwardness=inwardness)
    dphi0 = -b.vStuff[-N:]/(b.vStuff[-N-1]*s**2)
    if (enough_nE and inward):

        def expertCosmo(nE, phiFlat):

            """
            Differential equation (formatted for solve_ivp).
            - nE: current e-fold
            - phiFlat: flatten array of [phi, dphi]
            """

            phi, dphi = np.reshape(phiFlat, (2,N))
            b.phiStuff[1,:,-1] = dphi
            b.fulltimes = np.append(b.fulltimes, [nE])

            vNew = sampleVData(b, phi, d1=np.arange(-1,N))
            V, dV = vNew[0], vNew[1:]
            phi, dphi, V, dV = phi*s, dphi*s, V*V0, dV*V0/s

            epsilon = np.dot(dphi,dphi)/2
            H = np.sqrt(V/(3 - epsilon))
            ddphi = (epsilon - 3)*dphi - 1/H**2*dV

            if b.nfev%20==0 and annotate: annotater('during', nE=nE, phi=phi)

            return np.append(dphi/s, ddphi/s).flatten()

        def stopCriterion(nE, phiFlat):

            """
            Terminal event function #1: epsilon = 1 (formatted for solve_ivp)
            - nE: current e-fold
            - phiFlat: flatten array of [phi, dphi]
            """

            phi, dphi = np.reshape(phiFlat, (2,N))
            dphi = dphi*s
            epsilon = np.dot(dphi,dphi)/2

            return epsilon - 1

        def failCriterion(nE, phiFlat):

            """
            Terminal event function #2: iota = 0 (formatted for solve_ivp)
            - nE: current e-fold
            - phiFlat: flatten array of [phi, dphi]
            """

            phi, dphi = np.reshape(phiFlat, (2,N))
            phi, dV = phi*s, b.vStuff[-N:]
            phihat, dVhat = phi/np.linalg.norm(phi), dV/np.linalg.norm(dV)

            return np.dot(phihat, dVhat)

        def submerging(nE, phiFlat):

            """
            Terminal event function #3: V = 0 (formatted for solve_ivp)
            - nE: current e-fold
            - phiFlat: flatten array of [phi, dphi]
            """

            return b.vStuff[-N-1]

        stopCriterion.terminal = True
        failCriterion.terminal = True
        submerging.terminal = True
        nEEnd = nEEst*10

        if True:
     #   try:
            b.phiSol = sp.integrate.solve_ivp(expertCosmo, [0,nEEnd], np.append(phi0,dphi0), events = (stopCriterion, failCriterion, submerging),
                                        method = method, max_step = nEEnd*maxJumpFrac, first_step = 0.8)
            b.phi, b.dphi = b.phiSol.y[:N], b.phiSol.y[N:]
            b.success = {"Inwardness": len(b.phiSol.t_events[1])==0, "Convergence": np.linalg.norm(b.phi[:,-1])<1/10,
                         "e-Folds": b.phiSol.t[-1]>nEBack4Pivot-psi_integrate[0]+nE_buffer}
  #      except Exception as e:
   #         if annotate: annotater('exception', e=e)
    #        b.fails['Exception'] = True

        calculable = all(val for val in b.success.values())
        if annotate: annotater('solved', success=b.success, calculable=calculable)
        if calculable:

            adjustVData(b)

            sampleVConcavity(b, annotate=annotate)

            b.params = (s,V0,N)
            unitize(b, redoVCOV=False)

            calculateQuantities(b, which='basic')

            if annotate: annotater('modes')
            calculateQuantities(b, which='adv', psi_integrate=psi_integrate)

        else:
            b.fails['Overall'], b.params = True, (s,V0,N)
            try: salvageTraj(b)
            except: pass
    else:
        b.fails['Overall'] = True
        b.params = (s,V0,N)
        if annotate: annotater("failure")
    b.state = np.array(state, dtype=object)
    b.simtime = time.time() - start
    if annotate: print("Finished.\n"); print(b)

    return b



###############################################################################
## -------------------------COSMOLOGICAL QUANTITIES------------------------- ##
###############################################################################



def kPowerFit(kVals, quantities):

    """
    Fits data (k, P_q(k)) to a power law A_q*(k/kStar)^(n_q).
    - kVals: k values
    - quantities: computed spectral values
    """

    p = np.polyfit(np.log(kVals), np.log(quantities), 1)
    A_q, n_q = np.exp(p[1] + p[0]*np.log(kStar)), p[0]

    return A_q, n_q



def getTensorSpectrum(b, points=5, plot=False):

    """
    Computes tensor power spectrum and fits it to a power law locally around kPivot.
    - b: BStuff simulation object
    - points: number of points around kPivot at which to sample spectrum
    - plot: show plot with computation
    """

    spectrum = (2/np.pi**2)*b.H**2
    specfunc = sp.interpolate.interp1d(b.k, spectrum)
    kVals = kPivot*np.logspace(-0.4*2, 0.4*2, points, base=np.e)
    A_t, n_t = kPowerFit(kVals, specfunc(kVals))

    if plot:
        plt.figure(figsize=(8,6))
        plt.loglog(b.k, spectrum, c='b', label='Spectrum')
        plt.scatter(kVals, specfunc(kVals), marker='+', c='k', label='Sampled Points')
        plt.loglog(b.k, A_t*(b.k/kStar)**n_t, c='g', label="Power Law Fit")
        plt.legend() and plt.title("Tensor Perturbation Spectrum")
        plt.xlabel(r"Wavenumber [$l_{Pl}^{-1}$]") and plt.ylabel(r"Amplitude Squared $[l_{Pl}^{-2}]$")

    return A_t, n_t, spectrum



def getCouplingTensor(b, cont=False, plot=False):

    """
    Computes coupling tensor at every point along the trajectory.
    - b: BStuff simulation object
    - cont: interpolate tensor elements
    - plot: show plot with computation
    """

    dV, ddV, dphi, nE, H, epsilon, N = b.dV, b.ddV, b.dphi, b.nE, b.H, b.epsilon, b.N
    C = np.empty((N,N,len(nE)))
    for i in range(N):
        for j in range(N):
            term1 = ddV[i,j]/H**2
            term2 = 1/H**2*dphi[i]*dV[j]
            term3 = 1/H**2*dphi[j]*dV[i]
            term4 = (3 - epsilon)*dphi[i]*dphi[j]
            C[i,j] = term1 + term2 + term3 + term4

    if cont:
        Cfuncs = np.empty((N,N), dtype=object)
        for i in range(N):
            for j in range(N):
                Cfuncs[i,j] = CS(nE, C[i,j])
        C = Cfuncs

    if plot:
        for i in range(N):
            for j in range(N):
                plt.semilogy(nE, np.abs(C[i,j]))
        plt.xlabel("Number of e-Folds") and plt.ylabel(r"Abs. Value $C[i,j]$")

    return C



def getMatterSpectrumPoint(b, k=None, nE_integrate=(-5,None), ICs=(0,1)):

    """
    Solve differential equation for matter spectrum at a specific wavenumber.
    - b: BStuff simulation object
    - k: wavenumber
    - nE_integrate: limits of integration
    - ICs: initial conditions of differential equation
    """

    nE, a, H, epsilon, N = b.nE, b.a, b.H, b.epsilon, b.N
    aHfunc, epsfunc = CS(nE, a*H), CS(nE, epsilon)
    C = getCouplingTensor(b, cont=True)

    nE_kfunc, kScaled, kCorrect = sp.interpolate.interp1d(b.k, b.nE), None, None
    if k is None: kScaled, kCorrect = kStar*b.upscale, kStar
    else: kScaled, kCorrect = k*b.upscale, k
    nECrossing = nE_kfunc(kCorrect)

    def callCoupler(nE):
        CVals = np.empty((N,N))
        for i in range(N):
            for j in range(N):
                CVals[i,j] = C[i,j](nE)
        return CVals

    def f(nE, psiFlat):
        psiStuff = np.reshape(psiFlat, (2,N,N))
        psi, dpsi = psiStuff[0], psiStuff[1]

        term1 = (epsfunc(nE) - 1)*dpsi
        term2 = (2 - (kScaled/aHfunc(nE))**2 - epsfunc(nE))*psi
        term3 = -1*callCoupler(nE) @ psi
        ddpsi = term1 + term2 + term3
        return np.append(dpsi.flatten(), ddpsi.flatten())

    nEStart = nE[0] if nE_integrate[0] is None else nECrossing + nE_integrate[0]
    nEEnd = nE[-1] if nE_integrate[1] is None else nECrossing + nE_integrate[1]

    psi0 = np.identity(N)*(ICs[0] + ICs[1])/np.sqrt(2*kCorrect)
    dpsi0 = np.identity(N)*kScaled*1j/(aHfunc(nEStart)*(ICs[0] - ICs[1])*np.sqrt(2*kCorrect))

    psiStuff = sp.integrate.solve_ivp(f, (nEStart, nEEnd), np.append(psi0.flatten(), dpsi0.flatten()), max_step=(nEEnd-nEStart)/500)
    psi = np.reshape(psiStuff.y[:N**2], (N,N,len(psiStuff.t)))

    fieldSpace = kCorrect/(2*np.pi**2)*(kScaled/a[-1])**2*(psi[:,:,-1] @ np.matrix(psi[:,:,-1]).H)
    omega = b.dphi[:,-1].flatten()
    omega = omega/np.linalg.norm(omega)
    sK = np.transpose(sp.linalg.null_space([omega]))
    adiabatic = np.real(np.sum(omega @ fieldSpace @ np.transpose(omega))/2)
    isocurvature = np.real(np.trace((sK @ fieldSpace @ np.transpose(sK))/2))
    #cross = np.real(np.sum((omega @ (fieldSpace + np.transpose(fieldSpace)) @ np.transpose(sK))/2))

    return adiabatic, isocurvature



def getMatterSpectrum(b, points=5, nE_integrate=(-5,None), ICs=(0,1), plot=False):

    """
    Compute matter spectrum at points log-spaced around kPivot.
    - b: BStuff simulation object
    - points: number of points around kPivot at which to sample spectrum
    - nE_integrate: limits of integration
    - ICs: initial conditions of differential equation
    - plot: show plot with computation
    """

    adiabatics, isocurvatures = [], []
    kVals = kPivot*np.logspace(-0.4*2, 0.4*2, points, base=np.e)
    for k in kVals:
        adiabatic, isocurvature = getMatterSpectrumPoint(b, k=k, nE_integrate=nE_integrate, ICs=ICs)
        adiabatics = np.append(adiabatics, adiabatic)
        isocurvatures = np.append(isocurvatures, isocurvature)

    A_s, n_s = kPowerFit(kVals, adiabatics)
    A_iso, n_iso = 0, 0
    if b.N>1: A_iso, n_iso = kPowerFit(kVals, isocurvatures)
    n_s, n_iso = n_s+1, n_iso+1

    if plot:
        plt.scatter(kVals, adiabatics, marker='+', c='k', label="Sampled Points")
        plt.loglog(b.k, A_s*(b.k/kStar)**(n_s - 1), label="Power Law Fit")
        plt.xlabel("Wavenumber") and plt.ylabel(r"$P_R(k)$") and plt.title("Adiabatic Curvature Power Spectrum")
        plt.legend()

    return A_s, n_s, A_iso, n_iso



def getSpectralRatio(b):

    """
    Compute tensor-to-scalar ratio.
    - b: BStuff simulation object
    """

    return b.A_t/b.A_s*(kStar/kPivot)**(1+b.n_t-b.n_s)



###############################################################################
## ---------------------------------PLOTTERS-------------------------------- ##
###############################################################################



def masterPlot(b):

    """
    Quickly visualizes several important features of a simulation in one figure.
    - b: BStuff simulation object
    """

    fig, axs = plt.subplots(2,4,figsize=(20,10))
    [axs[0,0].plot(b.nE, b.phi[i], label=rf"$\phi_{i}$") for i in range(b.N)]
    axs[0,0].set_title("Inflaton Values") and axs[0,0].legend()
    [axs[1,0].plot(b.nE, b.dphi[i], label=rf"$d\phi_{i}/dN_e$") for i in range(b.N)]
    axs[1,0].set_title("Inflaton Component Velocities") and axs[1,0].legend()
    axs[0,1].plot(b.nE, b.V) and axs[0,1].set_title("Inflaton Potential")
    [axs[1,1].plot(b.nE, b.dV[i], label=rf"$\partial V/\partial\phi_{i}$") for i in range(b.N)]
    axs[1,1].set_title("Inflaton Potential Gradient") and axs[1,1].legend()
    axs[0,2].plot(b.nE, b.H) and axs[0,2].set_title("Hubble Parameter")
    axs[1,2].plot(b.t, b.nE) and axs[1,2].set_title("Inflation")
    axs[0,3].loglog(b.k, (2/np.pi**2)*b.H**2, label="Spectrum")
    axs[0,3].loglog(b.k, b.A_t*(b.k/kStar)**b.n_t, label="Power Law Fit")
    axs[0,3].set_title("Tensor Perturbation Spectrum") and axs[0,3].legend()

    axs[1,3].loglog(b.k, b.A_s*(b.k/kStar)**(b.n_s - 1), label="Power Law Fit")
    axs[1,3].set_title("Matter Perturbation Spectrum")

    try:
        if b.name is not None: fig.suptitle(rf"Master Plot of b:{hex(id(b))} AKA $\it{b.name}$", fontsize=30)
    except:
        fig.suptitle(f"Master Plot of {hex(id(b))}", fontsize=30)
    return



def manyPlot(b, d=0, **kwargs):

    """
    Plots inflaton field as a function of N_e, componentwise.
    - b: BStuff simulation object
    - d: [0 or 1] N_e derivative number
    """

    plt.figure(**kwargs)

    if d==0: [plt.plot(b.nE, b.phi[i], label=rf"$\phi^{i}$") for i in range(b.N)]
    if d==1: [plt.plot(b.nE, b.dphi[i], label=rf"$d\phi^{i}/dN_e$") for i in range(b.N)]

    plt.xlabel("# of e-Folds", fontsize=20) and plt.legend(fontsize=15)
    plt.ylabel("Inflaton Component Velocities", fontsize=20) if d==1 else plt.ylabel("Inflaton Components", fontsize=20)

    return



def VPlot(b, d1=-1, d2=-1):

    """
    Plots inflaton potential as a function of N_e.
    - b: BStuff simulation object
    - d1: [int from -1 to N-1] first derivative number
    - d2: [int from -1 to N-1] second derivative number
    """

    if d1==-1: plt.plot(b.nE, b.V) and plt.ylabel("Inflaton Potential Value")
    if d1!=-1 and d2==-1: plt.plot(b.nE, b.dV[d1]) and plt.ylabel(f"Gradient of Potential: Axis {d1}")
    if d1!=-1 and d2!=-1: plt.plot(b.nE, b.ddV[d1,d2]) and plt.ylabel(f"Concavity of Potential: Axes {d1}, {d2}")

    plt.xlabel("# of e-Folds")

    return



def crossSectionPlot(b, i=0):

    """
    Plots V as a function of phi^(i).
    - b: BStuff simulation object
    - i: axis of cross section
    """

    if isinstance(i, int): plt.plot(b.phi[i], b.V, label = rf"$V(\phi_{i})$")
    else: [plt.plot(b.phi[dim], b.V, label = rf"$V(\phi_{dim})$") for dim in i]
    plt.xlabel("Inflaton Space") and plt.ylabel("Inflaton Potential Value") and plt.legend()

    return



def topoPlot(b, i=(0,1)):

    """
    Topographic plot (birds' eye view) of V as a function of two components of phi.
    - b: BStuff simulation object
    - i: [2-tuple] phi components to plot
    """

    if len(i)!=2 or b.N<max(i)+1: return "Cannot execute topoPlot with given data."

    plt.figure(dpi=300)
    plt.scatter(b.phi[i[0]], b.phi[i[1]], c=b.V, marker='+', cmap='inferno')
    cbar = plt.colorbar()
    cbar.set_label(r"$V(\phi)$", rotation=0)
    plt.axis("scaled") and plt.xlabel(rf"$\phi^{(i[0])}$") and plt.ylabel(rf"$\phi^{i[1]}$")
    plt.title("Topographic Plot")
    plt.show()

    return



def waterfallPlot(b, i=(0,1), ee=1, azim=115):

    """
    Three-dimensional plot of V as a function of two components of phi, including 3-d quivers to visualize gradient of V.
    - b: BStuff simulation object
    - i: [2-tuple] phi components to plot
    - ee: period of 3-d quivers
    - azim: azimuthal angle of view
    """

    if len(i)!=2 or b.N<max(i)+1: return "Cannot execute waterfallPlot with given data."

    s, V0, N = b.params
    fig = plt.figure(figsize=(10,10), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    x = b.phi[i[0]]/s
    y = b.phi[i[1]]/s
    z = b.V/V0
    u = -b.dV[i[1]]/(V0/s)
    v = b.dV[i[0]]/(V0/s)
    w = np.zeros(len(u))
    ax.set_box_aspect(( np.ptp(x), np.ptp(y), np.mean((np.ptp(x),np.ptp(y)))*2 ))
    ax.quiver(x[::ee], y[::ee], z[::ee], u[::ee], v[::ee], w[::ee], pivot='middle', arrow_length_ratio=0, color='k', alpha=0.5, length=0.1, normalize=True)
    sc = ax.scatter(x, y, z, c=z, cmap='inferno', alpha=0.9)
    cbar = plt.colorbar(sc, pad=0.0, shrink=0.75)
    cbar.ax.set_title(r"$V$ [$V_{\star}$]", fontsize=15)
    ax.set_xlabel(rf"$\phi^{i[0]}$ [$s$]", fontsize=15)
    ax.set_ylabel(rf"$\phi^{i[1]}$ [$s$]", fontsize=15)
    ax.set_zlabel(r"$V$ [$V_{\star}$]", fontsize=15)

    ax.view_init(azim=azim)

    return



def earthwormPlot(b, i=(0,1,2)):

    """
    Three-dimensional plot of three inflaton components.
    - b: BStuff simulation object
    - i: [3-tuple] phi components to plot
    """

    if b.N<3 or len(i)!=3: return "Cannot execute earthwormPlot with given input."

    ax = plt.figure().add_subplot(111, projection='3d')
    sc = ax.scatter(b.phi[i[0]], b.phi[i[1]], b.phi[i[2]], c = b.V)
    ax.set_xlabel(rf"$\phi_{i[0]}$") and ax.set_ylabel(rf"$\phi_{i[1]}$") and ax.set_zlabel(rf"$\phi_{i[2]}$")
    cbar = plt.colorbar(sc)
    cbar.set_label("Inflaton Potential")
    plt.title("3-Dimensional Inflaton Plot")

    return



def bowlPlot2d(s, V0, ppr=10, domain=(-1,1), proj_d=(1,3), figres=1):

    """
    Creates a BStuff object with N = 2 to seed a minimum and sample potential in a grid around the origin.
    - s: coherence length
    - V0: inflationary energy scale
    - ppr: number of points per row/column to sample
    - domain: phi domain in which to sample V
    - proj_d: [tuple of 1, 2, a/o 3] dimension(s) of projection in visual(s)
    - figres: figsize multiple of (8,8)
    """

    bowl = forceMin(s, V0, 2)
    vL, phiList = len(bowl.vStuff), np.linspace(s*domain[0],s*domain[1],ppr)
    newphiList = np.transpose(list(itertools.product(phiList, repeat=2)))
    figsize = figres*np.array((8,8))

    for i in range(ppr**2): sampleVData(bowl, newphiList[:,i], include=True)

    if 1 in proj_d:
        fig1, ax1 = plt.subplots(figsize=figsize)
        [ax1.plot(newphiList[1, ppr*i:ppr*(i+1)], bowl.vStuff[vL + ppr*i:vL + ppr*(i+1)]) for i in range(ppr)]
        ax1.set_xlabel(r"$\phi_1$", fontsize=20)
        ax1.set_ylabel(r"$V$", fontsize=20)
        plt.show()

    if 2 in proj_d:
        fig2, ax2 = plt.subplots(figsize=figsize, dpi=300)
        cf = ax2.contourf(phiList, phiList, bowl.vStuff[vL:].reshape((ppr,ppr)), ppr, cmap='inferno')
        cbar = plt.colorbar(cf)
        cbar.ax.set_title(r"$V$")
        ax2.set_xlabel(r"$\phi^{(0)}$", fontsize=20)
        ax2.set_ylabel(r"$\phi^{(1)}$", fontsize=20)
        plt.show()

    if 3 in proj_d:
        fig = plt.figure(figsize=figsize, dpi=300)
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(newphiList[0], newphiList[1], bowl.vStuff[vL:], c=bowl.vStuff[vL:], cmap='inferno')
        plt.colorbar(sc, pad=0.1)
        ax.set_xlabel(r"$phi^{(0)}$", fontsize=20)
        ax.set_ylabel(r"$\phi^{(1)}$", fontsize=20)
        ax.set_zlabel("V", fontsize=20)
        plt.show()

    return bowl



def bowlPlot1d(s, V0, points=100, domain=(-1,1)):

    """
    Creates a BStuff object of N = 1 to seed a minimum and sample potential in an interval around the origin.
    - s: coherence length
    - V0: inflationary energy scale
    - points: number of points to sample
    - domain: interval on which to sample V
    """

    bowl = forceMin(s, V0, 1)
    vL, newphiList = len(bowl.vStuff), np.array([np.linspace(s*domain[0],s*domain[1],points)])

    for i in range(points): sampleVData(bowl, newphiList[:,i], include=True)

    plt.plot(newphiList[0], bowl.vStuff[vL:]) and plt.xlabel(r"$\phi_0$") and plt.ylabel("V")

    return bowl



def vcovPlot(b, xrange=(None,None), yrange=(None,None)):

    """
    Heat map of covariance matrix.
    - b: BStuff simulation object
    - xrange: [2-tuple] rows to plot
    - yrange: [2-tuple] columns to plot
    """

    vcov, infoVec = b.vcov, b.infoVec
    vIndices = np.where(infoVec[1]==-1)[0]
    vvLocations = np.array(list(itertools.product(vIndices, repeat=2)))
    vcovV = vcov[vvLocations[:,0], vvLocations[:,1]].reshape(len(vIndices),len(vIndices))
    plt.matshow(vcovV[xrange[0]:xrange[1], yrange[0]:yrange[1]]) and plt.colorbar()

    return vcovV



def manySnakes(b, d=0, fps=20, name=None):

    """
    Animates inflaton components as a function of N_e.
    - b: BStuff simulation object
    - d: [0 or 1] N_e derivative number
    - fps: frames per second of animation
    - name: name of .gif file
    """

    fig, ax = plt.subplots()

    def animate(i):
        ax.clear()
        ax.set_xlabel('# of e-Folds')
        ylabel = 'Inflaton Components' if d==0 else 'Inflaton Component Velocities'
        ax.set_ylabel(ylabel)
        if d==0: [ax.plot(b.nE[:i], b.phi[dim,:i]) for dim in range(b.N)]
        if d==1: [ax.plot(b.nE[:i], b.dphi[dim,:i]) for dim in range(b.N)]

    anim = ani.FuncAnimation(fig, animate, frames=len(b.nE))
    writer = ani.PillowWriter(fps=fps)
    if name is None: name = hex(id(anim)) if b.name is None else b.name
    anim.save(f'manySnakes_{name}.gif', writer=writer)

    return



###############################################################################
## -----------------------------DATA COLLECTORS----------------------------- ##
###############################################################################



def explore3(testname='test', sVals=(30,), V0Vals=(5e-9,), NVals=(2,), from0_lbs=(0.5,), from0_ubs=(0.8,), sims=5):

    """
    Explore parameter space with parallelized simulations.
    - testname: name of folder to encompass simulation results
    - sVals: coherence length values
    - V0Vals: inflationary energy scale values
    - NVals: values of dimension of V
    - from0_lbs: values of lower bound of initial |phi|
    - from0_ubs: values of upper bound of initial |phi|
    - sims: number of simulations per parameter set to compute
    """

    today = date.today().strftime("%m-%d-%y")
    dirname = f"{testname}_{today}"
    path = os.path.join(os.getcwd(), dirname)
    os.mkdir(path)

    combos = list(itertools.product(sVals, V0Vals, NVals))
    combos = tackBounds(combos, from0_lbs, from0_ubs)

    for combo in combos:

        nesteddirname = f"params_{combo[0]}_{combo[1]}_{int(combo[2])}_{combo[3]}_{combo[4]}"
        nestedpath = os.path.join(path, nesteddirname)
        os.mkdir(nestedpath)
        allstats = []

        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = [executor.submit(interior, combo) for sim in range(sims)]

            for f in concurrent.futures.as_completed(results):
                name, traj, stats, state = f.result()

                if traj is not None: traj.to_csv(os.path.join(nestedpath, f"sim{name}_trajectory.csv"), index=False)
                if stats is not None: allstats = stats if len(allstats)==0 else pd.concat((allstats, stats))
                if state is not None: np.save(os.path.join(nestedpath, f"sim{name}_state"), state)

        if len(allstats)>0: allstats.to_csv(os.path.join(nestedpath, "major_quantities.csv"), index=False)

    shutil.make_archive(dirname, 'zip', path)

    return



def interior(combo):

    """
    Execute bushwhack with given parameter combination, then extract dataframes and random state.
    Formatted for concurrent.futures and protected with multiple try-except clauses.
    - combo: parameter set (s, V0, N, from0_lb, from0_ub)
    """

    name, stats, traj, state = hex(id(np.random.random())), None, None, None
    try:
        b = bushwhack(s = combo[0], V0 = combo[1], N = int(combo[2]), from0_bounds = (combo[3],combo[4]), fails_coef = 100, annotate=True)
        name = hex(id(b))
        try: stats = b.stats_df()
        except: pass
        try: traj = b.traj_df()
        except: pass
        try: state = b.state
        except: pass
    except Exception as e:
        print(f'Critical failure: {repr(e)}')

    return name, traj, stats, state



def tackBounds(combos, from0_lbs, from0_ubs):

    """
    Format combos to include from0 bounds.
    - combos: combinations without from0 bounds
    - from0_lbs: values of lower bound of initial |phi|
    - from0_ubs: values of upper bound of initial |phi|
    """

    fullcombos = []
    for i in range(len(from0_lbs)): fullcombos = combos if i==0 else np.concatenate((fullcombos, combos))
    fullcombos = np.concatenate((fullcombos, np.zeros((len(fullcombos),2))), axis=1)

    for i in range(len(from0_lbs)):
        fullcombos[i*len(combos):(i+1)*len(combos),-2] = np.repeat(from0_lbs[i], len(combos))
        fullcombos[i*len(combos):(i+1)*len(combos),-1] = np.repeat(from0_ubs[i], len(combos))

    return fullcombos



###############################################################################
## -----------------MISCELLANEOUS HASTILY-CREATED PLOTTERS------------------ ##
###############################################################################



def vcovTest1(b, start=1/20, end=1, step=1, plot=True):

    iV = np.where(b.infoFull[1]==-1)[0][1:]
    info = b.infoFull[:,iV]
    iPhi = np.array(info[0], dtype=np.int64)
    phi = b.phiStuff[0,:,iPhi]
    phi = np.vstack((phi, np.zeros(b.N)))
    infoPart = np.transpose([[np.max(info[0])+1,-1,-1]])
    info = np.concatenate((info,infoPart), axis=1)
    traces = np.array([])

    start = np.round(len(phi)*start).astype(np.int64)
    end = np.round(len(phi)*end).astype(np.int64)
    if start==0: start = 2
    if end==1: end = end-1
    kept = np.append(np.arange(start, end, step), [len(phi)])
    for i in kept:

        phiTemp = phi[-i:]
        infoTemp = copy.copy(info[:, -i:])
        infoTemp[0] = infoTemp[0] - np.min(infoTemp[0])

        vcov, M = lightning_fast_vcov(phiTemp,infoTemp,b.s())
        vcov_L = sp.linalg.cholesky(vcov[:-1,:-1] + noise*np.identity(len(vcov)-1), lower = True)

        OO, ON, NO, NN = quarter(vcov, -1)

        halfSandwich = sp.linalg.solve_triangular(vcov_L, ON, lower = True)
        gammaC = NN - (np.transpose(halfSandwich) @ halfSandwich) ## Cleverly, this is NO @ OOI @ ON

        traces = np.append(traces, np.trace(gammaC))

    keptFracs = kept/len(phi)
    if plot:
        plt.figure(dpi=300)
        plt.semilogy(kept/len(phi), traces/traces[-1]-1, 'g')
        plt.xlabel(r"Fraction of Rows/Cols Kept in $\Gamma$")
        plt.ylabel(r"Fractional Change in $\mathrm{trace}(\Gamma_{\mathscr{C}})$")
        plt.grid()

    return keptFracs, traces



def masterPlotThesis(b):

    fig, ax = plt.subplots(6,2, figsize=(25,35))

    [ax[0,0].plot(b.nE, b.phi[i]) for i in range(b.N)]
    ax[0,0].set_xlabel(r"$N_e$", fontsize=20)
    ax[0,0].set_ylabel(r"Components of $\vec{\phi}$", fontsize=20)
    ax[0,0].grid()

    [ax[0,1].plot(b.nE, b.dphi[i]) for i in range(b.N)]
    ax[0,1].set_xlabel(r"$N_e$", fontsize=20)
    ax[0,1].set_ylabel(r"Components of $d\vec{\phi}/dN_e$", fontsize=20)
    ax[0,1].grid()

    ax[1,0].plot(b.nE, b.V)
    ax[1,0].set_xlabel(r"$N_e$", fontsize=20)
    ax[1,0].set_ylabel(r"$V$", fontsize=20)
    ax[1,0].grid()

    [ax[1,1].plot(b.nE, b.dV[i]) for i in range(b.N)]
    ax[1,1].set_xlabel(r"$N_e$", fontsize=20)
    ax[1,1].set_ylabel(r"Components of $\nabla_{\phi}V$", fontsize=20)
    ax[1,1].grid()

    ax[2,0].semilogy(b.nE, b.epsilon)
    ax[2,0].set_xlabel(r"$N_e$", fontsize=20)
    ax[2,0].set_ylabel(r"$\epsilon$", fontsize=20)
    ax[2,0].grid()

    ax[2,1].plot(b.nE, b.inw)
    ax[2,1].set_xlabel(r"$N_e$", fontsize=20)
    ax[2,1].set_ylabel(r"$\iota$ (Inwardness)", fontsize=20)
    ax[2,1].grid()

    ax[3,0].plot(b.t, b.nE)
    ax[3,0].set_xlabel(r"Cosmic Time $t$", fontsize=20)
    ax[3,0].set_ylabel(r"$N_e$", fontsize=20)
    ax[3,0].grid()

    for i in range(b.N):
        for j in range(b.N): ax[3,1].plot(b.nE, b.ddV[i,j])
    ax[3,1].set_xlabel(r"$N_e$", fontsize=20)
    ax[3,1].set_ylabel(r"Second Derivatives of $V$", fontsize=20)
    ax[3,1].grid()

    ax[4,0].scatter(b.phi[0], b.phi[1], marker='+', c=b.V)
    ax[4,0].set_xlabel(r"$\phi^{(0)}$", fontsize=20)
    ax[4,0].set_ylabel(r"$\phi^{(1)}$", fontsize=20)
    ax[4,0].grid()

    ax[4,1].plot(b.nE, b.H)
    ax[4,1].set_xlabel(r"$N_e$", fontsize=20)
    ax[4,1].set_ylabel(r"$H$", fontsize=20)
    ax[4,1].grid()

    ax[5,0].loglog(b.k, b.A_s*(b.k/kStar)**(b.n_s - 1), c='k', label='Power Law Fit')
    ax[5,0].set_xlabel(r'$k$', fontsize=20)
    ax[5,0].set_ylabel(r'$\mathcal{P}_S(k)$', fontsize=20)
    ax[5,0].grid()

    ax[5,1].loglog(b.k, (2/np.pi**2)*b.H**2, label="Spectrum")
    ax[5,1].loglog(b.k, b.A_t*(b.k/kStar)**b.n_t, c='k', label="Power Law Fit")
    ax[5,1].set_xlabel(r'$k$', fontsize=20)
    ax[5,1].set_ylabel(r'$\mathcal{P}_T(k)$', fontsize=20)
    ax[5,1].legend(fontsize=15)
    ax[5,1].grid()

    #plt.suptitle(rf"$(s,V_0,N)=({b.s()},{b.V0()},{b.N})$", fontsize=30)

    return



def masterPlotPaper(b):

    fig, ax = plt.subplots(2, 2, sharex='col', figsize=(6,6), dpi=300)
    fig.subplots_adjust(wspace=0.5)

    ax[0,0].semilogy(b.nE, b.epsilon)
    ax[0,0].set_ylabel(r"$\epsilon$")
    ax[0,0].grid()

    ax[1,0].plot(b.nE, b.inw)
    ax[1,0].set_xlabel(r"$N_e$") and ax[1,0].set_ylabel(r"$\iota$")
    ax[1,0].grid()

    k = b.k[b.k != 0]#np.ma.masked_array(b.k)
    H = b.H[b.k != 0]
    ax[0,1].loglog(k, b.A_s*(k/kStar)**(b.n_s - 1), c='k', label='Power Law Fit')
    ax[0,1].set_ylabel(r'$\mathscr{P}_S$')
    ax[0,1].grid()

    ax[1,1].loglog(k, (2/np.pi**2)*H**2, label="Spectrum")
    ax[1,1].loglog(k, b.A_t*(k/kStar)**b.n_t, c='k', label="Power Law Fit")
    ax[1,1].set_xlabel(r'$k$')
    ax[1,1].set_ylabel(r'$\mathscr{P}_T$')
    ax[1,1].legend()
    ax[1,1].grid()

    return



def quantifySpeedup(sims=10, mCLs=[200,400,600,800,1000], output=None, plot=True):

    global maxCovLen

    if output is None:
        times = np.zeros((len(mCLs)+1,sims))
        A_s = np.zeros((len(mCLs)+1,sims))

        for i in range(sims):
            maxCovLen = int(10**5)
            bSlow = None
            successful = False
            while not successful:
                bSlow = bushwhack(from0_bounds=(0.8,0.8), annotate=True)
                successful = np.all([val for val in bSlow.success.values()])
            times[-1,i] = bSlow.simtime
            A_s[-1,i] = bSlow.A_s
            for j in range(len(mCLs)):
                print(i, j)
                maxCovLen = mCLs[j]
                b = bushwhack(from0_bounds=(0.8,0.8), annotate=True, state = tuple(bSlow.state))
                times[j,i] = b.simtime
                A_s[j,i] = b.A_s
    else: times, A_s = output

    if plot:
        fig, ax = plt.subplots(1,2,figsize=(10,4), dpi=300)
        fig.subplots_adjust(wspace=0.3)
        [ax[0].loglog(times[-1], times[i]/times[-1], marker='+', linewidth=0, label=rf"Max $\Gamma$ Size: ${mCLs[i]}^2$") for i in range(len(mCLs))]
        ax[0].set_xlabel(r"$t_{\mathrm{sim}}$", fontsize=15)
        ax[0].set_ylabel(r"$\tilde{t}_{\mathrm{sim}}/t_{\mathrm{sim}}$", fontsize=15)
        ax[0].grid(which='both')

        [ax[1].loglog(times[i]/times[-1], np.abs(A_s[i]/A_s[-1]-1), marker='+', linewidth=0, label=rf"Max $\Gamma$ Size: ${mCLs[i]}^2$") for i in range(len(mCLs))]
        ax[1].set_xlabel(r"$\tilde{t}_{\mathrm{sim}}/t_{\mathrm{sim}}$", fontsize=15)
        ax[1].set_ylabel(r"$|\tilde{A}_S/A_S - 1|$", fontsize=15)
        ax[1].grid()
        ax[1].legend(loc='lower left')

    return times, A_s



def compileSuccessfulSims(sims=100, **kwargs):

    bs = np.empty((sims,), dtype=object)

    i = 0
    while i<sims:

        print(f"Attempting to store trajectory {i+1}/{sims}...")
        try:
            b = bushwhack(**kwargs)
            if np.all([val for val in b.success.values()]):
                print(f"Boom! Looking good after {b.simtime} seconds.")
                bs[i] = b
                i += 1
            else:
                print(f"Unsuccessful simulation! {b.success}")

        except:
            print("Oh no, an error! Let's try that again.")

    return bs



def compileTraceFuncs(bs, ppf=1000, range_params=(1/20,1,20), output=None, linewidth=0.5):

    start, end, step = range_params
    if output is None:
        traceFuncs = np.empty((len(bs),), dtype=object)
        traceVals = np.zeros((len(bs),ppf))
        keptFracVals = np.linspace(start, end, ppf)
        for i in range(len(bs)):
            print(f"Compiling traces from trajectory {i+1}/{len(bs)}...")
            keptFracs, traces = vcovTest1(bs[i], start=start, end=end, step=step, plot=False)
            try:
                traceFunc = sp.interpolate.CubicSpline(keptFracs, np.log(traces/traces[-1]))
                traceFuncs[i] = traceFunc
                traceVals[i] = np.exp(traceFunc(keptFracVals))-1
            except: None

    else: keptFracVals, traceVals, traceFuncs = output

    ## Plot

    lengths = np.array([len(b.phiStuff[0,0,:]) for b in bs])
    fig, ax = plt.subplots(dpi=500)

    normalize = mpl.colors.Normalize(vmin=lengths.min(), vmax=lengths.max())
    colormap = mpl.cm.viridis

    for j in range(len(bs)): ax.semilogy(keptFracVals[:-1], traceVals[j][:-1], color=colormap(normalize(lengths[j])), linewidth=linewidth)

    scalarmappaple = mpl.cm.ScalarMappable(norm=normalize, cmap=colormap)
    scalarmappaple.set_array(lengths)
    cbar = plt.colorbar(scalarmappaple)
    cbar.set_label(r"Size of $\Gamma$")

    ax.set_xlabel(r"Fraction of Rows/Cols Kept in $\tilde{\Gamma}$")
    ax.set_ylabel(r"Fractional Change in $\mathrm{trace}(\tilde{\Gamma}_{\mathscr{C}})$")
    ax.grid()

    return keptFracVals, traceVals, traceFuncs



## Minitest (feasible on your local computer).
##s = 30, V0 = 5e-9, N = 2, |phi| = {[0.6,0.7], [0.7,0.8]}, sims = 10
"""
if __name__=='__main__':
    
    explore3('minitest2', from0_lbs=np.arange(6,8)/10, from0_ubs=np.arange(7,9)/10, sims=10)
"""

## Megatest (appropriate for supercomputer).
## s = 30, V0 = 5e-9, N = {1, 2, 3}, |phi| = {[0.3,0.4], [0.4,0.5], ... , [1.2,1.3], [2.0,2.1]}, sims = 1000
"""
if __name__=='__main__':
    NVals = (1,2,3)
    from0_lbs = np.append(np.arange(3,13)/10, [2.0])
    from0_ubs = np.append(np.arange(4,14)/10, [2.1])
    explore3('megatest', NVals=NVals, from0_lbs=from0_lbs, from0_ubs=from0_ubs, sims=1000)
"""

###
###Temporary testing functions
###

## Temporary testing for single simulation
def plottest(tests):
    endtime = np.zeros(tests)
    for i in range(tests):
        start = time.time()
        n = bushwhack(s=30,V0=5e-9,N=2,from0_bounds=(0.8,1))
        try:
            #masterPlotPaper(n)
            #plt.show()
            waterfallPlot(b=n)
            plt.show()
        except: pass
        endtime[i] = time.time() - start
    return endtime

def exploretest(testname='test', sVals=(30,), V0Vals=(5e-9,), NVals=(5,), from0_lbs=(0.5,0.8), from0_ubs=(0.8,1), sims=5):
    """
    Explore parameter space with parallelized simulations.
    - testname: name of folder to encompass simulation results
    - sVals: coherence length values
    - V0Vals: inflationary energy scale values
    - NVals: values of dimension of V
    - from0_lbs: values of lower bound of initial |phi|
    - from0_ubs: values of upper bound of initial |phi|
    - sims: number of simulations per parameter set to compute
    """

    today = date.today().strftime("%m-%d-%y")
    dirname = f"{testname}_{today}"
    path = os.path.join(os.getcwd(), dirname)
    os.mkdir(path)

    combos = list(itertools.product(sVals, V0Vals, NVals))
    combos = tackBounds(combos, from0_lbs, from0_ubs)

    for combo in combos:

        nesteddirname = f"params_{combo[0]}_{combo[1]}_{int(combo[2])}_{combo[3]}_{combo[4]}"
        nestedpath = os.path.join(path, nesteddirname)
        os.mkdir(nestedpath)
        allstats = []

        results = interior(combo)
        name,traj,stats,state = hex(id(results)),results[1],results[2],results[3]
        if traj is not None: traj.to_csv(os.path.join(nestedpath, f"sim{name}_trajectory.csv"), index=False)
        if stats is not None: allstats = stats if len(allstats) == 0 else pd.concat((allstats, stats))
        if state is not None: np.save(os.path.join(nestedpath, f"sim{name}_state"), state)

        if len(allstats) > 0: allstats.to_csv(os.path.join(nestedpath, "major_quantities.csv"), index=False)

    shutil.make_archive(dirname, 'zip', path)

    return