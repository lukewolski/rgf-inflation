##Principled Forgetting, multiple compression steps

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve, eigh

m = 1
n = 5000
sigma = 30
max = 8*sigma
mu = [0]*m
noise = 1e-8
forgets = 0
compressed = 0
toskip = 0
A = [[]]
lastA = [[]]
N = 0

def f(j,k):
    return np.e**(-((j-k)**2)/(2*sigma**2))

def update(L,x):
    n = len(L)
    PFFP = np.fromfunction(lambda i,j: np.sqrt(f(i,j)),(n,n),dtype=float)
    tempvcov = np.fromfunction(lambda i,j: f(i,j), (len(L),len(L)),dtype=float)
    tempvcov += np.identity(len(tempvcov))*noise
    w,v = eigh(PFFP,tempvcov)
    floor = 0.9999*sum(w)
    V = np.array((v[:,-1]).T)
    eigv = w[-1]
    ind = -2
    while eigv<floor:
        V = np.vstack((V,(v[:,ind]).T))
        eigv = eigv+w[ind]
        ind += -1
    newvcov = V@tempvcov@V.T
    newvcov = np.array(newvcov)
    newvcov = newvcov + noise*np.identity(int(np.sqrt(newvcov.size)))
    L,low = cho_factor(newvcov,lower=True)
    for i in range(len(L)):
        L[i][i+1:]=0
    x = V@x
    print(len(V))
    return L,x,V

vcov = np.zeros((m,m))
for j in range(m):
    for k in range(m):
        vcov[j][k] = f(j,k)
x = np.random.multivariate_normal(mu,vcov,size=1)
x = x.T
vcov = vcov + noise*np.identity(len(vcov))
finalx = x[0]
skipfirst = True
L,low = cho_factor(vcov,lower=True)
for i in range(len(L)):
    L[i][i+1:]=0
for i in range(n-m):
    #print(i+1)
    if len(L)>max:
        compressed = compressed + len(L)
        if forgets > 0:
            lastA = A
            lastA = np.atleast_2d(lastA)
        finalx = np.vstack((finalx,x[toskip:]))
        L,x,A = update(L,x)
        if forgets > 0:
            NewA = A@np.block([[np.array(lastA),np.zeros((len(lastA),N))],[np.zeros((N,len(lastA.T))),np.identity(N)]])
            A = NewA
            del(NewA)
        else:
            forgets = 1
        toskip = len(L)
        compressed = compressed - toskip
        N = 0
    PF = np.zeros((len(L)+compressed,1))
    for j in range(len(L)+compressed):
        PF[j] = f(j,len(L)+compressed)
    if forgets > 0:
        PF = np.vstack((A@(PF[:(len(A.T))]),PF[(len(A.T)):]))
    FP, FF = PF.T, np.array(f(len(L)+compressed,len(L)+compressed))+noise
    u = solve(L,PF,assume_a='lower triangular')
    z = np.array(np.sqrt(FF-(u.T@u)))
    if forgets == 0:
        newmean = FP@(cho_solve((L, low),x))
    else:
        tempfx = np.vstack((finalx,x[toskip:]))
        newmean = FP @ (cho_solve((L, low), np.atleast_2d(tempfx[compressed + 1:])))
    stdev = np.sqrt(FF - FP@(cho_solve((L, low),PF)))
    x = np.vstack((x,np.random.normal(newmean,stdev)))
    newL = np.block([[L,np.zeros((len(L),1))],[u.T,z]])
    L = newL
    N = N+1
finalx = np.vstack((finalx,x[toskip:]))
if skipfirst==True:
    finalx = finalx[1:]