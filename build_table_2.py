#Remez make table based on least square
#
# For x^p, -1<p<1, x_min<=x<=x_max find
#   a0,d[i],b[i] (i=1:k) so that a0+sum(d[i]/(1+x/b[i])) ~ x^p
#   within err
#
# Input
#   p   :  power                                         ,-1 <  p < 1
#   x_min :  inferior limit of the of the approximation  ,   >  0
#   x_max :  superior limit of the of the approximation  ,   >  x_min
#   k   :  number of term of the approximation           ,   >= 0
#
# Output
#   a0, d[i], b[i] : terms of the approximation
#   err            : maximum error of the best fit
#
# A) Will use my own least square algoritme
#    Will be slower => Numpy optimization first, then numba or cython 
#    
#    Change the coefficient : d = a/b 
# 


import numpy as np
import scipy as sp
from scipy.optimize import leastsq


type = np.float64


#make the function
def make_fx_lin(x_min,x_max,p,N):
    X = np.linspace(x_min, x_max, N, endpoint=True, dtype=type)
    Y = X**p
    return X,Y
    
def make_fx_log(x_min,x_max,p,N):
    X = np.logspace(x_min, x_max, N, endpoint=True, dtype=type)
    Y = X**p
    return X,Y
    
def make_fx_rand(x_min,x_max,p,N):
    X = np.random.rand(N, dtype=type)*(x_max-x_min)+x_min
    X = np.sort(X)
    Y = X**p
    return X,Y

def make_fx(x_min,x_max,p,N):
    X, Y  = make_fx_log(x_min,x_max,p,int(N/3))
    x1,y1 = make_fx_log(x_min,x_max,p,int(N/3))
    X = np.concatenate((X,x1))
    Y = np.concatenate((Y,y1))
    x1,y1 = make_fx_log(x_min,x_max,p,int(N/3))
    X = np.concatenate((X,x1))
    Y = np.concatenate((Y,y1))
    i = np.argsort(X)
    X = X[i];Y=Y[i]
    return X,Y
    
    
    
def f_approx(x, a0, a, coef):  #x vector no scalar
    y = np.ones(len(x))*a0
    for i in range(len(x)):
        y[i] += np.sum(a/(x[i]+coef))
    return y

def err_func(coef, x, y):
    a0,a = fit_a_d(x,y,coef)
    return (f_approx(x, a0, a, coef)-y)
    
def fit_a_d(X,Y,coef): #given the b coefficient, the a0, and d can be fited exactly using lin_alg
    k = len(coef)
    N = len(X)
    A = np.zeros((k+1,k+1))
    B = np.zeros( k+1 )
    A[0,0] = N; B[0] = np.sum(Y)
    for i in range(k):
        A[i+1,0] = np.sum(1/(X+coef[i])) 
        B[i+1]   = np.sum(Y/(X+coef[i])) 
        for j in range(k):
            A[i+1,j+1] = np.sum(1/((X+coef[i])*(X+coef[j]))) 
    A[0,1:] = A[1:,0]
    ad = np.linalg.solve(A, B)
    return ad[0],ad[1:]
    
    
    
def fit_b
    
    
    
    
    
def run_min(x_min,x_max,p,N,coef):
    X,Y = make_fx(x_min,x_max,p,N)
    fit_b(coef,X,Y)
    coef.sort()
    X,Y = make_fx(x_min,x_max,p,1000)
    error = max([np.max(err_func(coef,X,Y)),-np.min(err_func(coef,X,Y))])
    return success, error, coef
       
       
def main(p, k, x_min, x_max):
    #set initial guess


    f = 3*k
    tries = 0
    error = 1000
    while ((error >= 1e-15) and (tries < 20)):
        tries += 1
        success, error, coef = run_min(x_min,x_max,p,int(f),coef)
        f+2
        if(success < 5):f *= 1.2
        print(error)
    print(coef)

main(-0.125, 20, 0.001, 10)



