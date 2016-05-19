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
# 3) Will use my own least square algoritme
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
    
    
    
    
    
    
    
    
    
    
    
    

    
    
    
    
    
    
    
    
    
    

def f_approx(x, coef):  #x vector no scalar
    k = (len(coef)-1)/2
    y = x*0+coef[0]
    for i in range(len(x)):
        y[i] += np.sum(coef[1:k+1]/(x[i]+coef[k+1:]))
    return y

    
def err_func(coef, x, y):
    return (f_approx(x, coef)-y)
    
def D_f_app(coef,x,y):
    k = int((len(coef)-1)/2)
    derr = []
    for p in x:
        d = np.zeros(2*k+1, dtype=type)
        d[0] = 1
        d[1:k+1] = 1/(p+coef[k+1:])
        d[k+1:]  = -coef[1:k+1]/(p+coef[k+1:])**2
        derr.append(d)
    return derr

    
def sort_coef(coef):
    ind = np.argsort(coef[2])
    coef[2] = coef[2][ind]
    coef[1] = coef[1][ind]
    
def def_coef(k,p):
    a0 = 1
    d = -p*np.arange(k,dtype=type)/k  # the sign of the d are usualy opposed to the sign of p
    b = np.arange(k,dtype=type)       # positive so the there is no singularity in the range
    return [a0,d,b]
    
    
    
    
    
    
    
    
    
    
    
    
def run_min(x_min,x_max,p,N,coef):
    X,Y = make_fx(x_min,x_max,p,N)
    coef_s,success=leastsq(err_func,coef,args=(X,Y), Dfun = D_f_app,ftol=1.49012e-15, xtol=1.49012e-15, gtol=0.0)
    sort_coef(coef_s)
    coef = coef_s
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



