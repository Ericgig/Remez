#Remez make table based on least square
#
# For x^p, -1<p<1, x_min<=x<=x_max find
#   a0,a[i],b[i] (i=1:k) so that a0+sum(a[i]/(x+b[i])) ~ x^p
#   within err
#
# Input
#   p   :  power                                         ,-1 <  p < 1
#   x_min :  inferior limit of the of the approximation  ,   >  0
#   x_max :  superior limit of the of the approximation  ,   >  x_min
#   k   :  number of term of the approximation           ,   >= 0
#
# Output
#   a0, a[i], b[i] : terms of the approximation
#   err            : maximum error of the best fit
#
# 1) scipy least-square
#     => Can't get double precision on fit, is sattisfied with the convergence too easily.
#      Want a maximum absolute error of at most 10e-15 on any point in the range
#      
# try 2) less points to fit but stricter condition
#     => Not enough control with least_square, converge before condition
# 3) Change the coefficient : d = a/b 
#    
#    
#     
# 

import numpy as np
import scipy as sp
from scipy.optimize import leastsq


type = np.float64


#make the function
def make_fx_lin(x_min,x_max,p,N):
    #X = x_min+np.arange(N,dtype=type)*((x_max-x_min)/(N-1))
    X = np.linspace(x_min, x_max, N, endpoint=True, dtype=type)
    Y = X**p
    return X,Y
    
def make_fx_log(x_min,x_max,p,N):
    #step = np.exp(np.log(x_max/x_min)/(N-1))
    #X = x_min*step**np.arange(N,dtype=type)
    X = np.logspace(x_min, x_max, N, endpoint=True, dtype=type)
    Y = X**p
    return X,Y

def make_fx(x_min,x_max,p,N):
    x1,y1 = make_fx_log(x_min,x_max,p,N/2)
    x2,y2 = make_fx_log(x_min,x_max,p,N/2)
    X = np.concatenate((x1,x2))
    Y = np.concatenate((y1,y2))
    return X,Y


def f_approx(x, coef):  #x vector no scalar
    k = (len(coef)-1)/2
    y = x*0+coef[0]
    for i in range(len(x)):
        y[i] += np.sum(coef[1:k+1]/(1+x[i]/coef[k+1:]))
    return y

    
def err_func(coef, x, y):
    return (f_approx(x, coef)-y)
    
def D_f_app(coef,x,y):
    k = int((len(coef)-1)/2)
    derr = []
    for p in x:
        d = np.zeros(2*k+1, dtype=type)
        d[0] = 1
        d[1:k+1] = 1/(1+p/coef[k+1:])
        d[k+1:]  = p*coef[1:k+1]/(p+coef[k+1:])**2
        derr.append(d)
    return derr

def sort_coef(coef):
    k = int((len(coef)-1)/2)
    a = coef[1:k+1]
    b = coef[k+1:]
    ind = np.argsort(b)
    coef[1:k+1] = a[ind]
    coef[ k+1:] = b[ind]
    
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
    coef = np.zeros(2*k+1,dtype=type)
    coef[0:k+1] = np.arange(k+1, dtype=type)+.5
    coef[k+1:]  = (np.arange(k, dtype=type)+1)/k

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

main(0.125, 20, 0.001, 10)



