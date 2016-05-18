#Remez make table based on least square
#
# For x^p, -1<p<1, x_min<=x<=x_max find
#   a0,a[i],b[i] (i=1:k) so that a0+sum(a[i]/(a+b[i])) ~ x^p
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
# 1) pure python 
# 2) numpy
# 3) 


import numpy as np
import scipy as sp
from scipy.optimize import leastsq


type = np.float32


#make X,Y as a 2D array? 
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
        y[i] += np.sum(coef[1:k+1]/(x[i]+coef[k+1:]))
    return y

    
def err_func(coef, x, y):
    return f_approx(x, coef)-y
    
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

    
def main(p, k, x_min, x_max):
    #set initial guess
    coef = np.zeros(2*k+1,dtype=type)
    coef[0:k+1] = np.arange(k+1, dtype=type)+.5
    coef[k+1:]  = -np.arange(k, dtype=type)

    f = 3
    tries = 0
    error = 1000
    while ((error >= 1e-9) and (tries < 15)):
       tries +=1
       X,Y = make_fx(x_min,x_max,p,int(f*k))
       coef,success=leastsq(err_func,coef,args=(X,Y), Dfun = D_f_app)
       f *= 1.5
       if(success < 5):f *= 1.5
       X,Y = make_fx(x_min,x_max,p,1000)
       error = max([np.max(err_func(coef,X,Y)),-np.min(err_func(coef,X,Y))])
       print(error)
    print(coef)

main(0.125, 20, 0.001, 10)



