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
# 
# 3) Change the function to have exact fit for a and use least square for only b
#     Fix the b in the positive by fitting log(b)
#    
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
    X = np.logspace(np.log10(x_min), np.log10(x_max), N, endpoint=True, dtype=type)
    Y = X**p
    return X,Y
    
def make_fx_rand(x_min,x_max,p,N):
    X = np.random.rand(N)*(x_max-x_min)+x_min
    X = np.sort(X)
    Y = X**p
    return X,Y

def make_fx(x_min,x_max,p,N):
    X, Y  = make_fx_lin(x_min,x_max,p,int(2*N/5))
    x1,y1 = make_fx_log(x_min,x_max,p,int(2*N/5))
    X = np.concatenate((X,x1))
    Y = np.concatenate((Y,y1))
    x1,y1 = make_fx_rand(x_min,x_max,p,int(N/5))
    X = np.concatenate((X,x1))
    Y = np.concatenate((Y,y1))
    i = np.argsort(X)
    X = X[i];Y=Y[i]
    return X,Y
    
    
    
    
    
def f_approx(x, a0, a, coef):  #x vector no scalar
    y = np.ones(len(x))*a0
    for i in range(len(x)):
        y[i] += np.sum(a/(x[i]+np.exp(coef)))
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
        A[i+1,0] = np.sum(1/(X+np.exp(coef[i]))) 
        B[i+1]   = np.sum(Y/(X+np.exp(coef[i]))) 
        for j in range(k):
            A[i+1,j+1] = np.sum(1/((X+np.exp(coef[i]))*(X+np.exp(coef[j])))) 
    A[0,1:] = A[1:,0]
    ad = np.linalg.solve(A, B)
    return ad[0],ad[1:]


    
def run_min(x_min,x_max,p,N,coef):
    X,Y = make_fx(x_min,x_max,p,N)
    coef_s,success=leastsq(err_func,coef,args=(X,Y), ftol=1.49012e-15, xtol=1.49012e-15, gtol=0.0)
    coef_s.sort()
    coef = coef_s
    X,Y = make_fx(x_min,x_max,p,1000)
    error = max([np.max(err_func(coef,X,Y)),-np.min(err_func(coef,X,Y))])
    return success, error, coef
       
       
def main(p, k, x_min, x_max):
    #set initial guess
    coef = np.linspace(-3, 3, k, endpoint=True, dtype=type)
    """    
    coef = np.array([5.94310E-05,3.10273E-04,8.45352E-04,1.85275E-03,3.68658E-03,6.99170E-03,1.29314E-02,
                     2.35994E-02,4.27665E-02,7.72416E-02,1.39380E-01,2.51806E-01,4.56613E-01,
                     8.34344E-01,1.54676E+00,2.94646E+00,5.91355E+00,1.31973E+01,3.75994E+01,2.33465E+02])

    coef = np.array([0.000059431017396241,
                     0.000310272613931839,
                     0.000845351674577485,
                     0.001852754276188100,
                     0.003686577558114470,
                     0.006991696956540050,
                     0.012931374435359300,
                     0.023599351199586700,
                     0.042766512207597700,
                     0.077241640982089000,
                     0.139380258563480000,
                     0.251806164289540000,
                     0.456612653889345000,
                     0.834344091045067000,
                     1.546763627967430000,
                     2.946464782892110000,
                     5.913554056706340000,
                    13.197287864186400000,
                    37.599354578217300000,
                   233.464650567396000000])"""

    X,Y = make_fx(x_min,x_max,p,1000)
    error = max([np.max(err_func(coef,X,Y)),-np.min(err_func(coef,X,Y))])
    print(error)
    a0,a = fit_a_d(X,Y,coef)
    print(a0)
    for i in range(k):
        print(a[i],np.exp(coef[i]))
    #print(X)
    f = 3*k
    tries = 0
    coef_b = coef
    error_b = error
    while ((error >= 1e-9) and (tries < 20)):
        tries += 1
        success, error, coef = run_min(x_min,x_max,p,int(f),coef)
        if error <= error_b:
            error_b = error
            coef_b = coef
        f+2
        if(success < 5):f *= 1.2
        print(error)
    
    x,y = make_fx(x_min,x_max,p,1000)
    a0,a = fit_a_d(x,y,coef_b)
    print(" ")
    print(a0)
    for i in range(k):
        print(a[i],np.exp(coef_b[i]))

main(0.125, 20, 0.001, 10)



