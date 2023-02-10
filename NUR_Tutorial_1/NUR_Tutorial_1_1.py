import numpy as np
import matplotlib.pyplot as plt
from math import factorial
from timeit import timeit

# 1. Numerical errors
# a) Implementing the sinc function

# using np.sin
def sinc_sin(x):
    sincx = np.sin(x)/x
    return sincx

# using power series
def sinc_power(x, N):
    sincx = 0
    n = 0
    while n<=N:
        term = (-1)**n*(x**(2*n))/factorial(2*n+1)
        sincx += term
        n += 1
    return sincx

# error function
def sinc_error(x, N):
    error = sinc_sin(x) - sinc_power(x, N)
    return error

if __name__ in "__main__":
    # 1. Numerical errors
    # a) Truncation 
    
    # b) Comparing implementations
    print(sinc_sin(7))
    print(sinc_power(7, N = 2))
    errors = np.array([])
    no_of_terms = 10
    N = np.linspace(1, no_of_terms, no_of_terms)
    for n in N:
        error = sinc_error(7, N = no_of_terms)
        errors = np.append(errors, error)
    # plt.plot(N, errors)
    # plt.show()
    # Error oscillates and then slowly reduces

    # c) Discrepancy between single and double precision numbers 
    no_of_terms = 5
    x = np.float32(2.)
    y = np.float64(2.)
    errors_single = np.array([])
    errors_double = np.array([])
    for n in N:
        errors_single = np.append(errors_single, sinc_error(x, N = no_of_terms))
        errors_double = np.append(errors_double, sinc_error(y, N = no_of_terms))
    plt.plot(N, errors_single, label = "Single precision")
    plt.plot(N, errors_double, label = "Double precision")
    plt.legend()
    plt.show()
     

     