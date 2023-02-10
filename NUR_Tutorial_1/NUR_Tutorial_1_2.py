import numpy as np
import matplotlib.pyplot as plt
from timeit import timeit


c = 3*10**8
# computes rs using c
def rs_slow(m):
    rs = 2*m/c**2
    return rs

# computes rs using c_inv 
def rs_fast(m):
    c_inv = 1/c
    rs = 2*m*c_inv**2
    return rs
    
# 2. Timing test
if __name__ in "__main__":
    M_bh = np.random.normal(loc = 10**6, scale = 10**5, size = 10000)
    print(M_bh)

    s1 = "rs_slow(M_bh)"
    s2 = "rs_fast(M_bh)"

    # comparing the run times
    t1 = timeit(s1, globals = globals())
    t2 = timeit(s2, globals = globals())
    print("Slower run time: ", t1)
    print("Faster run time: ", t2)
    
#new dditions to the code

def factorial(num):
    '''
    Documentation: FILL IN
    '''
    
    #ACCOUNT FOR SPECIAL CASES OF INPUT
    #E.G. NEGATIVES, FLOATS, 0,  ETC
    
    factorial=1
    
    for i in range(1,num+1):
       factorial*=i
    
    return factorial
