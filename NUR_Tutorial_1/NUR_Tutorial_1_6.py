import numpy as np
import matplotlib.pyplot as plt

# 1. Binary divisibility test

# a) Function to check if n is divisible by 2^m using & operator
def divisibility_check_and(n, m):
    """Function to check if n is divisible by 2^m

    Parameters
    ----------
    n : int
        Number of interest
    m : int
        Exponent of 2
    """
    powers_of_2 = 2**np.linspace(0, m-1, m)
    mask = int(np.sum(powers_of_2))
    if n & mask == 0:
        print("Divisible!")
    else:
        print("Not divisible!")

# b) Function to check if n is divisible by 2^m using shifting operators
def divisibility_check_shift(n, m):
    """Function to check if n is divisible by 2^m

    Parameters
    ----------
    n : int
        Number of interest
    m : int
        Exponent of 2
    """
    mask = (n >> m) << m
    if n == mask:
        print("Divisible!")
    else:
        print("Not divisible!")

# c) 
if __name__ in "__main__":

    divisibility_check_and(n = 512, m = 8)
    divisibility_check_shift(n= 512, m = 8)

