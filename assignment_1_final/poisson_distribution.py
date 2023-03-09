# Importing the necessary modules
import numpy as np

# Poisson distribution for a given lambda and k
def poisson(lambda_, k):
    """Generates the value of the Poisson distribution at a given lambda_ and k

    Parameters
    ----------
    lambda_ : float32
        Lambda
    k : float32
        k

    Returns
    -------
    prod: float32
        Value of Poisson distribution.
    """
    k_values = np.arange(0, k+1, 1)
    prod = 1
    if k == 0:
        prod = 1
    
    # Computes lambda/k iteratively and multiplies them together. 
    for i in range(1,k+1):
        prod *= lambda_/i
    return prod

if __name__ in "__main__":

    lambdas = np.array([1, 5, 3, 2.6, 101])
    lambdas = lambdas.astype("float32")
    ks = np.array([0, 10, 21, 40, 200])
    
    # Computing the Poisson distribution for the given values of lambda and k.  
    with open("\outputs\poisson.txt", "w") as f:
        print("Poisson distribution for the following values: ", file = f)
        for i in range(len(ks)):
            l = lambdas[i]
            k = ks[i]
            print(f"lambda = {l}, k = {k}: P_lambda(k) = {poisson(lambda_ = l, k = k)}", file = f)
    