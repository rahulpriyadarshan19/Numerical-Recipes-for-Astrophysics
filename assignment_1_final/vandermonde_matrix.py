# Importing the necessary modules
import numpy as np
import matplotlib.pyplot as plt
from interpolation_routines.interpolation import interpolator
from numerical_solvers.equation_solver import numerical_solver
from timeit import timeit

# Rendering LaTeX style plots in matplotlib
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"

if __name__ in "__main__":

    # Initialising interpolator() and numerical_solver() objects
    interp = interpolator()
    solver = numerical_solver()

    # Loading the data
    data = np.loadtxt("Vandermonde.txt")
    x_data = data[:,0]
    y_data = data[:,1]
    N = len(x_data)

    # Creating the Vandermonde matrix
    vandermonde = np.zeros((N,N))
    for i in range(N):
        vandermonde[:,i] = x_data**i

    # LU decomposition that solves for the coefficients
    L_v, U_v, index_v = solver.LU_Crout(A = vandermonde)
    intermediate_y = solver.forward_sub(L = L_v, b = y_data, index = index_v)
    coefficients = solver.backward_sub(U = U_v, y = intermediate_y)
    with open("outputs/coefficients.txt", "w") as f:
        print("Coefficients: ", coefficients, file = f)

    # Defining the x_points at which the interpolated polynomial is computed
    x_points = np.array([])
    for i in range(N-1):
        x_interp = np.linspace(x_data[i], x_data[i+1], num = 50)
        if i == 0:
            x_points = np.append(x_points, x_interp)
        else:
            x_points = np.append(x_points, x_interp[1:])

    # Obtaining the value of the polynomial at each x in the x_point array
    polynomial = np.zeros((len(x_points),))
    for j in range(N):
        polynomial += coefficients[j]*x_points**j
        
    # Creating a temporary polynomial to find the difference in y-values
    y_polynomial = [polynomial[x_points == x][0] for x in x_data]
    y_polynomial = np.asarray(y_polynomial)

    # Plotting the data points, interpolated polynomial and their difference in heights
    fig, (ax_LU_1, ax_LU_1_diff) = plt.subplots(2, 1, figsize = (7, 7), sharex = True)
    ax_LU_1.scatter(x_data, y_data, color = "black", label = "Data points")
    ax_LU_1.plot(x_points, polynomial, color = "green", label = "Polynomial")
    ax_LU_1.set_ylabel("$y = \sum_{i=0}^N c_ix^i$")
    ax_LU_1.set_title("Data points and fitted polynomial (1 iteration of LU)")
    ax_LU_1.set_ylim(-200, 200)
    ax_LU_1.grid()
    ax_LU_1.legend()
    ax_LU_1_diff.plot(x_data, np.abs(y_polynomial - y_data), color = "red", label = "Difference")
    ax_LU_1_diff.set_ylabel("$|y(x) - y_i|$")
    ax_LU_1_diff.set_xlabel("Data points")
    ax_LU_1_diff.set_title("Difference between fitted polynomial and data points")
    ax_LU_1_diff.grid()
    ax_LU_1_diff.legend()
    fig.savefig("plots/lu_1_iteration.png", dpi = 300)
    plt.show()
    
    # Computing the polynomial obtained from Neville's algorithm
    polynomial_neville = [] 
    for x in x_points:
        y = interp.neville(x = x, x_data = x_data, y_data = y_data, order = 19)
        polynomial_neville.append(y)
    polynomial_neville = np.asarray(polynomial_neville)

    # Creating a temporary polynomial to find the difference
    y_polynomial_neville = [polynomial_neville[x_points == x][0] for x in x_data]
    y_polynomial_neville = np.asarray(y_polynomial_neville)

    # Plotting the data points, interpolated polynomial and their difference in heights
    fig, (ax_neville, ax_neville_diff) = plt.subplots(2, 1, figsize = (7,7), sharex = True)
    ax_neville.scatter(x_data, y_data, color = "black", label = "Data points")
    ax_neville.plot(x_points, polynomial_neville, color = "blue", label = "Neville Polynomial")
    ax_neville.set_ylabel("$y = \sum_{i=0}^N c_ix^i$")
    ax_neville.set_ylim(-200, 200)
    ax_neville.set_title("Data points and fitted polynomial (Neville's algorithm)")
    ax_neville.grid()
    ax_neville.legend()
    ax_neville_diff.plot(x_data, np.abs(y_polynomial_neville - y_data), color = "orange", label =  "Difference")
    ax_neville_diff.set_xlabel("Data points")
    ax_neville_diff.set_ylabel("$|y(x) - y_i|$")
    ax_neville_diff.set_title("Difference between fitted polynomial and data points")
    ax_neville_diff.grid()
    ax_neville_diff.legend()
    fig.savefig("plots/neville.png", dpi = 300)
    plt.show()

    # Iteratively doing LU decomposition 10 times for improved accuracy
    coeffs_iterated = coefficients
    for j in range(10):
        b_new = vandermonde@coeffs_iterated - y_data
        int_y = solver.forward_sub(L = L_v, b = b_new, index = index_v)
        delta_coeffs = solver.backward_sub(U = U_v, y = int_y)
        coeffs_iterated = coeffs_iterated - delta_coeffs
    
    polynomial_iterated = np.zeros((len(x_points),))
    for j in range(N):
        polynomial_iterated += coeffs_iterated[j]*x_points**j        

    # Creating a temporary polynomial to find the difference
    y_polynomial_iterated = [polynomial_iterated[x_points == x][0] for x in x_data]
    y_polynomial_iterated = np.asarray(y_polynomial_iterated)

    # Plotting the results for 10 LU iterations
    fig, (ax_LU_10, ax_LU_10_diff) = plt.subplots(2, 1, figsize = (7, 7), sharex = True)
    ax_LU_10.scatter(x_data, y_data, color = "black", label = "Data points")
    ax_LU_10.plot(x_points, polynomial_iterated, color = "lightslategray", label = "Polynomial")
    ax_LU_10.set_ylabel("$y = \sum_{i=0}^N c_ix^i$")
    ax_LU_10.set_title("Data points and fitted polynomial (10 iterations of LU)")
    ax_LU_10.set_ylim(-200, 200)
    ax_LU_10.grid()
    ax_LU_10.legend()
    ax_LU_10_diff.plot(x_data, np.abs(y_polynomial_iterated - y_data), color = "tomato", label = "Difference")
    ax_LU_10_diff.set_ylabel("$|y(x) - y_i|$")
    ax_LU_10_diff.set_xlabel("Data points")
    ax_LU_10_diff.set_title("Difference between fitted polynomial and data points")
    ax_LU_10_diff.grid()
    ax_LU_10_diff.legend()
    fig.savefig("plots/lu_10_iterations.png", dpi = 300)
    plt.show()

    # Plotting all the results in one plot along with their differences in y-values
    fig, (ax_all_poly, ax_all_diff) = plt.subplots(2, 1, figsize = (7,7), sharex = True)
    ax_all_poly.scatter(x_data, y_data, color = "black", label = "Data points")
    ax_all_poly.plot(x_points, polynomial, color = "crimson", linestyle = "-", label = "1 LU iteration")
    ax_all_poly.plot(x_points, polynomial_neville, color = "springgreen", linestyle = "--", label = "Neville")
    ax_all_poly.plot(x_points, polynomial_iterated, color = "teal", linestyle = "-.", label = "10 LU iterations")
    ax_all_poly.set_ylim(-200,200)
    ax_all_poly.set_ylabel("$y = \sum_{i=0}^N c_ix^i$")
    ax_all_poly.set_title("Data points and polynomials")
    ax_all_poly.grid()
    ax_all_poly.legend()
    ax_all_diff.plot(x_data, np.abs(y_data - y_polynomial),color = "crimson", linestyle = "-", label = "1 LU iteration")
    ax_all_diff.plot(x_data, np.abs(y_data - y_polynomial_neville), color = "springgreen", linestyle = "--", label = "Neville")
    ax_all_diff.plot(x_data, np.abs(y_data - y_polynomial_iterated), color = "teal", linestyle = "-.", label = "10 LU iterations")
    ax_all_diff.set_xlabel("Data points")
    ax_all_diff.set_ylabel("$|y(x) - y_i|$")
    ax_all_diff.set_title("Difference between fitted polynomial and data points")
    ax_all_diff.grid()
    ax_all_diff.legend()
    fig.savefig("plots/all_plots.png", dpi = 300)
    plt.show()

    # Execution times of the three cases
    setup_lu_1 = """
import numpy as np
import matplotlib.pyplot as plt
from interpolation_routines.interpolation import interpolator
from numerical_solvers.equation_solver import numerical_solver

interp = interpolator()
solver = numerical_solver()

data = np.loadtxt("Vandermonde.txt")
x_data = data[:,0]
y_data = data[:,1]
N = len(x_data)

vandermonde = np.zeros((N,N))
for i in range(N):
    vandermonde[:,i] = x_data**i
    """

    s_lu_1 = """
L_v, U_v, index_v = solver.LU_Crout(A = vandermonde)
intermediate_y = solver.forward_sub(L = L_v, b = y_data, index = index_v)
coefficients = solver.backward_sub(U = U_v, y = intermediate_y)
    """

    t_lu_1 = timeit(stmt = s_lu_1, setup = setup_lu_1, number = 100)
    with open("outputs/runtimes.txt", "w") as f:
        print("Time to run 1 LU iteration: ", t_lu_1, file = f)
    
    setup_neville = """
import numpy as np
import matplotlib.pyplot as plt
from interpolation_routines.interpolation import interpolator
from numerical_solvers.equation_solver import numerical_solver

interp = interpolator()
solver = numerical_solver()

data = np.loadtxt("Vandermonde.txt")
x_data = data[:,0]
y_data = data[:,1]
N = len(x_data)

x_points = np.array([])
for i in range(N-1):
    x_interp = np.linspace(x_data[i], x_data[i+1], num = 50)
    if i == 0:
        x_points = np.append(x_points, x_interp)
    else:
        x_points = np.append(x_points, x_interp[1:])

    """

    s_neville = """
polynomial_neville = [] 
for x in x_points:
    y = interp.neville(x = x, x_data = x_data, y_data = y_data, order = 19)
    polynomial_neville.append(y)
polynomial_neville = np.asarray(polynomial_neville)
    """

    t_neville = timeit(stmt = s_neville, setup = setup_neville, number = 100)
    with open("outputs/runtimes.txt", "a") as f:
        print("Time to run Neville's algorithm: ", t_neville, file = f)

    setup_lu_10 = """
import numpy as np
import matplotlib.pyplot as plt
from interpolation_routines.interpolation import interpolator
from numerical_solvers.equation_solver import numerical_solver

interp = interpolator()
solver = numerical_solver()

data = np.loadtxt("Vandermonde.txt")
x_data = data[:,0]
y_data = data[:,1]
N = len(x_data)

vandermonde = np.zeros((N,N))
for i in range(N):
    vandermonde[:,i] = x_data**i

L_v, U_v, index_v = solver.LU_Crout(A = vandermonde)
intermediate_y = solver.forward_sub(L = L_v, b = y_data, index = index_v)
coefficients = solver.backward_sub(U = U_v, y = intermediate_y)
    """

    s_lu_10 = """
coeffs_iterated = coefficients
for j in range(10):
    b_new = vandermonde@coeffs_iterated - y_data
    int_y = solver.forward_sub(L = L_v, b = b_new, index = index_v)
    delta_coeffs = solver.backward_sub(U = U_v, y = int_y)
    coeffs_iterated = coeffs_iterated - delta_coeffs
        """
    
    t_lu_10 = timeit(stmt = s_lu_10, setup = setup_lu_10, number = 100)
    with open("outputs/runtimes.txt", "a") as f:
        print("Time to run 10 LU iterations: ", t_lu_10, file = f)
