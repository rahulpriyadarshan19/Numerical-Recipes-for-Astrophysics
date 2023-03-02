import numpy as np
import matplotlib.pyplot as plt
import copy

class numerical_solver():

    # Numerical solving using Gauss-Jordan elimination
    def gauss_jordan(self, A, b):
        """Solves a system of simultaneous linear equations using Gauss-Jordan elimination.

        Parameters
        ----------
        A : numpy.ndarray
            The matrix containing the coefficients of the linear system.
        b : numpy.ndarray
            The vector containing the constants of the linear system.

        Returns
        -------
        A_solved : numpy.ndarray
            The identity matrix
        b : numpy.ndarray
            The solution vector of the system of equations.
        """         
        # Attaching the coefficients matrix and constants vector together
        cols = np.shape(A)[1]
        b = np.array([b])
        b = np.transpose(b) 
        A = np.hstack((A, b))

        # Looping over only A and not the constants vector
        for i in range(cols):  

            # Finding the pivot for each column and its index
            col = A[:,i]
            pivot_indices = np.where(col != 0)
            pivot_indices = pivot_indices[0]

            # temp is the pivot index in the sliced column
            pivot_index_temp = np.argmax(np.abs(col[i:]))
            pivot = col[i:][pivot_index_temp]
            pivot_index = np.where(col == pivot)[0][0]

            # Moving the pivot row to the right position
            A[[i, pivot_index]] = A[[pivot_index, i]] 

            # Setting pivot to 1; dividing each row by pivot value
            A[i] = A[i]*(1/pivot)
            other_indices = np.delete(pivot_indices, np.where(pivot_indices == i))

            # Scaling the other elements in the column to zero by using the pivot
            for non_zero_index in other_indices:
                non_zero_row = A[non_zero_index]
                first_element = non_zero_row[i]
                A[non_zero_index] = non_zero_row - first_element*A[i]
        
        # Separating the identity matrix and the solutions' vector
        A_solved = A[:,:-1]
        b = A[:,-1]

        return A_solved, b


    # Improved Crout's algorithm for LU decomposition
    def LU_Crout(self, A):
        """LU decomposition for a non-singular, square matrix.  

        Parameters
        ----------
        A : numpy.ndarray
            Matrix of interest

        Returns
        -------
        L: numpy.ndarray
            Decomposed lower triangular matrix.
        U: numpy.ndarray
            Decomposed upper triangular matrix.
        index: numpy.ndarray
            Vector containing indices of pivot elements in the matrix. 
        """
        n = np.shape(A)[0]
        LU = copy.deepcopy(A)
        index = np.zeros((n,))

        # Looping over columns
        for k in range(n):
            # Finding the index of the largest element
            col = LU[:,k]
            i_max_temp = np.argmax(np.abs(col[k:]))
            pivot = col[k:][i_max_temp]
            i_max = np.where(col == pivot)[0][0]

            # Swapping rows
            if i_max != k:
                LU[[i_max, k]] = LU[[k, i_max]]
            index[k] = i_max

            # Setting the elements of LU
            for i in range(k+1, n):
                LU[i,k] = LU[i,k]/LU[k,k]
                for j in range(k+1, n):
                    LU[i,j] = LU[i,j] - LU[i,k]*LU[k,j]

        # Splitting the upper and lower triangular matrices
        U = np.triu(LU)
        L = np.tril(LU)
        for i in range(n):
            L[i,i] = 1

        return L, U, index
        

    # Function for forward substitution using L and b
    def forward_sub(self, L, b, index):
        """Forward substitution for a lower triangular matrix and a constant vector. 

        Parameters
        ----------
        L : numpy.ndarray
            Decomposed lower triangular matrix. 
        b : numpy.ndarray
            Vector of constants
        index : numpy.ndarray
            Vector containing indices of pivot elements in the original matrix.
            

        Returns
        -------
        y: numpy.ndarray
            Solution vector. 
        """
        n = len(b)
        # Swapping b based on the index array
        for i in range(n):
            b[[i, int(index[i])]] = b[[int(index[i]), i]]
        y = np.zeros((n,))
        y[0] = b[0]/L[0,0]
        for i in range(1,n):
            alpha_ij_y_j = np.array([L[i,j]*y[j] for j in range(0,i)])
            y[i] = (1/L[i,i])*(b[i] - np.sum(alpha_ij_y_j))

        return y

    # Function for backward substitution using U and y
    def backward_sub(self, U, y):
        """Backward substitution for an upper triangular matrix and a constant vector.

        Parameters
        ----------
        U : numpy.ndarray
            Decomposed upper triangular matrix.
        y : _type_
            Vector of constants.  

        Returns
        -------
        x: numpy.ndarray
            Solution vector.
        """
        n = len(y)
        x = np.zeros((n,))
        x[n-1] = y[n-1]/U[n-1,n-1]
        for i in range(n-1,-1,-1):
            beta_ij_x_j = np.array([U[i,j]*x[j] for j in range(i+1,n)])
            x[i] = (1/U[i,i])*(y[i] - np.sum(beta_ij_x_j))

        return x 


if __name__ in "__main__":

    A = np.array([[3,8,1,-12,-4], 
                  [1,0,0,-1,0], 
                  [4,4,3,-40,-3], 
                  [0,2,1,-3,-2], 
                  [0,1,0,-12,0]])
    A = A.astype("float64")
    b = np.array([2,0,1,0,0])
    b = b.astype("float64")
    print("A: ")
    print(A) 
    solver = numerical_solver()
    _, sol = solver.gauss_jordan(A, b)
    print("Solution vector after using Gauss-Jordan elimination: ")
    print(sol)

    L, U, index = solver.LU_Crout(A)
    y = solver.forward_sub(L, b, index)
    sol_LU = solver.backward_sub(U, y)
    print("Solution vector after using LU decomposition: ")
    print(sol_LU)
