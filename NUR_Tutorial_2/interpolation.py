import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["text.usetex"] = True

class interpolator:
   
    # Function to find the slope of the line joining two coordinates
    def slope(self, x_1, y_1, x_2, y_2):
        """Returns the slope of the line joining two points

        Parameters
        ----------
        x_1 : float64
            x-coordinate of the first point
        y_1 : float64
            y-coordinate of the first point
        x_2 : float64
            x-coordinate of the second point
        y_2 : float64
            y-coordinate of the second point

        Returns
        -------
        slope : float64 
            The slope of the line joining the two points 
        """
        slope = (y_2 - y_1)/(x_2 - x_1)
        return slope

    # Bisection algorithm to find the interval in which a point/array belongs
    def bisection(self, x_value, x_data, order):
        """Returns an interval containing the point of interest.

        Parameters
        ----------
        x_value : float64
            The point of interest whose interval needs to be found.
        x_data : numpy.ndarray
            Set of increasing coordinates which act as different intervals.
        order : int
            Determines the number of extra points to be included.

        Returns
        -------
        interval: numpy.ndarray
            The interval (and possibly a few surrounding points) containing the point of interest.
        """
        # Finding the interval in which the point resides
        x_half = x_data
        while len(x_half) > 2:
            n = len(x_half)
            x_right_half = x_half[int(n/2):]
            x_left_half = x_half[0:int(n/2)+1]
            if x_value <= x_left_half[-1]:
                x_half = x_left_half
            if x_value >= x_right_half[0]:
                x_half = x_right_half

        # Need to add new points if order != 1
        no_of_new_points = order - 1
        left_bdy, right_bdy = x_half
        left_ind = np.where(x_data == left_bdy)[0][0]
        right_ind = np.where(x_data == right_bdy)[0][0]

        # Half of the number of new points are added on either side
        right_new_indices_no = int(0.5*no_of_new_points)
        left_new_indices_no = no_of_new_points - right_new_indices_no
        new_right_ind = np.arange(right_ind + 1, right_ind + right_new_indices_no + 1, 1)
        new_right_ind = new_right_ind.astype(int)
        new_left_ind = np.arange(left_ind - 1, left_ind - left_new_indices_no - 1, -1)
        new_left_ind = new_left_ind.astype(int)

        # The arrays containing the new left and right indices 
        # are added to the old array and sorted
        indices = np.array([left_ind, right_ind])
        indices = np.append(indices, new_left_ind)
        indices = np.append(indices, new_right_ind)
        indices = np.sort(indices)

        # Indices outside x_data are shifted inside appropriately
        negative_indices = indices[indices < 0]
        swap_negative_indices = np.arange(indices[-1] + 1, indices[-1] + 1 + len(negative_indices), 1)
        indices = np.append(indices, swap_negative_indices)
        indices = indices[~(indices < 0)]
        indices = np.sort(indices)
        extended_indices = indices[indices > len(x_data) - 1]
        swap_extended_indices = np.arange(indices[0] - 1, indices[0] - len(extended_indices) - 1, -1)
        indices = np.append(indices, swap_extended_indices)
        indices = indices[~(indices > len(x_data) - 1)]
        indices = np.sort(indices)
        interval = np.take(x_data, indices)
        return interval


    # Linear interpolator
    def lin_interp(self, x, x_data, y_data):
        """Computes the linear interpolation at the point of interest given a set of known data points.

        Parameters
        ----------
        x : float64
            The point of interest at which the linear interpolation needs to be found.
        x_data : numpy.ndarray
            x-coordinates of known data points.
        y_data : numpy.ndarray
            y-coordinates of known data points.

        Returns
        -------
        y: float64
            The linear interpolation at x. 
        """
        x_1, x_2 = self.bisection(x, x_data, order = 1)
        y_1 = y_data[x_data == x_1]
        y_2 = y_data[x_data == x_2]
        y = self.slope(x_1, y_1, x_2, y_2)*(x - x_1) + y_1
        return y

    # Neville's algorithm
    def neville(self, x, x_data, y_data, order):
        """Computes the Lagrange polynomial at a point of interest using Neville's algorithm.

        Parameters
        ----------
        x : float64
            Point of interest at which Lagrange polynomial needs to be evaluated.
        x_data : numpy.ndarray
            x-coordinates of known data points
        y_data : numpy.ndarray
            y-coordinates of known data points
        order : int
            Order of the Lagrange polynomial.

        Returns
        -------
        p[0]: float64
            Value of Lagrange polynomial at point of interest.
        """
        interval = self.bisection(x_value = x, x_data = x_data, order = order)
        M = order + 1
        indices = np.where(np.in1d(x_data, interval))
        p = y_data[indices]
        for k in range(1, M):
            for i in range(0, M - k):
                j = i + k
                weighted_sum = (interval[j] - x)*p[i] + (x - interval[i])*p[i+1]
                p[i] = weighted_sum/(interval[j] - interval[i])
        return p[0]
            

if __name__ in "__main__":

    # Available data points
    # Providing random y-values for t = 0 and t = 40
    t_data = np.array([0.0000, 1.0000,  4.3333, 7.6667, 11.000, 14.333, 17.667, 21.000, 40.000])
    I_t_data = np.array([-2.0000, 1.4925, 15.323, 3.2356, -29.472, -22.396, 24.019, 36.863, 60.000])

    t_main = np.array([])
    for i in range(len(t_data)-1):
        t_interp = np.linspace(t_data[i], t_data[i+1], num = 40)
        if i == 0:
            t_main = np.append(t_main, t_interp)
        else:
            t_main = np.append(t_main, t_interp[1:])

    # Plot showing data points and linear interpolation
    # t_main = np.linspace(0, 40, 101)
    y_lin_interp = np.array([])
    y_neville = np.array([])

    interp = interpolator()
    for t in t_main:
        y_i_lin_interp = interp.lin_interp(x = t, x_data = t_data, y_data = I_t_data)
        y_lin_interp = np.append(y_lin_interp, y_i_lin_interp)
        y_i_neville = interp.neville(x = t, x_data = t_data, y_data = I_t_data, order = 3)
        y_neville = np.append(y_neville, y_i_neville)

    plt.scatter(t_data, I_t_data, color = "black", label = "Data points")
    plt.plot(t_main, y_lin_interp, label = "Linear interpolation")
    plt.plot(t_main, y_neville, '--', label = "Lagrange polynomial")
    plt.legend()
    plt.show()

