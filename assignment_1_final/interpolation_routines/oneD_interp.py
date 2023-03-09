import numpy as np
import matplotlib.pyplot as plt

# Function to find the slope of the line joining two coordinates
def slope(x_1, y_1, x_2, y_2):
    slope = (y_2 - y_1)/(x_2 - x_1)
    return slope

# Bisection algorithm to find the interval in which a point/array belongs
def bisection(x_value, x_data):
    # for one value (works!)
    x_half = x_data
    while len(x_half) > 2:
        n = len(x_half)
        x_right_half = x_half[int(n/2):]
        x_left_half = x_half[0:int(n/2)+1]
        if x_value <= x_left_half[-1]:
            x_half = x_left_half
        if x_value >= x_right_half[0]:
            x_half = x_right_half
    return x_half
        

# Linear interpolator
def lin_interp(x_values, x_data, y_data):
    y_values = np.array([])
    for x in x_values:
        x_1, x_2 = bisection(x, x_data)
        y_1 = y_data[x_data == x_1]
        y_2 = y_data[x_data == x_2]
        y = slope(x_1, y_1, x_2, y_2)*(x - x_1) + y_1
        y_values = np.append(y_values, y)
    return y_values

# Neville's algorithm
def neville(x, x_data, y_data):
    p_i = y_data
    M = len(y_data)
    for k in range(1,M):
        for i in range(M-k):
            j = i + k
            


if __name__ in "__main__":

    # Available data points
    # Providing random y-values for t = 0 and t = 40
    t_data = np.array([0.0000, 1.0000,  4.3333, 7.6667, 11.000, 14.333, 17.667, 21.000, 40.000])
    I_t_data = np.array([-2.0000, 1.4925, 15.323, 3.2356, -29.472, -22.396, 24.019, 36.863, 60.000])

    # Plot showing data points and linear interpolation
    t_main = np.linspace(0, 40, 101)
    y_main = lin_interp(x_values = t_main, x_data = t_data, y_data = I_t_data)
    plt.plot(t_main, y_main, label = "Linear interpolation")
    plt.scatter(t_data, I_t_data, color = "black", label = "Data points")
    plt.legend()
    plt.show()

        

