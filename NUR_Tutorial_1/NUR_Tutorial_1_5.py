import numpy as np
from timeit import timeit
from time import time

# 5. Function to calculate the area of a polygon given 
# coordinate lists as inputs
def polygon_area(x, y):

    sum = 0
    n = len(x)
    for i in range(n):
        j = i+1
        if i+1 == n:
            j = 0
        sum += (x[i]*y[j] - x[j]*y[i])
    A = 0.5*np.abs(sum)
    return A

# c) Vectorised function to calculate the area of a polygon
def polygon_area_vectorized(x,y):
    x_i = x
    # x_i_1 is an array with x's last element at the 0th position
    x_i_1 = np.concatenate((np.array([x[-1]]), x[0:-1]))
    y_i = y
    # y_i_1 is an array with x's last element at the 0th position
    y_i_1 = np.concatenate((np.array([y[-1]]), y[0:-1]))
    A = 0.5*np.abs(np.dot(x_i, y_i_1) - np.dot(x_i_1, y_i))
    return A

# e) Computing the area of triangles for many runs
def area_calculator(type, n_iter):
    count = 1
    if type == 1:
        while count <= n_iter:
            x_coords = np.random.randint(0, 11, 3)
            y_coords = np.random.randint(0, 11, 3)
            area = polygon_area(x = x_coords, y = y_coords)
            count += 1
    else:
        while count <= n_iter:
            x_coords = np.random.randint(0, 11, 3)
            y_coords = np.random.randint(0, 11, 3)
            area = polygon_area_vectorized(x = x_coords, y = y_coords)
            count += 1


if __name__ in "__main__":

    # # a) Testing the function on standard polygons
    # x_triangle = np.array([0, 0, 3])
    # y_triangle = np.array([4, 0, 0])
    # print(f"Triangle \nExpected area: 6 \nActual area: {polygon_area(x_triangle, y_triangle)}")
    # x_square = np.array([-1, 1, 1, -1])
    # y_square = np.array([1, 1, -1, -1])
    # print(f"Square \nExpected area: 4 \nActual area: {polygon_area(x_square, y_square)}")
    # x_pentagon = np.array([0, -1, -1, 1, 1])
    # y_pentagon = np.array([2, 1, -1, -1, 1])
    # print(f"Pentagon \nExpected area: 5\n Actual area: {polygon_area(x_pentagon, y_pentagon)}") 

    # # b) Computing the time taken to run the codes
    # t_triangle = timeit("polygon_area(x_triangle, y_triangle)", globals = globals())
    # t_square = timeit("polygon_area(x_square, y_square)", globals = globals())
    # t_pentagon = timeit("polygon_area(x_pentagon, y_pentagon)", globals = globals())
    # print("Triangle area runtime: ", t_triangle)
    # print("Square area runtime: ", t_square)
    # print("Pentagon area runtime: ", t_pentagon)

    # # c) Comparing normal and vectorized implementations
    # print(f"""
    #       Triangle \n
    #       Normal: {polygon_area(x_triangle, y_triangle)} \n
    #       Vectorized: {polygon_area_vectorized(x_triangle, y_triangle)}""")
    # print(f"""
    #       Square \n
    #       Normal: {polygon_area(x_square, y_square)} \n
    #       Vectorized: {polygon_area_vectorized(x_square, y_square)}""")
    # print(f"""
    #       Pentagon \n
    #       Normal: {polygon_area(x_pentagon, y_pentagon)} \n
    #       Vectorized: {polygon_area_vectorized(x_pentagon, y_pentagon)}""")

    # # d) Comparing times of normal and vectorised implementations
    # print("Times")
    # print("Triangle")
    # t_3_norm = timeit("polygon_area(x_triangle, y_triangle)", globals = globals())
    # t_3_vectorized = timeit("polygon_area_vectorized(x_triangle, y_triangle)", globals = globals())
    # print(f"Normal: {t_3_norm} \nVectorized: {t_3_vectorized}")
    # frac_3 =  t_3_norm/t_3_vectorized
    # print(f"{frac_3} times faster")

    # print("Square")
    # t_4_norm = timeit("polygon_area(x_square, y_square)", globals = globals())
    # t_4_vectorized = timeit("polygon_area_vectorized(x_square, y_square)", globals = globals())
    # print(f"Normal: {t_4_norm} \nVectorized: {t_4_vectorized}")
    # frac_4 =  t_4_norm/t_4_vectorized
    # print(f"{frac_4} times faster")

    # print("Pentagon")
    # t_5_norm = timeit("polygon_area(x_pentagon, y_pentagon)", globals = globals())
    # t_5_vectorized = timeit("polygon_area_vectorized(x_pentagon, y_pentagon)", globals = globals())
    # print(f"Normal: {t_5_norm} \nVectorized: {t_5_vectorized}")
    # frac_5 =  t_5_norm/t_5_vectorized
    # print(f"{frac_5} times faster")

    # e) Comparison for 10000 triangles
    # For loop implementation
    normal_run_t = timeit("area_calculator(type = 1, n_iter = 10000)", 
                          number = 1, globals = globals())
    # Vectorized implementation
    vectorized_run_t = timeit("area_calculator(type = 2, n_iter = 10000)", 
                              number = 1, globals = globals())
    print("Times for 10000 triangles:")
    print("For loop: ", normal_run_t)
    print("Vectorized run: ", vectorized_run_t)


    
