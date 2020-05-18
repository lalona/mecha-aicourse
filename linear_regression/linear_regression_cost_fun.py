import matplotlib.pyplot as plt
import numpy as np

def cost_func(es_y, gt_y):
    """"
    es_x representa la estimacion
    y gt_x representa lso valroes reales
    """
    s = 0
    for e, g in zip(es_y, gt_y):
        s += (e - g) ** 2
    m = len(es_y)
    error = s / (2 * m)
    return error

def plot_err(data_x, data_y, err_range=(-10,10), error_step=0.5, slope=1, old_slope=1):

    tests = []
    errors = []
    for i in np.arange(err_range[0], err_range[1], error_step):
        estimation_y = data_x * i
        error = cost_func(estimation_y, data_y)
        tests.append(i)
        errors.append(error)

    i = slope
    estimation_y = data_x * i
    error = cost_func(estimation_y, data_y)
    print(error)

    estimation_y = data_x * old_slope
    error_old = cost_func(estimation_y, data_y)
    print(error)

    # for j in range(len(data_x)):
    #     err_line =

    err_line1 = np.arange(data_y[0], estimation_y[0], 0.2)
    err_line2 = np.arange(data_y[1], estimation_y[1], 0.2)
    err_line3 = np.arange(data_y[2], estimation_y[2], 0.2)

    ones = np.tile(1,err_line1.shape)
    twos = np.tile(2, err_line2.shape)
    trees = np.tile(3, err_line3.shape)

    plt.figure(figsize=(10, 5))

    print(ones)
    plt.subplot(121)
    #plt.plot(data_x, data_y, "bs", data_x, estimation_y, ones, err_line1, 'r--', twos, err_line2, 'r--', trees, err_line3, 'r--')
    plt.plot(data_x, data_y, "bs", data_x, estimation_y)
    plt.ylabel('some numbers')

    plt.subplot(122)
    plt.title("p1: {} err: {}".format(i, error))
    plt.plot(tests, errors, [i], [error], "ro", [old_slope], [error_old], "go")
    plt.ylabel('some numbers')

    plt.show()

def gradient_descent_lr_p1(data_x, data_y, p1, a):
    """"
    Gradient descent aplicado a la funcion de costo de linear regression para el parametro 1
    a is the alfa representing learning rate
    """
    gradient = der_cost_func_p1(data_x, data_y, p1)
    return p1 - (a * gradient)



def der_cost_func_p1(es_x, gt_y, p1):
    """"
    La derivada parcial de la funcion de costo con respectoa p1
    """
    s = 0
    for ex, gy in zip(es_x, gt_y):
        ey = ex * p1
        s += ((ey - gy) * ex)
    m = len(es_x)
    # gradiente
    g = s / m
    print(g)
    return g

if __name__ == "__main__":
    data_x = np.array([1, 2, 3], dtype=float)
    data_y = np.array([1, 2, 3], dtype=float)
    plot_err(data_x, data_y, err_range=(-10, 10), error_step=0.5, slope=2)