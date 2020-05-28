import matplotlib.pyplot as plt
import csv
import numpy as np

np.random.seed(0)
x = 2 - 3 * np.random.normal(0, 1, 20)
y = x - 2 * (x ** 2) + 0.5 * (x ** 3) + np.random.normal(-3, 3, 20)

# transforming the data to include another axis
data_x = x
data_y = y
print(data_x)
plt.figure(figsize=(10, 5))
plt.plot(data_x, data_y, "bs")
plt.xlabel("")
plt.ylabel("")
plt.show()

data_x = np.array(x, dtype=float)
data_y = np.array(y, dtype=float)

p = data_x.argsort()
data_x = data_x[p]
data_y = data_y[p]

l_rate = 5e-4 # 0.0005
epochs = 100

n_exp = 3

def h_(x, p0, p1, p2, p3):
    return p0 + x * p1 + p2 * (x ** 2) + p3 * (x ** 3)

# y = p0 + p1 * x + p2 * x^2 + p3 * x ^ 3
p0 = 0
p1 = 0
p2 = 0
p3 = 0
p_n = np.zeros(n_exp, dtype=float)
errors = []

for e in range(epochs):
    # Gradient descent p0
    s = 0
    # sumatoria de h(x) - y
    for x, y in zip(data_x, data_y):
        # x_n = []
        # x_n.append(1)
        # for ei in range(n_exp):
        #     x_n.append(x ** ei)
        # x_n = np.array(x_n, dtype=float)
        # h = x_n.dot(p_n)
        h = h_(x, p0, p1, p2, p3)
        s += (h - y)
    m = len(data_y)
    # gradiente = sumatoria / numero de muestas
    g = s / m
    p0_1 = p0 - l_rate * g

    # Gradient descent p1
    s = 0
    # sumatoria de h(x) - y * x
    for x, y in zip(data_x, data_y):
        h = h_(x, p0, p1, p2, p3)
        s += (h - y) * x
    m = len(data_y)
    # gradiente = sumatoria / numero de muestas
    g = s / m
    p1_1 = p1 - l_rate * g

    # Gradient descent p1
    s = 0
    # sumatoria de h(x) - y * x
    for x, y in zip(data_x, data_y):
        h = h_(x, p0, p1, p2, p3)
        s += (h - y) * (x ** 2)
    m = len(data_y)
    # gradiente = sumatoria / numero de muestas
    g = s / m
    p2_1 = p2 - l_rate * g

    # Gradient descent p1
    s = 0
    # sumatoria de h(x) - y * x
    for x, y in zip(data_x, data_y):
        h = h_(x, p0, p1, p2, p3)
        s += (h - y) * (x ** 3)
    m = len(data_y)
    # gradiente = sumatoria / numero de muestas
    g = s / m
    p3_1 = p3 - l_rate * g

    p0 = p0_1
    p1 = p1_1
    p2 = p2_1
    p3 = p3_1

    print("{} {} {}".format(p0, p1, p2))
    estimacion_y = h_(data_x, p0, p1, p2, p3)
    print(estimacion_y)
    # calcular el error
    s = 0
    # sumatoria de (h(x) - y) ^ 2
    for x, y in zip(data_x, data_y):
        h = h_(x, p0, p1, p2, p3)
        s += (h - y) ** 2
    m = len(data_y)
    # error = sumatoria / numero de muestas x 2
    e = s / 2 * m
    errors.append(e)
    errors_np = np.array(errors, dtype=float)
    errors_x = np.arange(len(errors))

    plt.figure(figsize=(10, 5))

    plt.subplot(121)
    plt.plot(data_x, data_y, "bd", data_x, estimacion_y)
    plt.xlabel("Poverty rate")
    plt.ylabel("Birth rate 15 - 17")

    plt.subplot(122)
    plt.plot(errors_x, errors_np)
    plt.xlabel("epochs")
    plt.ylabel("error")

    plt.show()




