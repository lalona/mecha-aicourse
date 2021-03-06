import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)
x = 2 - 3 * np.random.normal(0, 1, 20)
y = x - 2 * (x ** 2) + 0.5 * (x ** 3) + np.random.normal(-3, 3, 20)

data_x = np.array(x, dtype=float)
data_y = np.array(y, dtype=float)

# transforming the data to include another axis
print(data_x)
plt.figure(figsize=(10, 5))
plt.plot(data_x, data_y, "bs")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

p = data_x.argsort()
data_x = data_x[p]
data_y = data_y[p]

l_rate = 5e-3 # 0.0005
epochs = 1000

# y = p0 + p1 * x + p2 * x ** 2
# [0, 0, 0]
size_p = 3 # cuadratica

# Inicializacion de parametros
p = np.zeros(size_p, dtype=float)
errors = []

def hipotesis(p, xi):
    # [x ** 0, x ** 1, x ** 2, .. x ** n]
    x = [xi ** c for c in range(len(p))]
    x = np.array(x, dtype=float)
    return x.dot(p)

# ciclo de entrenamiento
for e in range(0, epochs, 1):
    p_temp = p
    for n in range(size_p):
        # Gradient descent pi
        s = 0
        m = len(data_y)
        # sumatoria de h(x) - y
        for i in range(m):
            xi = data_x[i]
            yi = data_y[i]
            h = hipotesis(p, xi)
            s += (h - yi) * xi ** n
        # gradiente = sumatoria / numero de muestas
        g = s / m
        p_temp[n] = p[n] - l_rate * g
    p = p_temp
    estimacion_y = [hipotesis(p, xi) for xi in data_x]
    estimacion_y = np.array(estimacion_y, dtype=float)

    # calcular el error
    s = 0
    # sumatoria de (h(x) - y) ^ 2
    for x, y, h in zip(data_x, data_y, estimacion_y):
        s += (h - y) ** 2
    m = len(data_y)
    # error = sumatoria / numero de muestas x 2
    e = s / 2 * m
    errors.append(e)
    errors_np = np.array(errors, dtype=float)
    epochs_np = np.arange(len(errors))

plt.figure(figsize=(10, 5))

plt.subplot(121)
plt.plot(data_x, data_y, "bd", data_x, estimacion_y)
plt.xlabel("x")
plt.ylabel("y")
plt.title("p {}".format(p))

plt.subplot(122)
plt.plot(epochs_np, errors_np)
plt.xlabel("epochs")
plt.ylabel("error")

plt.show()




