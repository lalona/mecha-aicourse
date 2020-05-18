import csv
import numpy as np
import matplotlib.pyplot as plt

price = 2
start_x = 2

data_y = []
data_x = []
with open('../data/kc_house_data.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0

    for row in csv_reader:
        if line_count != 0:
            x_default = ['1']
            x_default.extend(row[start_x:])
            data_x.append(x_default)
            data_y.append(row[price])
        line_count += 1

data_x = np.array(data_x, dtype=float)
data_y = np.array(data_y, dtype=float)

print(data_x.shape)
print(data_x[:,0])

# Iterar sobre las columnas para aplicar MEAN NORMALIZATION excepto para la columna 1
# que contiene puros 1
for c in range(1, data_x.shape[1]):
    x_c = data_x[:, c]
    mean = np.mean(x_c)
    max = np.amax(x_c)
    min = np.amin(x_c)
    data_x[:, c] = (x_c - mean) / (max - min)

l_rate = 5e-1 # 0.0005
epochs = 30

# parametros
pi = np.zeros(data_x[0].shape)

errors = []
for ep in range(epochs):
    # Gradient descent for the parameters array
    # Tenemos parametros temporales
    pt = np.zeros(pi.shape)
    for j, p in enumerate(pi):
        s = 0
        # sumatoria de h(x) - y
        for xi, yi in zip(data_x, data_y):
            # ahora se aplica el producto scalar al vector de x con el vector de parametros
            h = xi.dot(pi)
            s += (h - yi) * xi[j]
        m = len(data_y)
        # gradiente = sumatoria / numero de muestas
        g = s / m
        pt[j] = p - l_rate * g
    #print("Old: {:.2f} New: {:.2f}".format(p, pt[j]))

    pi = pt
    # Se multiplica la tranpuesta de pi con el vector data_x

    estimacion_y = np.zeros(data_y.shape)
    for i, xi in enumerate(data_x):
        estimacion_y[i] = xi.dot(pi)

    # calcular el error
    s = 0
    # sumatoria de (h(x) - y) ^ 2
    for h, y in zip(estimacion_y, data_y):
        s += (h - y) ** 2
    # error = sumatoria / numero de muestas x 2
    e = s / 2 * m
    errors.append(e)
    errors_np = np.array(errors, dtype=float)
    errors_x = np.arange(len(errors))
    print("Epoch: {} Error: {}".format(ep, e))
plt.figure(figsize=(10, 5))

plt.subplot(111)
plt.plot(errors_x, errors_np)
plt.xlabel("epochs")
plt.ylabel("error")
plt.title("Error: {}".format(e))

plt.show()
