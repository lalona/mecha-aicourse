import matplotlib.pyplot as plt
import csv
import numpy as np

poverty = 1
brtrate_1517 = 2

# leyendo nuestros datos de entrenamiento
data_y = []
data_x = []
with open('../data/poverty_usa.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count != 0:
            data_x.append(row[poverty])
            data_y.append(row[brtrate_1517])
        line_count += 1

data_x = np.array(data_x, dtype=float)
data_y = np.array(data_y, dtype=float)

l_rate = 5e-5 # 0.005
epochs = 200

# y = p0 + p1 * x
p0 = 0
p1 = 0
errors = []
# ciclo de entrenamiento
for e in range(0, epochs, 1):
    # Gradient descent p0
    s = 0
    # sumatoria de h(x) - y
    for x, y in zip(data_x, data_y):
        h = x * p1 + p0
        s += h - y
    m = len(data_y)
    # gradiente = sumatoria / numero de muestas
    g = s / m
    p0_temp = p0 - l_rate * g

    # Gradient descent p1
    s = 0
    # sumatoria de h(x) - y * x
    for x, y in zip(data_x, data_y):
        h = x * p1 + p0
        s += (h - y) * x
    m = len(data_y)
    # gradiente = sumatoria / numero de muestas
    g = s / m
    p1_temp = p1 - l_rate * g

    p0 = p0_temp
    p1 = p1_temp

    estimacion_y = data_x * p1 + p0

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
plt.xlabel("Poverty porcentage")
plt.ylabel("Birth rate 15 - 17")
plt.title("p0 = {:.2f} p1 = {:.2f}".format(p0, p1))

plt.subplot(122)
plt.plot(epochs_np, errors_np)
plt.xlabel("epochs")
plt.ylabel("error")

plt.show()




