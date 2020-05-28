import matplotlib.pyplot as plt
import csv
import numpy as np

# numero de columnas que nos interesan
poverty = 1
brtrate_1517 = 2

data_y = []
data_x = []

with open('../data/poverty_usa.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    count = 0
    for row in csv_reader:
        if count != 0:
            data_x.append(row[poverty])
            data_y.append(row[brtrate_1517])
        count += 1

data_x = np.array(data_x, dtype=float)
data_y = np.array(data_y, dtype=float)



# y = p0 + p1 * x
p0 = 0
p1 = 1

iteraciones = 100
learning_rate = 0.0005

errors = []
for i in range(iteraciones):

    s = 0
    for xi, yi in zip(data_x, data_y):
        h = xi * p1 + p0
        s += (h - yi)
    m = len(data_x)
    p0_temp = p0 - (learning_rate * (s / m))

    # sumatoria
    s = 0
    for xi, yi in zip(data_x, data_y):
        h = xi * p1 + p0
        s += (h - yi) * xi
    m = len(data_x)
    p1_temp = p1 - (learning_rate * (s / m))

    # calcular el error
    s = 0
    # sumatoria de (h(x) - y) ^ 2
    for x, y in zip(data_x, data_y):
        h = x * p1 + p0
        s += (h - y) ** 2
    m = len(data_y)
    # error = sumatoria / numero de muestas x 2
    e = s / 2 * m
    errors.append(e)
    errors_np = np.array(errors, dtype=float)
    errors_x = np.arange(len(errors))

    p0 = p0_temp
    p1 = p1_temp

    estimacion_y = data_x * p1 + p0

plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.plot(data_x, data_y, "bs", data_x, estimacion_y)

plt.subplot(122)
plt.plot(errors_x, errors_np)
plt.xlabel("epochs")
plt.ylabel("error")

plt.show()