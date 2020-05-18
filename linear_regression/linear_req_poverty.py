import csv
from linear_re_utils import plot_err, gradient_descent_lr_p1, gradient_descent_lr_p0
import numpy as np
PovPct = 1
Brth15to17 = 2

data_x = []
data_y = []
p0 = -2
p1 = -2
with open('../data/poverty_usa.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            pov = float(row[PovPct])
            brt = float(row[Brth15to17])
            if pov not in data_y and brt not in data_x:
                data_x.append(pov)
                data_y.append(brt)

# The training porcentage is gonna be 70 and the test is gonna be 30
porcentrage_training = 70
training_lenght = int(len(data_x) * porcentrage_training / 100)

data_x_test = data_x[training_lenght:]
data_y_test = data_y[training_lenght:]
data_x = data_x[:training_lenght]
data_y = data_y[:training_lenght]
data_x = np.array(data_x, dtype=float)
data_y = np.array(data_y, dtype=float)
data_x_test = np.array(data_x_test, dtype=float)
data_y_test = np.array(data_y_test, dtype=float)

l_rate = 5e-3
errors = []
#err = plot_err(data_x, data_y, slope=p1, p0=p0)
#errors.append(err)
for l in range(5):
    err = plot_err(data_x, data_y, slope=p1, p0=p0, error_y=errors)
    errors.append(err)
    p1_2 = gradient_descent_lr_p1(data_x, data_y, p0=p0, p1=p1, a=l_rate)
    p0_2 = gradient_descent_lr_p0(data_x, data_y, p0=p0, p1=p1, a=l_rate)
    p1 = p1_2
    p0 = p0_2

err, err_t = plot_err(data_x_test, data_y_test, slope=p1_2, p0=p0, error_y=errors)
