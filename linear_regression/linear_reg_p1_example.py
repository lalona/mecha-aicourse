import csv
from linear_regression_cost_fun import plot_err, gradient_descent_lr_p1
import numpy as np
PovPct = 1
Brth15to17 = 2

data_x = []
data_y = []
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


data_x = np.array(data_x, dtype=float)
data_y = np.array(data_y, dtype=float)

l_rate = 5e-3
plot_err(data_x, data_y, err_range=(-3, 5), error_step=0.05, slope=p1, old_slope=p1)

for l in range(5):
    p1_2 = gradient_descent_lr_p1(data_x, data_y, p1=p1, a=l_rate)
    print(p1_2)
    plot_err(data_x, data_y, err_range=(-3, 5), error_step=0.05, slope=p1_2, old_slope=p1)
    p1 = p1_2


