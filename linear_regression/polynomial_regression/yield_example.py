import math
def example_sin(x):
    yield math.sin(x)

z = [1, 2, 3]
for i in example_sin(z):
    print(i)