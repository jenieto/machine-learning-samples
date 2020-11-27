import random
import numpy as np

file_name = 'data.csv'
columns = np.array(['x1', 'x2', 'x3', 'x4', 'x5'])


def xor(a, b):
    return bool(a) ^ bool(b)


def xor_3(a, b, c):
    return xor(xor(a, b), c)


def random_bool():
    return bool(random.getrandbits(1))


def generate_data(size=100):
    data = []
    for i in range(size):
        x1 = random_bool()
        x2 = random_bool()
        x3 = random_bool()
        x4 = xor(x2, x3)
        x5 = random_bool()
        label = xor_3(x1, x2, x3)
        row = [x1, x2, x3, x4, x5, label]
        data.append(row)
    return np.array(data)


def store_data():
    data = generate_data()
    np.savetxt(file_name, data, delimiter=",")


def read_data():
    raw_data = np.loadtxt(file_name, delimiter=',')
    data = raw_data[:, 0:5]
    labels = raw_data[:, 5]
    return data, labels, columns


# read_data()
# store_data()
