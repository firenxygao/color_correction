import numpy as np
from scipy.io import savemat
import csv

path = 'ciexyz31_1.csv' # step=1
# path = 'ciexyz31.csv'   # step=5

wave = []
x = []
y = []
z = []

with open(path, newline='') as f:
	reader = csv.reader(f, delimiter=' ', quotechar='|')
	for row in reader:
		data = row[0].split(',')
		wave.append(data[0])
		x.append(data[1])
		y.append(data[2])
		z.append(data[3])

wave = list(map(lambda x: int(x), wave))
x = list(map(lambda x: float(x), x))
y = list(map(lambda x: float(x), y))
z = list(map(lambda x: float(x), z))

wave = np.array(wave)
x = np.array(x)
y = np.array(y)
z = np.array(z)
xyz = np.stack((x, y, z), axis=0)

mydic = {'wavelength': wave, 'xyz': xyz}

savemat("xyz.mat", mydic)