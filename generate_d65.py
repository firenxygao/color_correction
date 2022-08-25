import numpy as np
from scipy.io import savemat

wave = []
power = []

with open('d65.txt', 'r') as f:
	line = f.readline()
	while line is not None and line != '':
		data = line.split('  ')
		wave.append(data[0])
		power.append(data[1])
		line = f.readline()

wave = list(map(lambda x: int(x), wave))
power = list(map(lambda x: float(x), power))

wave = np.array(wave)
power = np.array(power)

mydic = {'wavelength': wave, 'power': power}

savemat("d65.mat", mydic)