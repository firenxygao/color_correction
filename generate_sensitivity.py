import numpy as np
from scipy.io import savemat

wave = []
red = []
green = []
blue = []

with open('nikon_sensitivity.txt', 'r') as f:
	line = f.readline()
	while line is not None and line != '':
		data = line.split('\t')
		wave.append(data[0])
		red.append(data[1])
		green.append(data[2])
		blue.append(data[3])
		line = f.readline()

wave = list(map(lambda x: int(x), wave))
red = list(map(lambda x: float(x), red))
green = list(map(lambda x: float(x), green))
blue = list(map(lambda x: float(x), blue))

wave = np.array(wave)
red = np.array(red)
green = np.array(green)
blue = np.array(blue)
rgb = np.stack((red, green, blue), axis=0)

mydic = {'wavelength': wave, 'rgb': rgb}

savemat("nikon_sensitivity.mat", mydic)