import numpy as np
from scipy.io import savemat

dataPath = 'SFU_dataset/reflect_db_1993.reflect'
low = 380
high = 780
step =4
dim = int((high - low)/4) + 1
wave = np.array(list(range(low, high + 1, 4)))

with open(dataPath, 'r') as f:
	content = f.readlines()

content = list(filter(lambda x: (x != '\n') , content))
content = list(map(lambda x: float(x), content))
shape = [1993, dim]
mat = np.array(content)
mat = np.reshape(mat, shape)

books     = mat[0:7,:]
cardboard = mat[7:28,:]
cloth     = mat[28:35,:]
lab_wall  = mat[35:36,:]
macbeth   = mat[36:60,:]
paint     = mat[60:79,:]
dupont    = mat[79:199,:]
krinov    = mat[199:554,:]
munsell   = mat[554:1823,:]
objects   = mat[1823:,:]

additional = np.concatenate((books, cardboard, cloth, lab_wall, paint), 0)

mydic = {'wavelength': wave, 'all': mat, 'macbeth': macbeth, 'munsell': munsell, 'dupont': dupont, 'objects': objects, 'krinov': krinov,
'additional': additional}

savemat("reflectance_1993.mat", mydic)