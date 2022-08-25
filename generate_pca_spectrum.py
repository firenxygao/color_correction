import numpy as np
from numpy.random import default_rng
from scipy.io import savemat
from scipy.io import loadmat
from scipy.interpolate import pchip_interpolate
from scipy.optimize import minimize
from sklearn.decomposition import PCA

rng = default_rng(0)


def crop_spectrum(mat, name, low, high):
    wave = mat['wavelength'][0].tolist()
    ind_low = wave.index(low)
    ind_high = wave.index(high)
    data = mat[name]
    cropdata = data[:, ind_low:ind_high + 1]
    return cropdata


def interp_spectrum(mat, x, x_interp):
    num = mat.shape[0]
    mat_interp = np.tile(np.zeros_like(x_interp, 'float'), (num, 1))
    for i in range(num):
        mat_interp[i, :] = pchip_interpolate(x, mat[i, :], x_interp)
    return mat_interp

# specify the spectrum range
low = 400
high = 700
dim = high - low + 1
samples4 = np.array(list(range(low, high + 1, 4)))
samples5 = np.array(list(range(low, high + 1, 5)))

# D65 standard illuminant source: http://www.npsg.uwaterloo.ca/data/illuminant.php
light = loadmat('d65.mat')                              # 300-830 step=1

# XYZ color matching function source: http://www.cvrl.org/ or http://cvrl.ioo.ucl.ac.uk/cmfs.htm
cmf = loadmat('xyz.mat')                                # 360-830 step=1

# camera sensitivity source: https://spectralestimation.wordpress.com/data/
camera = loadmat('nikon_sensitivity.mat')               # 380-780 step=5

# SFU reflectance dataset
reflectance = loadmat('reflectance_1993.mat')           # 380-780 step=4

spd = crop_spectrum(light, 'power', low, high)          # 1xD
spd = spd / np.max(spd)
xyz = crop_spectrum(cmf, 'xyz', low, high)              # 3xD
rgb = crop_spectrum(camera, 'rgb', low, high)           # 3x(D/5)
# 'all' 1993; 'macbeth' 24; 'munsell' 1269; 'dupont' 120; 'objects' 170; 'krinov' 355; 'additional' 55
reflect = crop_spectrum(reflectance, 'all', low, high)  # Nx(D/4)

samples_interp = np.linspace(low, high, num=dim)
rgb_interp = interp_spectrum(rgb, samples5, samples_interp)
reflect_interp = interp_spectrum(reflect, samples4, samples_interp)
N = reflect_interp.shape[0]
K = 20 # number of principal components
pca = PCA(n_components=K)
pca.fit(reflect_interp)
basis = pca.components_ # KxD
values = pca.singular_values_ # sqrt of eigenvalues
variance = pca.explained_variance_ratio_ # eigenvalues percentage
reflect_interp_mean = np.mean(reflect_interp, 0)
reflect_interp_center = reflect_interp - reflect_interp_mean
# cov = np.matmul(np.transpose(reflect_interp_center), reflect_interp_center)
# evalues, evectors = eig(cov) # eigenvectors in column
reflect_interp_coef = np.matmul(reflect_interp_center, np.transpose(basis))
bound_lower = np.min(reflect_interp_coef, 0)
bound_upper = np.max(reflect_interp_coef, 0)

dataset_number = 10000
dataset_spectrum = np.zeros((dataset_number, dim), 'float')
dim_keep = 10
cnt = 0

while cnt < dataset_number:
    flag = True
    spectrum = np.mean(reflect_interp,0)
    for i in range(dim_keep):
        weight = rng.uniform(bound_lower[i], bound_upper[i])
        spectrum += weight * basis[i,:]
        # print(weight)
        # print(spectrum[::60])
        if np.min(spectrum) < 0.0 or np.max(spectrum) > 1.0:
            flag = False
            break
    if not flag:
        continue
    dataset_spectrum[cnt, :] = spectrum
    cnt += 1
    if cnt % 100 == 0:
        print(f'finish {cnt}')

mydic = {'number': dataset_number, 'spectrum': dataset_spectrum}
savemat(f"reflectance_pca_{dataset_number}.mat", mydic)