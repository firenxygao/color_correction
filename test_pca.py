import numpy as np
from numpy.random import default_rng
from numpy.linalg import pinv
from numpy.linalg import eig
from scipy.io import loadmat
from scipy.interpolate import pchip_interpolate
from scipy.optimize import minimize
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

torch.manual_seed(0)
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


def calculate_regression_matrix(illum, reflect, xyz, rgb, dx=1.0):
    # PM = Q
    rows = reflect.shape[0]
    Q = np.zeros((rows, 3), 'float')
    PL = np.zeros((rows, 3), 'float')
    PP = np.zeros((rows, 9), 'float')
    PR = np.zeros((rows, 6), 'float')

    Q[:, 0] = np.trapz(xyz[0, :] * illum * reflect, dx=dx)
    Q[:, 1] = np.trapz(xyz[1, :] * illum * reflect, dx=dx)
    Q[:, 2] = np.trapz(xyz[2, :] * illum * reflect, dx=dx)

    PR[:, 0] = PP[:, 0] = PL[:, 0] = np.trapz(rgb[0, :] * illum * reflect, dx=dx)
    PR[:, 1] = PP[:, 1] = PL[:, 1] = np.trapz(rgb[1, :] * illum * reflect, dx=dx)
    PR[:, 2] = PP[:, 2] = PL[:, 2] = np.trapz(rgb[2, :] * illum * reflect, dx=dx)

    PP[:, 3] = PL[:, 0] * PL[:, 0]
    PP[:, 4] = PL[:, 1] * PL[:, 1]
    PP[:, 5] = PL[:, 2] * PL[:, 2]
    PP[:, 6] = PL[:, 0] * PL[:, 1]
    PP[:, 7] = PL[:, 1] * PL[:, 2]
    PP[:, 8] = PL[:, 2] * PL[:, 0]

    PR[:, 3] = np.sqrt(PP[:, 6])
    PR[:, 4] = np.sqrt(PP[:, 7])
    PR[:, 5] = np.sqrt(PP[:, 8])
    return Q, PL, PP, PR


def least_square(A, B):
    X = np.matmul(np.matmul(pinv(np.matmul(np.transpose(A), A)), np.transpose(A)), B)
    return X


def nelder_mead_simplex(A, B, x0):
    # x0 = rng.random(A.shape[1]*3)
    # f = lambda x: np.sum(np.sqrt(np.sum((np.matmul(A, np.reshape(x, (A.shape[1], 3))) - B) ** 2, 1)))
    f = lambda x: np.sum(np.sqrt(np.sum((xyz2lab(np.matmul(A, np.reshape(x, (A.shape[1], 3)))) - xyz2lab(B)) ** 2, 1)))
    res = minimize(f, x0, method='Nelder-Mead', tol=1e-6)
    result = res.x
    result = np.reshape(result, (A.shape[1], 3))
    return result


def xyz2lab(xyz, whitepoint=(95.05, 100.0, 108.88)):
    delta = 6 / 29
    step = lambda t, delta: np.heaviside(t - np.power(delta, 3.0), 1.0)
    f = lambda t: np.power(abs(t), (1.0 / 3.0)) * step(t, delta) + (t / (3 * np.power(delta, 2.0)) + 4 / 29) * (1.0 - step(t, delta))
    x = xyz[:, 0] / whitepoint[0]
    y = xyz[:, 1] / whitepoint[1]
    z = xyz[:, 2] / whitepoint[2]
    lab = np.zeros_like(xyz)
    xx = f(x)  # xx = np.array(list(map(f, x)))
    yy = f(y)  # yy = np.array(list(map(f, y)))
    zz = f(z)  # zz = np.array(list(map(f, z)))
    lab[:, 0] = 116 * yy - 16
    lab[:, 1] = 500 * (xx - yy)
    lab[:, 2] = 200 * (yy - zz)
    return lab


def xyz2lab_torch(xyz, whitepoint=(95.05, 100.0, 108.88)):
    delta = torch.tensor([6 / 29]).cuda()
    step = lambda t, delta: torch.heaviside(t - torch.pow(delta, 3.0), torch.tensor([1.0]).cuda())
    f = lambda t: torch.pow(torch.abs(t), 1.0 / 3.0) * step(t, delta) + (t / (3 * torch.pow(delta, 2.0)) + 4 / 29) * (1.0 - step(t, delta))
    x = xyz[:, 0] / whitepoint[0]
    y = xyz[:, 1] / whitepoint[1]
    z = xyz[:, 2] / whitepoint[2]
    lab = torch.zeros_like(xyz)
    xx = f(x)  # xx = torch.Tensor(list(map(f, x)), dtype=torch.float).cuda()
    yy = f(y)  # yy = torch.Tensor(list(map(f, y)), dtype=torch.float).cuda()
    zz = f(z)  # zz = torch.Tensor(list(map(f, z)), dtype=torch.float).cuda()
    lab[:, 0] = 116 * yy - 16
    lab[:, 1] = 500 * (xx - yy)
    lab[:, 2] = 200 * (yy - zz)
    return lab


def calculate_deltaE(P, Q, M):
    lab_input = xyz2lab(np.matmul(P, M))
    lab_target = xyz2lab(Q)
    residual = lab_input - lab_target
    error = np.sqrt(np.sum(np.power(residual, 2.0), axis=1))
    return error


def loss_deltaE(output, target):
    lab_output = xyz2lab_torch(output)
    lab_target = xyz2lab_torch(target)
    residual = lab_output - lab_target
    loss_vec = torch.sqrt(torch.sum(torch.pow(residual, 2.0), dim=1))
    loss = torch.mean(loss_vec)
    return loss, loss_vec


def calculate_deltaE2000(P, Q, M, Kl=1, Kc=1, Kh=1):
    lab_input = xyz2lab(np.matmul(P, M))
    lab_target = xyz2lab(Q)
    L1, a1, b1 = lab_input.transpose()
    L2, a2, b2 = lab_target.transpose()
    Lm = (L1 + L2) / 2.0
    C1 = np.sqrt(np.sum(np.power(lab_input[:, 1:], 2.0), axis=1))
    C2 = np.sqrt(np.sum(np.power(lab_target[:, 1:], 2.0), axis=1))
    Cm = (C1 + C2) / 2.0
    G = 0.5 * (1 - np.sqrt(np.power(Cm, 7.0) / (np.power(Cm, 7.0) + np.power(25.0, 7.0))))
    a1p = (1.0 + G) * a1
    a2p = (1.0 + G) * a2
    C1p = np.sqrt(np.power(a1p, 2.0) + np.power(b1, 2.0))
    C2p = np.sqrt(np.power(a2p, 2.0) + np.power(b2, 2.0))
    Cmp = (C1p + C2p) / 2.0
    h1p = np.degrees(np.arctan2(b1, a1p))
    h1p += (h1p < 0) * 360
    h2p = np.degrees(np.arctan2(b2, a2p))
    h2p += (h2p < 0) * 360
    Hmp = (((np.fabs(h1p - h2p) > 180) * 360) + h1p + h2p) / 2.0
    T = 1 - 0.17 * np.cos(np.radians(Hmp - 30)) + \
            0.24 * np.cos(np.radians(2 * Hmp)) + \
            0.32 * np.cos(np.radians(3 * Hmp + 6)) - \
            0.2 * np.cos(np.radians(4 * Hmp - 63))

    diff_h2p_h1p = h2p - h1p
    delta_hp = diff_h2p_h1p + (np.fabs(diff_h2p_h1p) > 180) * 360
    delta_hp -= (h2p > h1p) * 720

    delta_Lp = L2 - L1
    delta_Cp = C2p - C1p
    delta_Hp = 2 * np.sqrt(C2p * C1p) * np.sin(np.radians(delta_hp) / 2.0)

    S_L = 1 + ((0.015 * np.power(Lm - 50, 2)) / np.sqrt(20 + np.power(Lm - 50, 2.0)))
    S_C = 1 + 0.045 * Cmp
    S_H = 1 + 0.015 * Cmp * T

    delta_ro = 30 * np.exp(-(np.power(((Hmp - 275) / 25), 2.0)))
    R_C = np.sqrt((np.power(Cmp, 7.0)) / (np.power(Cmp, 7.0) + np.power(25.0, 7.0)))
    R_T = -2 * R_C * np.sin(2 * np.radians(delta_ro))

    error = np.sqrt(np.power(delta_Lp / (S_L * Kl), 2) + np.power(delta_Cp / (S_C * Kc), 2) + np.power(delta_Hp / (S_H * Kh), 2) + \
            R_T * (delta_Cp / (S_C * Kc)) * (delta_Hp / (S_H * Kh)))
    return error


def loss_deltaE2000(output, target, Kl=1, Kc=1, Kh=1):
    lab_output = xyz2lab_torch(output)
    lab_target = xyz2lab_torch(target)
    L1, a1, b1 = torch.t(lab_output)
    L2, a2, b2 = torch.t(lab_target)
    Lm = (L1 + L2) / 2.0
    C1 = torch.sqrt(torch.sum(torch.pow(lab_output[:, 1:], 2.0), dim=1))
    C2 = torch.sqrt(torch.sum(torch.pow(lab_target[:, 1:], 2.0), dim=1))
    Cm = (C1 + C2) / 2.0
    one = torch.tensor([1.0]).cuda()
    G = 0.5 * (1 - torch.sqrt(torch.pow(Cm, 7.0) / (torch.pow(Cm, 7.0) + torch.pow(one * 25, 7.0))))
    a1p = (one + G) * a1
    a2p = (one + G) * a2
    C1p = torch.sqrt(torch.pow(a1p, 2.0) + torch.pow(b1, 2.0))
    C2p = torch.sqrt(torch.pow(a2p, 2.0) + torch.pow(b2, 2.0))
    Cmp = (C1p + C2p) / 2.0
    h1p = torch.rad2deg(torch.atan2(b1, a1p))
    h1p += (h1p < 0) * 360
    h2p = torch.rad2deg(torch.atan2(b2, a2p))
    h2p += (h2p < 0) * 360
    Hmp = (((torch.abs(h1p - h2p) > 180) * 360) + h1p + h2p) / 2.0
    T = 1 - 0.17 * torch.cos(torch.deg2rad(Hmp - 30)) + \
            0.24 * torch.cos(torch.deg2rad(2 * Hmp)) + \
            0.32 * torch.cos(torch.deg2rad(3 * Hmp + 6)) - \
            0.2 * torch.cos(torch.deg2rad(4 * Hmp - 63))

    diff_h2p_h1p = h2p - h1p
    delta_hp = diff_h2p_h1p + (torch.abs(diff_h2p_h1p) > 180) * 360
    delta_hp -= (h2p > h1p) * 720

    delta_Lp = L2 - L1
    delta_Cp = C2p - C1p
    delta_Hp = 2 * torch.sqrt(C2p * C1p) * torch.sin(torch.deg2rad(delta_hp) / 2.0)

    S_L = 1 + ((0.015 * torch.pow(Lm - 50, 2)) / torch.sqrt(20 + torch.pow(Lm - 50, 2.0)))
    S_C = 1 + 0.045 * Cmp
    S_H = 1 + 0.015 * Cmp * T

    delta_ro = 30 * torch.exp(-(torch.pow(((Hmp - 275) / 25), 2.0)))
    R_C = torch.sqrt((torch.pow(Cmp, 7.0)) / (torch.pow(Cmp, 7.0) + torch.pow(one * 25, 7.0)))
    R_T = -2 * R_C * torch.sin(2 * torch.angle(delta_ro))

    loss_vec = torch.sqrt(torch.pow(delta_Lp / (S_L * Kl), 2) + torch.pow(delta_Cp / (S_C * Kc), 2) + torch.pow(delta_Hp / (S_H * Kh), 2) + \
            R_T * (delta_Cp / (S_C * Kc)) * (delta_Hp / (S_H * Kh)))
    loss = torch.mean(loss_vec)
    return loss, loss_vec


def run_model(net, device, nn_num_train, inputs, targets, input_test, output_test, lr, batch_size, epoch_size):
    net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    nn_index = list(range(nn_num_train))
    loss_y = np.zeros(epoch_size, 'float')
    iteration = int(nn_num_train / batch_size)

    for epoch in range(epoch_size):
    	rng.shuffle(nn_index)
    	for j in range(iteration):
    		optimizer.zero_grad()
    		sub = nn_index[j * batch_size:(j + 1) * batch_size]
    		input = Variable(inputs[sub, :], requires_grad=False)
    		target = Variable(targets[sub, :], requires_grad=False)
    		output = net(input)
    		loss, loss_vec = loss_deltaE2000(output, target)
    		loss.backward()
    		optimizer.step()
    	# print(f'Loss at epoch {epoch:-4d}: {loss}')
    	loss_y[epoch] = loss
    	# if np.mod(epoch, 100) == 0 or epoch == epoch_size - 1:
    	# 	torch.save(net.state_dict(), f'models/mlp3/section_{i}_epoch_{epoch:04d}.pt')

    output_test = net(input_test)
    loss_test, loss_vec_test = loss_deltaE2000(output_test, target_test)
    loss_vec_test = loss_vec_test.detach().cpu().numpy()
    return loss_y, loss_vec_test



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
reflect_interp_pca = loadmat('reflectance_pca_10000.mat')
reflect_train_interp = reflect_interp_pca['spectrum']


from models import MLP3, MLP5, MLP7, MLP9
cudnn.benchmark = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
lr = 0.001
batch_size = 16
epoch_size = 500

num = reflect.shape[0]
index = rng.permutation(num)
reflect = reflect[index, :]
sections = 5
limit = int(np.ceil(num / sections))
total = list(range(0, num))
deltaL = np.zeros(num, 'float')
deltaP = np.zeros(num, 'float')
deltaR = np.zeros(num, 'float')
deltaL_lab = np.zeros(num, 'float')
deltaP_lab = np.zeros(num, 'float')
deltaR_lab = np.zeros(num, 'float')
deltaNN3 = np.zeros(num, 'float')
deltaNN5 = np.zeros(num, 'float')
deltaNN7 = np.zeros(num, 'float')
deltaNN9 = np.zeros(num, 'float')
samples_interp = np.linspace(low, high, num=dim)
rgb_interp = interp_spectrum(rgb, samples5, samples_interp)

for i in range(sections):
    test_index = list(range(i*limit, min((i+1)*limit, num)))
    # train_index = list(set(total) - set(test_index))
    # reflect_train = reflect[train_index, :]
    reflect_test = reflect[test_index, :]
    # reflect_train_interp = interp_spectrum(reflect_train, samples4, samples_interp)
    reflect_test_interp = interp_spectrum(reflect_test, samples4, samples_interp)

    Q, PL, PP, PR = calculate_regression_matrix(spd, reflect_train_interp, xyz, rgb_interp)
    # Moore-Penrose inverse
    ML_ls = least_square(PL, Q)
    MP_ls = least_square(PP, Q)
    MR_ls = least_square(PR, Q)
    # Nelder-Mead simplex
    ML_simp = nelder_mead_simplex(PL, Q, ML_ls)
    MP_simp = nelder_mead_simplex(PP, Q, MP_ls)
    MR_simp = nelder_mead_simplex(PR, Q, MR_ls)

    Q_t, PL_t, PP_t, PR_t = calculate_regression_matrix(spd, reflect_test_interp, xyz, rgb_interp)
    deltaL[test_index] = calculate_deltaE2000(PL_t, Q_t, ML_ls)
    deltaP[test_index] = calculate_deltaE2000(PP_t, Q_t, MP_ls)
    deltaR[test_index] = calculate_deltaE2000(PR_t, Q_t, MR_ls)
    deltaL_lab[test_index] = calculate_deltaE2000(PL_t, Q_t, ML_simp)
    deltaP_lab[test_index] = calculate_deltaE2000(PP_t, Q_t, MP_simp)
    deltaR_lab[test_index] = calculate_deltaE2000(PR_t, Q_t, MR_simp)

    # NN approach data
    nn_num_train = reflect_train_interp.shape[0]
    nn_num_test = reflect_test.shape[0]
    inputs = torch.from_numpy(PL.astype('float32')).cuda()
    targets = torch.from_numpy(Q.astype('float32')).cuda()
    inputs_test = torch.from_numpy(PL_t.astype('float32')).cuda()
    targets_test = torch.from_numpy(Q_t.astype('float32')).cuda()
    input_test = Variable(inputs_test, requires_grad=False)
    target_test = Variable(targets_test, requires_grad=False)
    
    # 1st model
    net = MLP3()
    loss_y3, loss_vec_test = run_model(net, device, nn_num_train, inputs, targets, input_test, target_test, lr, batch_size, epoch_size)
    deltaNN3[test_index] = loss_vec_test
    print('Finish MLP3')

    # 2nd model
    net = MLP5()
    loss_y5, loss_vec_test = run_model(net, device, nn_num_train, inputs, targets, input_test, target_test, lr, batch_size, epoch_size)
    deltaNN5[test_index] = loss_vec_test
    print('Finish MLP5')

    # 3rd model
    net = MLP7()
    loss_y7, loss_vec_test = run_model(net, device, nn_num_train, inputs, targets, input_test, target_test, lr, batch_size, epoch_size)
    deltaNN7[test_index] = loss_vec_test
    print('Finish MLP7')

    # 4th model
    net = MLP9()
    loss_y9, loss_vec_test = run_model(net, device, nn_num_train, inputs, targets, input_test, target_test, lr, batch_size, epoch_size)
    deltaNN9[test_index] = loss_vec_test
    print('Finish MLP9')
    print(f'round {i} is complete')

deltaL_sort = np.sort(deltaL)
deltaP_sort = np.sort(deltaP)
deltaR_sort = np.sort(deltaR)
deltaL_lab_sort = np.sort(deltaL_lab)
deltaP_lab_sort = np.sort(deltaP_lab)
deltaR_lab_sort = np.sort(deltaR_lab)
deltaNN3_sort = np.sort(deltaNN3)
deltaNN5_sort = np.sort(deltaNN5)
deltaNN7_sort = np.sort(deltaNN7)
deltaNN9_sort = np.sort(deltaNN9)
ind50 = int(num / 2)
ind95 = int(num * 0.95)
# print(f'LCC mean     {np.average(deltaL)}, median {deltaL_sort[ind50]}, 95% {deltaL_sort[ind95]}')
# print(f'PCC mean     {np.average(deltaP)}, median {deltaP_sort[ind50]}, 95% {deltaP_sort[ind95]}')
# print(f'RCC mean     {np.average(deltaR)}, median {deltaR_sort[ind50]}, 95% {deltaR_sort[ind95]}')
# print(f'LCC-lab mean {np.average(deltaL_lab)}, median {deltaL_lab_sort[ind50]}, 95% {deltaL_lab_sort[ind95]}')
# print(f'PCC-lab mean {np.average(deltaP_lab)}, median {deltaP_lab_sort[ind50]}, 95% {deltaP_lab_sort[ind95]}')
# print(f'RCC-lab mean {np.average(deltaR_lab)}, median {deltaR_lab_sort[ind50]}, 95% {deltaR_lab_sort[ind95]}')
print(f'NN3 mean     {np.average(deltaNN3)}, median {deltaNN3_sort[ind50]}, 95% {deltaNN3_sort[ind95]}')
print(f'NN5 mean     {np.average(deltaNN5)}, median {deltaNN5_sort[ind50]}, 95% {deltaNN5_sort[ind95]}')
print(f'NN7 mean     {np.average(deltaNN7)}, median {deltaNN7_sort[ind50]}, 95% {deltaNN7_sort[ind95]}')
print(f'NN9 mean     {np.average(deltaNN9)}, median {deltaNN9_sort[ind50]}, 95% {deltaNN9_sort[ind95]}')

kernel_size = 15
kernel = np.ones(kernel_size) / kernel_size
loss_x = np.array(list(range(epoch_size)))
plt.figure(num=4)
plt.plot(loss_x, np.convolve(loss_y3, kernel, mode='same'), "r-", label="3 layers")
plt.plot(loss_x, np.convolve(loss_y5, kernel, mode='same'), "g-", label="5 layers")
plt.plot(loss_x, np.convolve(loss_y7, kernel, mode='same'), "b-", label="7 layers")
plt.plot(loss_x, np.convolve(loss_y9, kernel, mode='same'), "m-", label="9 layers")
plt.legend(title='')
plt.title('Smoothed loss curve with number of training epochs')
plt.show()