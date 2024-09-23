from sepws.scattering import filterbank
import matplotlib.pyplot as plt
import numpy as np
from sepws.scattering.config import cfg

cfg.set_alpha(1,    2.5, False)
cfg.set_alpha(1,    1.8, True)
cfg.set_beta(1,     2.5)

N = [256, 256]
d = [8, 8]
Npad = [filterbank.calculate_padding_1d(n, di)[2] for n, di in zip(N, d)]
print(Npad)

fb = filterbank.scattering_filterbank_separable(Npad, d, [[1], [1]], allow_ds=[False, False])
lambdas = filterbank.get_Lambda_set(fb, 0, [1, 1])
print(len(lambdas))
for l in lambdas:
    print(f'{l[0]:.3f}, {l[1]:.3f}')


fig = plt.figure()
ax = fig.add_subplot(111)
X = np.linspace(-0.5, 0.5, Npad[1])
Y = np.linspace(-0.5, 0.5, Npad[0])

s = np.zeros(Npad)


c = 0.7

for l in lambdas:
    psi0 = filterbank.get_wavelet_filter(fb, 0, 0, 1, l[0])
    psi1 = filterbank.get_wavelet_filter(fb, 1, 0, 1, l[1])
    psi = psi0[:, None] * psi1[None, :] # broadcast for getting wavelet
    s += psi**2
    print(np.max(psi))     
    ax.contour(X, Y, np.fft.fftshift(psi), [c], colors='k')
    
    
phi0 = filterbank.get_wavelet_filter(fb, 0, 0, 1, 0)
phi1 = filterbank.get_wavelet_filter(fb, 1, 0, 1, 0)
phi = phi0[:, None] * phi1[None, :] # broadcast for getting wavelet
plt.contour(X, Y, np.fft.fftshift(phi), [c], colors='k', linestyles='dashed')

ax.axvline(c='k', lw=1)
ax.axhline(c='k', lw=1)
# ax.axis('off')
    
plt.ylim([-0.05, 0.5])

# fig = plt.figure()

# ax = fig.add_subplot(111, projection='3d')

# x, y = np.meshgrid(X, Y)
# ax.plot_surface(x, y, np.fft.fftshift(s))

plt.show()
    
    