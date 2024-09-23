import numpy as np
import pickle as pkl

S = []
for i in range(1, 200+1):
    s = np.loadtxt(f'/home/marco/Repos/seperable-wavelet-scattering/indian-pines/{i}.csv', delimiter=',')    
    s = s[:, :, None]
    S.append(s)
    
X = np.concatenate(S, axis=2)
labels = np.loadtxt(f'/home/marco/Repos/seperable-wavelet-scattering/indian-pines/labels.csv', delimiter=',')

with open('indian-pines/indian-pines.pkl', 'wb') as file:
    pkl.dump((X, labels), file)