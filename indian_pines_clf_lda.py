import numpy as np

import matplotlib.pyplot as plt
from sepws.scattering.separable_scattering import SeparableScattering
import torch
from sepws.scattering.config import cfg
from skimage.measure import block_reduce
from scipy.stats import mode
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
from tqdm import tqdm



# plt.figure()    
# plt.subplot(121)
# plt.imshow(labels/16, cmap='Set1')
# plt.subplot(122)
# plt.imshow(np.mean(X, axis=2))

# print(np.unique(labels, return_counts=True))


d_hyp_configs = [8, 16, 32]
Q_hyp_configs = [1, 1, 1]
d_im_configs =  [4, 4, 4]
Q_im_configs =  [1, 1, 1]
cfg.cuda()
cfg.set_alpha(1,    2.5, False)
cfg.set_alpha(1,    1.8, True)
cfg.set_beta(1,     2.5)

res = []

for d_im, d_hyp, Q_im, Q_hyp in zip(d_im_configs, d_hyp_configs, Q_im_configs, Q_hyp_configs):
    
    S = [] 
    for i in range(1, 200+1):
        s = np.loadtxt(f'/home/marco/Repos/seperable-wavelet-scattering/indian-pines/{i}.csv', delimiter=',')    
        s = s[:, :, None]
        S.append(s)
        
    X = np.concatenate(S, axis=2)
    labels = np.loadtxt(f'/home/marco/Repos/seperable-wavelet-scattering/indian-pines/labels.csv', delimiter=',')

    # print(X.shape, labels.shape)
    
    torch.cuda.empty_cache()

    sws = SeparableScattering(list(X.shape), [d_im, d_im, d_hyp], [[Q_im], [Q_im], [Q_hyp]], allow_ds=[False, False, True])
    X = torch.from_numpy(X[None, :, :, :]).cuda()
    s = sws.scattering(X).cpu().numpy()[0, :, :, :, :]
    # print(s.shape)
    # unpadding disabled when downsampling is disabled for any dimension, so do it manually
    nb = d_im // 2
    ne = nb + 145
    s = s[:, nb:ne, nb:ne, 1:-1]
    # print(s.shape)



    # print(np.unique(labels, return_counts=True))
    # exit()


    x = np.swapaxes(s, 0, 2).swapaxes(0, 1).reshape((145,145, -1)).reshape((145*145, -1))
    labels = labels.reshape(145*145)
    # print(x.shape, labels.shape)
    #drop 0
    idx = labels != 0
    X = x[idx, :]
    y = labels[idx]
    # print(X.shape, y.shape)

    # plt.figure()
    # plt.imshow(y_im/16, cmap='Set1')
    
    def split(y: np.ndarray, n=15):
        N = len(y)
        train_idx = [False for _ in range(N)]
        idx = [i for i in range(N)]
        for yu in np.unique(y):
            train_samples = np.nonzero(y == yu)[0]            
            np.random.shuffle(train_samples)
            train_samples = train_samples[:n] 
            for k in train_samples:
                train_idx[k] = True
                
        train_idx_ret = []
        test_idx_ret = []
        for i in range(N):
            if train_idx[i]:
                train_idx_ret.append(i)
            else:
                test_idx_ret.append(i)        
        return train_idx_ret, test_idx_ret
    
    # train_idx, test_idx = split(y)
    # print(np.unique(y[train_idx], return_counts=True))
    

    acc = []
    Ntrails = 20
    for i in tqdm(list(np.random.randint(0,high=100000, size=(Ntrails,)))):
        # print(i)
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=i, shuffle=True, stratify=y)
        train_idx, test_idx = split(y)
        X_train, y_train = X[train_idx, :], y[train_idx]
        X_test, y_test = X[test_idx, :], y[test_idx]
        
        # print(X_train.shape)
        
        mu = np.mean(X_train, axis=0)
        std = np.std(X_train, axis=0)
        X_train = (X_train - mu) / std
        X_test = (X_test - mu) / std

        clf = LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto')
        # clf = GaussianNB()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc.append(np.sum(y_pred == y_test) / len(y_pred))
        
    # print(acc)
    oa = np.array(acc)*100
    print('Randomised trials overall accuracy (%) result')
    print(f'd = {(d_im, d_im, d_hyp)}, Q = {(Q_im, Q_im, Q_hyp)}: {np.mean(oa):.2f} +- {np.std(oa):.2f} min {np.min(oa):.2f} max {np.max(oa):.2f}')
    
    res += [{
        'Q': (Q_im, Q_im, Q_hyp),
        'd': (d_im, d_im, d_hyp),
        'mean': np.mean(oa),
        'std': np.std(oa),
        'min': np.min(oa),
        'max': np.max(oa),
        'raw': oa.tolist(),
        'nfeat': X_train.shape[1]
    }]
    
import json
with open('/home/marco/Repos/seperable-wavelet-scattering/indian-pines-results.json', 'w') as file:
    json.dump(res, file, indent=4)
    

# plt.show()

