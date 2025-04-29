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
from sklearn.svm import SVC
from sepws.dataprocessing import hsi
import gc
from scipy.fftpack import dct    
from torch import Tensor
from sepws.scattering.filterbank import get_Lambda_set
from sklearn.decomposition import PCA

Q = 1
d_hyp_configs = [8, 16, 32]
d_im_configs =  [4, 4 ,  4]

cfg.cuda()

cfg.set_alpha(1,    2.5, False)
cfg.set_alpha(1,    1.8, True)
cfg.set_beta(1,     2.5)

# cfg.set_alpha(Q,    3.5, False)
# cfg.set_alpha(Q,    3.5, True)
# cfg.set_beta(Q,     2.5)

hsi_data = hsi.load()

for hsi_im in hsi_data[:]:
    gc.collect()
    
    if hsi_im['name'] not in ['indian_pines_corrected', 'KSC', 'paviaU']: continue
    
    print('\n------------------------')
    print(hsi_im['name'])
    print('------------------------\n')

    res = []

    for d_im, d_hyp in zip(d_im_configs, d_hyp_configs):
        gc.collect()
        
        
        X = hsi_im['image'].astype(np.float32)
        labels = hsi_im['labels']
        
        im_shape = X.shape

        print(X.shape, labels.shape)
        
        torch.cuda.empty_cache()

        sws = SeparableScattering(list(X.shape), [d_im, d_im, d_hyp], [[Q], [Q], [Q]], allow_ds=[False, False, True], remove_highly_corr_filter=True)
        X = torch.from_numpy(X[None, :, :, :]).cuda()
        s = sws.scattering(X).cpu().numpy()[0, :, :, :, :]
        
        
        # print(s.shape)
        # unpadding disabled when downsampling is disabled for any dimension, so do it manually
        nb = d_im // 2
        s = s[:, nb:(nb+X.shape[1]), nb:(nb+X.shape[2]), 1:-1]
        print(s.shape)

        # s = dct(dct(s, axis=0), axis=3)


        # print(np.unique(labels, return_counts=True)) 
        # exit()


        x = np.swapaxes(s, 0, 2).swapaxes(0, 1).reshape((im_shape[0], im_shape[1], -1)).reshape((im_shape[0]*im_shape[1], -1))
        del s
        labels = labels.reshape(im_shape[0]*im_shape[1])
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
        
        gc.collect()

        acc = []
        Ntrails = 20
        for i in tqdm(list(np.random.randint(0,high=100000, size=(Ntrails,)))):
            # print(i)
            # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=i, shuffle=True, stratify=y)
            train_idx, test_idx = split(y)
            X_train, y_train = X[train_idx, :], y[train_idx]
            X_test, y_test = X[test_idx, :], y[test_idx]
            
            
            mu = 0 # np.mean(X_train, axis=0)
            std = 1 # np.std(X_train, axis=0)
            X_train = (X_train - mu) / std
            X_test = (X_test - mu) / std
            
            # pca = PCA(n_components=1000)
            # pca.fit(np.concatenate([X_train, X_test], 0))
            # X_train = pca.transform(X_train)
            # X_test = pca.transform(X_test)
            # print(X_train.shape)

            clf = LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto')
            # clf = LinearDiscriminantAnalysis(solver='svd')
            # clf = SVC(kernel='poly', degree=2)
            # clf = GaussianNB()
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            acc.append(np.sum(y_pred == y_test) / len(y_pred))
            
        # print(acc)
        oa = np.array(acc)*100
        print('Randomised trials overall accuracy (%) result')
        print(f'd = {(d_im, d_im, d_hyp)}, Q = {(Q, Q, Q)}: {np.mean(oa):.2f} +- {np.std(oa):.2f} min {np.min(oa):.2f} max {np.max(oa):.2f}')
        
        res += [{
            'Q': (Q, Q, Q),
            'd': (d_im, d_im, d_hyp),
            'mean': np.mean(oa),
            'std': np.std(oa),
            'min': np.min(oa),
            'max': np.max(oa),
            'raw': oa.tolist(),
            'nfeat': X_train.shape[1]
        }]
        
    import json
    with open(f"/home/marco/Repos/seperable-wavelet-scattering/{hsi_im['name']}-results.json", 'w') as file:
        json.dump(res, file, indent=4)
    

# plt.show()

