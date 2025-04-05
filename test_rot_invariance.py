

import numpy as np

import matplotlib.pyplot as plt
from sepws.scattering.separable_scattering import SeparableScattering
from sepws.scattering.filterbank import get_Lambda_set
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

def rot_invariance(x: Tensor, lambdas):
    ds = []
    l_np = np.array(lambdas)
    l1 = list(np.unique(l_np[:, 0]))
    l2 = list(np.unique(l_np[:, 1]))
    l3 = list(np.unique(l_np[:, 2]))
        
    rings = []
    start_y = len(l2)//2 + 1
    n_x = 1
    n_y = 3
    
    while n_y <= len(l2):
        
        ring = []
    
        # move right
        for i in range(n_x):
            ring.append((l1[i], l2[start_y]))
            
        # move down
        for j in range(n_y):
            ring.append((l1[n_x], l2[start_y - j]))
            
        # move left
        for i in range(n_x):
            ring.append((l1[i], l2[start_y - n_y + 1]))
            
        # increase ring size
        n_x += 1
        n_y += 2
        start_y += 1
        
        rings.append(ring)
        
    n = 0
    for r in rings: n += len(r)
    
    # make sure we cover all possibilities
    assert n == len(l1) * len(l2) - 1, 'all cases not covered'
    
    y = [x[[0], :, :, :]]
    for ring in rings:
        idx = []
        for l in ring:
            for d in l3:
                i = lambdas.index((l[0], l[1], d))
                idx.append(i)
            y.append(torch.sum(x[idx, :, :, :]))
            
    return torch.concat(y, dim=0)
            
    
        
        
        
    

d_im = 4
d_hyp = 8
Q_im, Q_hyp = 1, 1
sws = SeparableScattering([145, 145, 200], [d_im, d_im, d_hyp], [[Q_im], [Q_im], [Q_hyp]], allow_ds=[False, False, True])
lambdas = get_Lambda_set(sws.fb, 0, [1]*3)
rot_invariance(lambdas)