import numpy as np

import matplotlib.pyplot as plt
from sepws.scattering.separable_scattering import SeparableScattering
import torch
from sepws.scattering.config import cfg
from skimage.measure import block_reduce
from scipy.stats import mode
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn import metrics

S = []
for i in range(1, 200+1):
    s = np.loadtxt(f'/home/marco/Repos/seperable-wavelet-scattering/indian-pines/{i}.csv', delimiter=',')    
    s = s[:, :, None]
    S.append(s)
    
X = np.concatenate(S, axis=2)
labels = np.loadtxt(f'/home/marco/Repos/seperable-wavelet-scattering/indian-pines/labels.csv', delimiter=',')

print(X.shape, labels.shape)

plt.figure()    
plt.subplot(121)
plt.imshow(labels/16, cmap='Set1')
plt.subplot(122)
plt.imshow(np.mean(X, axis=2))

print(np.unique(labels, return_counts=True))



cfg.cuda()
sws = SeparableScattering(list(X.shape), [2, 2, 8], [[1], [1], [4]])
X = torch.from_numpy(X[None, :, :, :]).cuda()
s = sws.scattering(X).cpu().numpy()[0, :, :, :]
print(s.shape)

print(np.unique(labels, return_counts=True))

labels = block_reduce(labels[0:-1, 0:-1], (2, 2), np.min)
x = np.swapaxes(s, 0, 2).swapaxes(0, 1).reshape((72,72, -1)).reshape((72*72, -1))
print(x.shape, labels.shape)
labels = labels.reshape(72*72)
#drop 0
idx = labels != 0
X = x[idx, :]
y = labels[idx]
print(X.shape, y.shape)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=321, shuffle=True)

lda = LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto')
lda.fit(X_train, y_train)
y_pred = lda.predict(X_test)
print(np.sum(y_pred == y_test) / len(y_pred))

y_im = lda.predict(x)
y_im = y_im.reshape((72, 72))
y_im[0,0]=0

cm = metrics.confusion_matrix(y_test, y_pred)
# disp = metrics.ConfusionMatrixDisplay()

metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_pred)

plt.figure()
plt.imshow(y_im/16, cmap='Set1')

plt.show()

