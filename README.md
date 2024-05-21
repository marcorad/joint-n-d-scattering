# Separable Wavelet Scattering
Separable wavelet scattering Implementation with Pytorch, similar to Kymatio implementations. This implementation only performs Fourier Convolutions.

Results for MNIST and MedMNIST3D are used to benchmark the feature-extraction capabilities of this transform.

## MNIST
mnist_clf_nn.py is the classification experiment used in the paper. mnist_clf.py repeats one of the original 2D scattering experiments with LDA, in which SWS is slightly worse (as expected, from separable filters).
Difference when using a NN is negligible.

## MedMNIST3D

Run medmnist3d_features.py before medmnist_clf.py to repeat benchmark experiment. The original benchmarks are present in medmnist3d_benchmark.txt.

