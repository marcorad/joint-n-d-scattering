# Joint Wavelet Scattering
Joint n-dimensional wavelet scattering Implementation with Pytorch, similar to Kymatio implementations. This implementation only performs Fourier convolutions. Direct computation may be faster in certain circumstances, but is left for future work.

Results for MNIST and MedMNIST3D are used to benchmark the feature-extraction capabilities of this transform. Four different hyperspectral images (HSIs) are evaluated.

To run these scripts, an NVidia GPU is required with at least 12 GB of VRAM. At least 32 GB of RAM is required. You may not be able to run some of these scripts without code modification if there hardware requirements are not met.

You may reduce the batch size of joint scattering computations if you don't have enough VRAM. 32GB of RAM is required for the LDA classifiers of some experiments.

## MNIST
mnist_clf.py is the classification experiment used in the paper. mnist_clf.py repeats one of the original 2D scattering experiments with LDA, in which joint scattering is slightly worse (as expected, from separable filters).
Difference when using a NN is negligible. The TeX table of results is roughly produced in the mnist-results folder.

## MedMNIST3D

Run medmnist_clf.py to repeat experiment (results may vary slightly due to randomness). The original benchmarks are present in medmnist3d_benchmark.txt. Features are cached in the medmnist3d-feats folder, which must be produced by running the medmnist3d_features.py script. 

## HSI

Run hsi_clf_lda.py to repeat the experiment, which utilises regularised LDA and a basic gridsearch for filter bank parameter optimisation. Depending on your hardware, it may take a while. Results are cached in the hsi-results/ folder.

