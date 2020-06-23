## Image Regression
This subfolder contains the MATLAB code developed for the empirical experiments carried out and presented in the paper *Unsupervised Image Regression for Heterogeneous Change Detection* which can be found at https://doi.org/10.1109/TGRS.2019.2930348.

Please refer to the paper for details. You are more than welcome to use the code, but acknowledge us by referencing to our paper and to this Github repository!

# How to use

**Affinity_matrices.m** takes in input two images and evaluate the change prior to select a training set of unchanged pixels to be used to perform image regression. The ground truth is only needed to check the quality of the output.
Once the training set is selected, one of the 4 regression methods can be used.

The *multi-output support vector regression* code is provided by Gustau Camps-valls at https://www.uv.es/gcamps/code/msvr-2-1.zip and it was developed for this paper: https://doi.org/10.1109/LGRS.2011.2109934

**HPT.m** implementes the *Homogeneous Pixel Transformation* method proposed in https://doi.org/10.1109/TIP.2017.2784560

We recommend the use of the MATLAB package **ParforProgress2** available at https://github.com/kotowicz/matlab-ParforProgress2


# Available data

Two datasets were used for this project.
The Texas dataset can be found at Michele Volpi's webpage (https://sites.google.com/site/michelevolpiresearch/codes/cross-sensor) and it is associated with this paper https://doi.org/10.1016/j.isprsjprs.2015.02.005. Please cite!

The California dataset was created for this project, and can be found at https://sites.google.com/view/luppino/data.
