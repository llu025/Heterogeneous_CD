## Code-Aligned Autoencoders
This subfolder contains the Python code developed for the empirical experiments carried out and presented in the paper *Code-Aligned Autoencoders for Unsupervised Change Detection in Multimodal Remote Sensing Images* which can be found at https://arxiv.org/abs/2004.07011.

Please refer to the paper for details. You are more than welcome to use the code, but acknowledge us by referencing to our paper and to this Github repository!

# How to use

**Code-Aligned_Autoencoders.py** is the main file.
**SCCN.py** and **cGAN.py** are the main files of the implemented state-of-the-art methods used for comparison. They refer to these two papers:
https://doi.org/10.1109/TNNLS.2016.2636227 and https://doi.org/10.1109/LGRS.2018.2868704.
We recommend the use of the Docker image found at the repository of madsadrian tagged
*madsadrian/tensorflow:latest*

# Available data

Four datasets were used for this project.

The Texas dataset can be found at Michele Volpi's webpage (https://sites.google.com/site/michelevolpiresearch/codes/cross-sensor) and it is associated with this paper https://doi.org/10.1016/j.isprsjprs.2015.02.005. Please cite!

The California dataset was created especially for this project, and can be found at https://sites.google.com/view/luppino/data.

The France and Italy dataset were provided privately and cannot be shared until further notice.
