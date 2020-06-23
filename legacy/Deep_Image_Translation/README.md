## Deep Image Translation
This subfolder contains the Python code developed for the empirical experiments carried out and presented in the paper *Deep Image Translation with an Affinity-Based Change Prior for Unsupervised Multimodal Change Detection* which can be found at https://arxiv.org/abs/2001.04271.

Please refer to the paper for details. You are more than welcome to use the code, but acknowledge us by referencing to our paper and to this Github repository!

# How to use

Each file stands on its own and can be used separately from the rest.
**SCCN.py** and **CAN.py** are the main files of the implemented state-of-the-art methods used for comparison. They refer to these two papers:
https://doi.org/10.1109/TNNLS.2016.2636227 and https://doi.org/10.1109/LGRS.2018.2868704.
This code was developed on an old machine, whose OS and environment is reproduced by the Docker image created with the Dockerfile in the docker subfolder.
In alternative, one can download the Docker image from my Docker hub repository, tagged *llu025/lenny:gpu* (in case you have an NVIDIA gpu on which CUDA can run) or *llu025/lenny:nogpu* otherwise.


# Available data

Two datasets were used for this project.
The Texas dataset can be found at Michele Volpi's webpage (https://sites.google.com/site/michelevolpiresearch/codes/cross-sensor) and it is associated with this paper https://doi.org/10.1016/j.isprsjprs.2015.02.005. Please cite!

The California dataset was created for this project, and can be found at https://sites.google.com/view/luppino/data.
