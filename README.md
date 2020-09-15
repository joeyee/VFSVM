# VFSVM
Small-floating Target Detection in Sea Clutter by Classifying Visual Feature in Time Doppler Spectra of IPIX data sets

![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)   
This is the **python** implementation of the - 
[Small-floating Target Detection in Sea Clutter by Classifying Visual Feature in Time Doppler Spectra](https://arxiv.org/abs/2009.04185).

%[Source paper](https://github.com/joeyee/MKCF/blob/master/MKCF_SourcePaper_SingleColumn.pdf) of the preprint version with supplementary %materials could be download at Github.

## Requirements
- python - 3.6.5
- opencv-python
- sklearn
- netCDF - 1.5.3

## How to use the code

### Step 1
Download the complex-sequential returns from the [IPIX 1993](http://soma.ece.mcmaster.ca/ipix/dartmouth/datasets.html#target) and [IPIX 1998](http://soma.mcmaster.ca/ipix.php)

We rewrite the Python code to load the IPIX raw data in 'Load_IPIX_xxxx.py'.

Convert the sequential returns to Time Doppler Spectra (TDS) images.

### Step 2
Compute the Local Binary Patterns (LBP) histogram for each TDS images.

### Step 3
Train the v-SVM with impure samples.

### Step 4
Sort the distances to the learned center. Select the sample with maximal distance as the target. 

## Introduction
This algorithm is introduced in [paper](https://arxiv.org/abs/2009.04185), which is under review. Once the paper is allowed to be published, we will release it soon.
Now we have published the data loading code, which is a Python re-implementation according to the Matlab version on the IPIX website.
## Bibtex

@misc{zhou2020smallfloating,
    title={Small-floating Target Detection in Sea Clutter via Visual Feature Classifying in the Time-Doppler Spectra},
    author={Yi Zhou and Yin Cui and Xiaoke Xu and Jidong Suo and Xiaoming Liu},
    year={2020},
    eprint={2009.04185},
    archivePrefix={arXiv},
    primaryClass={eess.SP}
}




