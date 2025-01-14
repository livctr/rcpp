# RCPP Implementation

`conda create --name rcpp`

`conda activate rcpp`

`pip install -r requirements.txt`

Download the kaggle dataset to `credit-scoring/data`.


## Quantile Risk Control Install Guide

`git submodule add https://github.com/jakesnell/quantile-risk-control.git`

`cd quantile-risk-control`

`make env-init` - use `torch` (vs. `torch=1.11.0`) and no `torchvision`

`conda activate var_control`

Next step: install `crossing-probability`

- `module load fftw/intel/3.3.9` for installing fftw lib
- Ran `make`, added `#include <limits>` to `src/common.cc` and `#include <iterator>` to `src/common.hh`
- `swig` isn't recognized: see https://www.linuxfromscratch.org/blfs/view/svn/general/swig.html
