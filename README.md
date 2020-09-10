[![Build Status](https://travis-ci.org/valentina-s/time-series-nmf.svg?branch=master)](https://travis-ci.org/valentina-s/time-series-nmf)

## time-series-nmf  

`time-series-nmf` is a Python package implementing non-negative matrix factorization for time series data. Currently, it supports a version with Tikhonov regularization and sparse constraints as proposed by [Fabregat R.. et. al.](https://arxiv.org/abs/1910.14576) and implemented in Matlab in https://github.com/raimon-fa/palm-nmf. It uses a [PALM](https://link.springer.com/article/10.1007/s10107-013-0701-9) optimization scheme. We plan to add other models and optimization algorithms. 

This work is under continuous development. 

### Installation:

```bash
# from github (latest)
pip install git+https://github.com/valentina-s/time-series-nmf

# from pipy (stable)
pip install time-series-nmf 
```

### Getting started:
```python
import tsnmf
    
# generate some data
from numpy.random import rand
data = rand(100,1000)
    
# fit time series nmf to data
model = tsnmf.smoothNMF(n_components=5)
model.fit(data)
    
# outputs
model.W
model.H

```

### Acknowledgements
This work has been supported by NSF award #1849930 and the Betty and Gordon Moore and Alfred P. Sloan Foundations Data Science Environments grant (MSDSE).


