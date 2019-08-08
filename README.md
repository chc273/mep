# Minimum Energy Path Tools

## Introduction 
This package contains various methods for finding the minimal energy path in atom simulations.

Currently the following methods are implemented:

> Nudged elastic band method


## How to use

```python

from mep.optimize import ScipyOptimizer
from mep.path import Path
from mep.neb import NEB
from mep.models import LEPS

leps = LEPS() # Test model 
op = ScipyOptimizer(leps) # local optimizer for finding local minima
x0 = op.minimize([1, 4], bounds=[[0, 4], [-2, 4]]).x # minima one
x1 = op.minimize([3, -1], bounds=[[0, 4], [-2, 4]]).x # minima two


path = Path.from_linear_end_points(x0, x1, 101, 1)  # set 101 images, and k=1
neb =NEB(leps, path) # initialize NEB
history = neb.run(verbose=True) # run

```

The results will be like the following

![LEPS example](./assets/leps.gif)
