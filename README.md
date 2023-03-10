# eindiff
Automatic differentiation using einsum notation. Unlike pytorch, tensorflow, or autograd, you can write your function in standard numpy and `eindiff` will compute jgradients, jacobians, and hessians from it.

```python
import numpy as np

def logistic(y, X, beta):
  p = 1/(1+np.exp(-X@beta))
  return np.sum(y*np.log(p)+(1-y)*np.log(1-p))
  
from eindiff.reverse import derivative_of

jh = derivative_of(logistic, argno=2, jacobian=True, hessian=True)

jacobian, hessian = jh(y, X, np.random.rand(10))
```

