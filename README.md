# eindiff
Automatic differentiation using einsum notation. Unlike pytorch, tensorflow, or autograd, you can write your function in standard numpy and `eindiff` will compute jgradients, jacobians, and hessians from it.

```python
import numpy as np

def logistic(y, X, beta):
  p = 1/(1+np.exp(-X@beta))
  return np.sum(y*np.log(p)+(1-y)*np.log(1-p))
  
jh = derivative_of(logistic, jacobian=True, hessian=True)

jacobian, hessian = jh(np.random.rand(10))
```

