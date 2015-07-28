---
title: Geodesic Distance and its Derivatives
layout: default
---

## Geodesic Distance

For two points on a unit sphere $$(x_1, y_1, z_1)$$ and $$(x_2, y_2, z_2)$$, their geodesic distance is the same as the angle corresponding to the arc.
Let $$\alpha$$ denote the angle (which is also the geodesic distance), and $$\cos(\alpha)$$ would be the dot product of the two vectors.

$$ \cos(\alpha) = x_{1}x_{2} + y_{1}y_{2} + z_{1}z_{2} $$

Therefore

$$ \text{gds} = \alpha = \arccos(x_{1}x_{2} + y_{1}y_{2} + z_{1}z_{2}) $$

## Computing Derivatives of Geodesic Distance

### Conclusions

$$ \frac{\partial\alpha}{\partial\phi_1} = \frac{1}{\sin\alpha} \times (\sin\phi_1\cos\phi_2 - \cos\phi_1\sin\phi_2\cos(\theta_1 - \theta_2)) $$

$$ \frac{\partial\alpha}{\partial\theta_1} = \frac{1}{\sin\alpha} \times (\sin\phi_1\sin\phi_2\sin(\theta_1 - \theta_2)) $$

### Proof

Let $$p = x_{1}x_{2} + y_{1}y_{2} + z_{1}z_{2}$$. The derivative of $$\arccos(p)$$ is

$$ \arccos'(p) = -\frac{1}{\sqrt{1-p^2}} = -\frac{1}{\sin(\alpha)}$$

Using spherical coordinates, $$(x_1, y_1, z_1)$$ can be denoted as $$(\sin\phi_1\cos\theta_1, \sin\phi_1\sin\theta_1, \cos\theta_1)$$,
and $$(x_2, y_2, z_2)$$ as $$(\sin\phi_2\cos\theta_2, \sin\phi_2\sin\theta_2, \cos\theta_2)$$.

Therefore we have

$$ p = \sin\phi_1\sin\phi_2\cos\theta_1\cos\theta_2 + \sin\phi_1\sin\phi_2\sin\theta_1\sin\theta_2 + \cos\phi_1\cos\phi_2$$

Therefore the partial derivatives of gds are:

$$ \frac{\partial\alpha}{\partial\phi_1} = \frac{d\alpha}{dp} \times \frac{\partial p}{\partial\phi_1} = -\frac{1}{\sqrt{1-p^2}} \times \frac{\partial p}{\partial\phi_1}$$

In which

$$
\begin{eqnarray}
\frac{\partial p}{\partial\phi_1} &=& \cos\phi_1\sin\phi_2\cos\theta_1\cos\theta_2 + \cos\phi_1\sin\phi_2\sin\theta_1\sin\theta_2 - \sin\phi_1\cos\phi_2 \\
&=& \cos\phi_1\sin\phi_2\cos(\theta_1 - \theta_2) - \sin\phi_1\cos\phi_2
\end{eqnarray}
$$

And therefore

$$ \frac{\partial\alpha}{\partial\phi_1} = \frac{1}{\sin\alpha} \times (\sin\phi_1\cos\phi_2 - \cos\phi_1\sin\phi_2\cos(\theta_1 - \theta_2)) $$

Similarly,

$$
\begin{eqnarray}
\frac{\partial p}{\partial\theta_1} &=& -\sin\phi_1\sin\phi_2\sin\theta_1\cos\theta_2 + \sin\phi_1\sin\phi_2\cos\theta_1\sin\theta_2 \\
&=& \sin\phi_1\sin\phi_2\sin(\theta_2 - \theta_1)
\end{eqnarray}
$$

And therefore

$$ \frac{\partial\alpha}{\partial\theta_1} = \frac{1}{\sin\alpha} \times (\sin\phi_1\sin\phi_2\sin(\theta_1 - \theta_2)) $$
