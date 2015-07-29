---
title: Regularization
layout: default
---

## Implementation

### Computation of Metric Terms

Input:

- `nbrs`
- `cart_coords`; `rho` for `gds`; `coord_maps` for derivatives
- `orig_metric_distances`
- `resolution`, used to calculate `gds_derivatives`

Depends on:

- `compute_geodesic_distances`
- `gds_derivatives` for computing derivatives.

### Computation of Areal Terms

Input:

- triangles
- coordinates (Cartesian; `coord_maps` only for computing derivatives)
- original unit normal vectors and areas

Depends on:

- `compute_partials_cartesian` for computing derivatives `dp_dphi` and `dp_dtheta`.


## Formulas

### Metric Distortions

$$ J_{d} = \frac{1}{4V} \sum_{i=1}^{V} \sum_{n\in{N(i)}} (d_{current}(i, n) - d_{original}(i, n))^2 $$

$$ \frac{\partial J_{d}}{\partial\phi_i} = \frac{1}{V} \sum_{n\in{N(i)}}(d_{current}(i, n) - d_{original}(i, n)) \frac{\partial d_{current}(i, n)}{\partial\phi_i} $$

$$ \frac{\partial J_{d}}{\partial\theta_i} = \frac{1}{V} \sum_{n\in{N(i)}}(d_{current}(i, n) - d_{original}(i, n)) \frac{\partial d_{current}(i, n)}{\partial\theta_i} $$

where $$\frac{\partial d_{current}(i, n)}{\partial\phi_i}$$ and $$\frac{\partial d_{current}(i, n)}{\partial\theta_i}$$ can be calculated using `gds_derivatives`.

### Folding

$$ J_{a} = \frac{1}{2T} \sum_{t=1}^{T} P(A_{current}(t)) (A_{current}(t) - A_{original}(t))^2, P(x) = \cases{1, x \le 0 \cr 0, \text{otherwise}} $$

$$ \frac{\partial J_a}{\partial\phi_i} = \frac{1}{T} \sum_{t\text{ containing }i} P(A_{current}(t)) (A_{current}(t) - A_{original}(t)) \frac{\partial A_{current}(t)}{\partial\phi_i}$$

$$ \frac{\partial J_a}{\partial\theta_i} = \frac{1}{T} \sum_{t\text{ containing }i} P(A_{current}(t)) (A_{current}(t) - A_{original}(t)) \frac{\partial A_{current}(t)}{\partial\theta_i}$$

For the three vertices that form the triangle $$V1 = (x_1, y_1, z_1)$$, $$V2 = (x_2, y_2, z_2)$$ and $$V3 = (x_3, y_3, z_3)$$, let

$$ \vec{a} = (x_2-x_1, y_2-y_1, z_2-z_1)\\
\vec{b} = (x_3-x_1, y_3-y_1, z_3-z_1) $$

We will have:

$$ \begin{eqnarray}
(\frac{\partial A}{\partial x_1}, \frac{\partial A}{\partial y_1}, \frac{\partial A}{\partial z_1}) &=& \vec{n}\times(\vec{b}-\vec{a}) \\
(\frac{\partial A}{\partial x_2}, \frac{\partial A}{\partial y_2}, \frac{\partial A}{\partial z_2}) &=& \vec{n}\times(-\vec{b}) \\
(\frac{\partial A}{\partial x_3}, \frac{\partial A}{\partial y_3}, \frac{\partial A}{\partial z_3}) &=& \vec{n}\times(\vec{a})
\end{eqnarray} $$


## Methods in Fischl et al. (1999)

### Regularization of Metric Distortions

$$ J_{d} = \frac{1}{4V} \sum_{i=1}^{V} \sum_{n\in{N(i)}} (d_{in}^{t} - d_{in}^{0})^2 $$

where:

- V is the number of vertices.
- N(i) is the neighbors of vertex i.
- $$d_{in}^{t} = \| x_{i}^{t} - x_n^t \|$$, it is the distance between the $$i^{th}$$ and $$n^{th}$$ vertices at iteration number t of the numerical optimization procedure.
- $$d_{in}^{0}$$ is the distance on the original cortical surface.
- $$x_{i}^{t}$$ is the $$(x, y, z)$$ position of vertex i at iteration number t.

The $$\frac{1}{4}$$ scaling factor is because each distance between two vertices is calculated twice, and a factor of 2 is introduced by taking the derivative.

### Regularization of Folds in the Surface

The unit normal vector can be calculated by

$$\vec{n_{i}} = \frac{\vec{a_{i}} \times \vec{b_{i}}}{\left|\vec{a_{i}} \times \vec{b_{i}}\right|}$$

The area of the triangle is

$$ \text{Area} = \frac{\left|\vec{a_{i}} \times \vec{b_{i}}\right|}{2} $$

The oriented area is

$$
\begin{eqnarray}
A_{i}^{t} = & \frac{\vec{a_i^t} \times \vec{b_i^t} \cdot \vec{n_i^0}}{2} \\
          = & (\vec{n_i^t} \cdot \vec{n_i^0}) \times \text{Area}^t
\end{eqnarray}
$$

The oriented area is the area projected to the plane orthogonal to $$\vec{n_i^0}$$, and its sign is decided by the directions of $$\vec{n_i^t}$$ and $$\vec{n_i^0}$$.

$$ J_{a} = \frac{1}{2T} \sum_{i=1}^{T} P(A_{i}^{t}) (A_{i}^{t} - A_{i}^{0})^2, P(A_i^t) = \cases{1, A_{i}^{t} \le 0 \cr 0, \text{otherwise}} $$


## References

Fischl, B., Sereno, M. I., & Dale, A. M. (1999). Cortical Surface-Based Analysis: II: Inflation, Flattening, and a Surface-Based Coordinate System. _NeuroImage, 9_(2), 195--207. http://doi.org/10.1006/nimg.1998.0396

Conroy, B., Singer, B., Haxby, J., & Ramadge, P. J. (2009). fMRI-Based Inter-Subject Cortical Alignment Using Functional Connectivity. _Advances in Neural Information Processing Systems, 22_, 378--386.
