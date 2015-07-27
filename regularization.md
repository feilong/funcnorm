---
title: Regularization
layout: default
---

## Regularization of Metric Distortions

$$ J_{d} = \frac{1}{4V} \sum_{i=1}^{V} \sum_{n\in{N(i)}} (d_{in}^{t} - d_{in}^{0})^2 $$

where:

- V is the number of vertices.
- N(i) is the neighbors of vertex i.
- $$d_{in}^{t} = \| x_{i}^{t} - x_n^t \|$$, it is the distance between the $$i^{th}$$ and $$n^{th}$$ vertices at iteration number t of the numerical optimization procedure.
- $$d_{in}^{0}$$ is the distance on the original cortical surface.
- $$x_{i}^{t}$$ is the $$(x, y, z)$$ position of vertex i at iteration number t.

The $$\frac{1}{4}$$ scaling factor is because each distance between two vertices is calculated twice, and a factor of 2 is introduced by taking the derivative.



## Reference

Fischl, B., Sereno, M. I., & Dale, A. M. (1999). Cortical Surface-Based Analysis: II: Inflation, Flattening, and a Surface-Based Coordinate System. _NeuroImage, 9_(2), 195–207. http://doi.org/10.1006/nimg.1998.0396
