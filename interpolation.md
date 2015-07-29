---
title: Interpolation
layout: default
---

## Spherical Radial Basis Function

$$ \Psi_i(p) = \psi(d(p,p_i)) = (1-\frac{2}{r}\sin(\frac{d(p,p_i)}{2}))^4 \times (\frac{8}{r}\sin(\frac{d(p,p_i)}{2})+1)$$

In the paper the $$\times$$ sign was a plus sign, which is probably a typo (otherwise some parentheses would be redundant).

I've changed $$\Phi_i(p)$$ and $$\phi(d(p,p_i))$$ in the paper into $$\Psi_i(p)$$ and $$\psi(d(p,p_i))$$, to avoid confusion with $$\phi$$ in spherical coordinates.

Let $$S = \frac{2}{r}\sin(\frac{d(p,p_i)}{2})$$, we have

$$ \Psi_i(p) = (1-S)^4 \times(4S+1) $$

Therefore,

$$ \begin{eqnarray}
\frac{\partial\Psi}{\partial d} &=& \frac{d\Psi}{dS} \times \frac{\partial S}{\partial d} \\
&=& -20S(1-S)^3 \times \frac{2}{r}\cos(\frac{d}{2}) \\
&=& -20(1-S)^3 \frac{\sin d}{r^2}
\end{eqnarray} $$

$$ \begin{eqnarray}
\frac{\partial\Psi}{\partial\phi} &=& -\frac{20(1-S)^3}{r^2} \times (\sin\phi_1\cos\phi_2 - \cos\phi_1\sin\phi_2\cos(\theta_1 - \theta_2)) \\
\frac{\partial\Psi}{\partial\theta} &=& -\frac{20(1-S)^3}{r^2} \times (\sin\phi_1\sin\phi_2\sin(\theta_1 - \theta_2))
\end{eqnarray} $$
