---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

(theory-active-subspaces)=
# Active subspaces

## Setting

Let $f$ be a scalar function of $m$ inputs. Its physical inputs $\xi$ have a density that you
declare with an input space. The library maps them to normalized inputs $x=A^{-1}(\xi-b)$ with a
fixed reference density $\rho$:

| Input space | Physical density | $A$ | $b$ | $\rho$ | $\Sigma=\mathrm{Cov}_\rho(x)$ |
|---|---|---|---|---|---|
| `Uniform(lower, upper)` | uniform on $\prod_j[\ell_j,u_j]$ | $\mathrm{diag}\bigl(\tfrac{u-\ell}2\bigr)$ | $\tfrac{u+\ell}2$ | uniform on $[-1,1]^m$ | $\tfrac13I$ |
| `Gaussian(mean, cov)` | $N(\mu,\Gamma)$ | $L$, with $\Gamma=LL^\top$ (Cholesky) | $\mu$ | $N(0,I_m)$ | $I$ |

From here on $f$ denotes the function of $x$, $f(x)=f_{\text{phys}}(Ax+b)$, and $\nabla f$ its
gradient with respect to $x$. By the chain rule, $\nabla_xf=A^\top\nabla_\xi f$. You pass physical
gradients $\nabla_\xi f$; the library applies this map.

## Definition

The active subspace is defined by the average outer product of the gradient
[Constantine, Dow and Wang 2014; Constantine 2015]:

$$C=\int\nabla f(x)\,\nabla f(x)^\top\rho(x)\,dx=\mathbb E_\rho\bigl[\nabla f\,\nabla f^\top\bigr].$$

$C$ is symmetric and positive semidefinite, so it has a real eigendecomposition

$$C=W\Lambda W^\top,\qquad\Lambda=\mathrm{diag}(\lambda_1,\dots,\lambda_m),\qquad\lambda_1\ge\dots\ge\lambda_m\ge0,$$

with $W$ orthogonal. Each eigenvalue is the mean squared directional derivative along its
eigenvector:

$$\lambda_k=w_k^\top Cw_k=\mathbb E_\rho\bigl[(w_k^\top\nabla f)^2\bigr],\qquad\sum_{k=1}^m\lambda_k=\mathbb E_\rho\bigl[\lVert\nabla f\rVert^2\bigr].$$

If $\lambda_k=0$, then $w_k^\top\nabla f=0$ almost everywhere, and $f$ does not change along $w_k$.
Small eigenvalues mean that $f$ changes little, on average, along their eigenvectors.

Split $W=[W_1\ W_2]$ after the $n$-th column. The **active subspace** is $\operatorname{range}(W_1)$;
the **active variables** are $y=W_1^\top x\in\mathbb R^n$ and the **inactive variables**
$z=W_2^\top x\in\mathbb R^{m-n}$. Because $W$ is orthogonal, $x=W_1y+W_2z$.

## The ridge approximation and its error

For fixed $W_1$, the best approximation of $f$ by a function of $y$ alone, in the mean-squared
sense under $\rho$, is the conditional mean

$$G(y)=\mathbb E_\rho[f(x)\mid W_1^\top x=y]=\int f(W_1y+W_2z)\,\pi(z\mid y)\,dz,$$

where $\pi(z\mid y)$ is the conditional density of the inactive variables. Its error is bounded by
the eigenvalues that were left out [Constantine, Dow and Wang 2014; Constantine 2015]:

$$\mathbb E_\rho\Bigl[\bigl(f(x)-G(W_1^\top x)\bigr)^2\Bigr]\le C_P\,(\lambda_{n+1}+\dots+\lambda_m),$$

with $C_P$ a Poincaré constant associated with $\rho$, which does not depend on $f$. If the
subspace is estimated, with $\hat W_1$ in place of $W_1$ and
$\varepsilon=\lVert W_1W_1^\top-\hat W_1\hat W_1^\top\rVert$ the distance between the two subspaces,
the bound becomes

$$\mathbb E_\rho\Bigl[\bigl(f(x)-\hat G(\hat W_1^\top x)\bigr)^2\Bigr]\le C_P\Bigl(\varepsilon\,(\lambda_1+\dots+\lambda_n)^{1/2}+(\lambda_{n+1}+\dots+\lambda_m)^{1/2}\Bigr)^2,$$

with $\hat G$ the conditional mean for $\hat W_1$. The trade-off in this bound is the basis of the
response-surface dimension rule (see [Choosing the dimension](dimension.md)). Both bounds hold for
the matrix $C$ defined above, and only for it; the other estimators on the
[next page](estimators.md) estimate different matrices.

The estimated subspace is well determined when there is a gap between $\lambda_n$ and
$\lambda_{n+1}$: a large gap makes $\hat W_1$ insensitive to sampling error, and a small one makes
it poorly determined [Constantine and Gleich 2014; Constantine 2015].

## Order and sign conventions

Eigenpairs are ordered by $\lvert\lambda\rvert$, largest first, and eigenvalues are reported with
their sign. For $C$ above this is the usual descending order; the convention matters for the
indefinite estimators of the next page. Equal magnitudes keep the order of the eigensolver, so the
result is deterministic.

An eigenvector is determined only up to sign. The library fixes it so that the
largest-magnitude component of every eigenvector is positive (on an exact tie in magnitude, the
first such component). This rule does not flip when the first component is near zero, and it is
equivariant under a permutation of the inputs.

(theory-physical-directions)=
## Normalized coordinates and physical directions

The eigenvalues are defined in normalized coordinates. The matrix of physical gradients,
$C_\xi=\mathbb E[\nabla_\xi f\,\nabla_\xi f^\top]$, is related to $C$ by $C=A^\top C_\xi A$. Its
entries carry the units of the inputs, so its eigenvalues mix quantities that cannot be compared,
and changing the units of one input changes them. In normalized coordinates every input ranges over
$[-1,1]$ (uniform) or has unit variance (Gaussian), and $C$ is scale-free.

**Physical view.** Suppose $f$ is a ridge in physical coordinates, $f_{\text{phys}}(\xi)=h(a^\top\xi)$.
In normalized coordinates $f(x)=h\bigl((A^\top a)^\top x+a^\top b\bigr)$, so the normalized active
direction is $w\propto A^\top a$, and conversely $a\propto A^{-\top}w$. The physical active
directions are therefore an orthonormal basis of

$$\operatorname{span}\bigl(A^{-\top}W_1\bigr),$$

computed by a thin QR factorization followed by the sign rule above. This is what
`result.physical_directions()` returns. It is the subspace of covectors $a$ such that $f$ varies
along $a^\top\xi$. The eigenvalues have no physical counterpart.

$AW_1$ is a different object: the displacement of $\xi$ caused by moving along $W_1$ in $x$. In
general its span is not the physical active subspace; the two agree for every $W_1$ only when $A$ is
a multiple of an orthogonal matrix.

Physical directions carry units, so their components are not comparable across inputs. Use the
normalized $W$ to judge which inputs matter.

**Gaussian inputs with correlation.** Any square root $L'=LQ$ of $\Gamma$, with $Q$ orthogonal,
gives $C'=Q^\top CQ$: the same eigenvalues, eigenvectors rotated by $Q^\top$, and the same physical
subspace $\operatorname{span}(L^{-\top}W_1)$. The library uses the Cholesky factor, so for a
non-diagonal covariance the normalized coordinates depend on the order of the inputs; the physical
directions do not.

The example below uses a ridge $f(\xi)=\sin(a^\top\xi)$ whose inputs have ranges from 1 to 1000. In
physical coordinates the direction $a$ is dominated by its first component. In normalized
coordinates all four inputs matter equally, because each component of $a$ is offset by the width of
its range.

```{code-cell} python
import numpy as np
import active_subspaces as ac

space = ac.Uniform(lower=[0.0, 0.0, 0.0, 0.0], upper=[1.0, 10.0, 100.0, 1000.0])
a = np.array([1.0, 0.1, 0.01, 0.001])          # physical ridge direction
X = space.sample(200, rng=0)                    # physical samples
grads = np.cos(X @ a)[:, None] * a[None, :]     # physical gradients of sin(a^T xi)

res = ac.active_subspace(space=space, grads=grads)
np.set_printoptions(precision=4, suppress=True)
print("n =", res.n)
print("eigenvalues:", res.eigenvalues)
print("normalized w1:", res.W1[:, 0])
print("physical direction:", res.physical_directions()[:, 0])
print("a / |a|:", a / np.linalg.norm(a))
```

## Estimating $C$ by Monte Carlo

With samples $x_i\sim\rho$ and weights $\omega_i$ (by default $1/M$), the estimate is

$$\hat C=\sum_{i=1}^M\omega_i\,\nabla f(x_i)\,\nabla f(x_i)^\top .$$

The weights can also be those of a quadrature rule on $\rho$; they must be nonnegative and sum
to 1. The library does not form $\hat C$. It computes the thin singular value decomposition of the
$M\times m$ matrix with rows $\sqrt{\omega_i}\,\nabla f(x_i)^\top$, $U\,\mathrm{diag}(\sigma)\,V^\top$,
and sets $\hat\lambda_k=\sigma_k^2$ and $\hat W=V$. This is the same $\hat C$, with small
eigenvalues computed to higher relative accuracy. When $M<m$, the remaining $m-M$ eigenvalues are 0
and their eigenvectors complete $V$ to an orthogonal matrix.

**How many samples.** A common heuristic takes

$$M=\alpha\,k\ln m,$$

with an oversampling factor $\alpha$ between 2 and 10 and $k$ the number of eigenvalues to be
estimated [Constantine 2015; Constantine and Gleich 2014]. The reduced model uses $\alpha=6$ and
$k=m+1$ when it samples a model itself (see [Reduced model](reduced_model.md)). This is a rule of
thumb, not a guarantee; the bootstrap below shows whether the estimate is stable.

## The bootstrap

The sampling variability of $\hat\lambda$ and $\hat W$ is assessed by the nonparametric bootstrap
[Constantine 2015]. For $b=1,\dots,B$ (`nboot`):

1. Draw $M$ row indices independently and uniformly from $\lbrace1,\dots,M\rbrace$.
2. Resample every given array ($X$, $f$, gradients) and the weights, and renormalize the weights to
   sum to 1.
3. Recompute $(\hat\lambda^{(b)},\hat W^{(b)})$ with the same estimator, order and sign rule.
4. Record, for $k=1,\dots,m-1$, the subspace distance
   $\delta_k^{(b)}=\lVert\hat W_{1:k}^\top\hat W^{(b)}_{k+1:m}\rVert$, the sine of the largest
   principal angle between the first $k$ eigenvectors of the full sample and of replicate $b$; and
   $d_k^{(b)}=\lvert\det(\hat W_{1:k}^\top\hat W^{(b)}_{1:k})\rvert$, the product of the cosines of
   the principal angles, which the ladle rule uses.

The replicates are stored in the result (`bootstrap_eigenvalues`, `bootstrap_distances`,
`bootstrap_dets`), so intervals at any level can be formed afterwards without recomputing; see
[Choosing the dimension](dimension.md). Replicate eigenvectors are not stored.

Two limits apply. The eigenvalue bootstrap is not consistent at repeated or zero eigenvalues
[Beran and Srivastava 1985; Eaton and Tyler 1991], so intervals for eigenvalues near a cluster or
near zero should be read as indications only. And the bootstrap treats the rows as independent
draws from $\rho$; with non-uniform quadrature weights that assumption does not hold, and the
library warns.

Each replicate draws from its own random stream, so the result is reproducible from the seed and
does not depend on how many workers run the replicates.

## Gradients from values

When the model gives values only, gradients can be approximated first and passed to the estimator.

**Finite differences.** Steps are taken along the normalized axes, so the result is $\nabla_xf$
directly. For sample $i$ and input $j$ the step is $h_{ij}=h\max(1,\lvert x_{ij}\rvert)$ with $h$
in normalized units (default $10^{-6}$):

$$\text{forward:}\ \frac{f(x_i+h_{ij}e_j)-f(x_i)}{h_{ij}},\qquad\text{central:}\ \frac{f(x_i+h_{ij}e_j)-f(x_i-h_{ij}e_j)}{2h_{ij}} .$$

The model is never evaluated outside the support: on a uniform space a step that would leave
$[-1,1]$ is taken backward instead (forward scheme), or the entry falls back to the one-sided
formula (central scheme). Forward differences cost $(m+1)M$ evaluations, central at most
$(2m+1)M$. The truncation error is $O(h)$ or $O(h^2)$ and the rounding error $O(\varepsilon_f/h)$,
with $\varepsilon_f$ the noise in $f$; you choose $h$ to balance them. The default suits a model
computed to near machine precision.

**Local linear gradients.** At each sample $x_i$, fit a linear function by ordinary least squares
to the $p$ nearest distinct sample locations (including $x_i$, distances in normalized
coordinates):

$$(\hat a_0,\hat a)=\arg\min_{a_0,a}\sum_{j\in\mathcal N_i}\bigl(f_j-a_0-a^\top(x_j-x_i)\bigr)^2,\qquad\widehat{\nabla f}(x_i)=\hat a .$$

The default is $p=\min\bigl(\max(m+1,\lfloor1.7m\rfloor),M_d\bigr)$, with $M_d$ the number of
distinct locations, and $M_d\ge m+1$ is required. The estimate is exact for linear $f$; for
nonlinear $f$ it is biased by the curvature over the neighbourhood, a bias that falls as the
sampling becomes denser. The averaged outer product of these gradients is the OPG estimator,
described with its departure from the published method on the [next page](estimators.md).

## References

- Beran, R. and Srivastava, M. S. (1985). Bootstrap tests and confidence regions for functions of a
  covariance matrix. *Annals of Statistics* 13(1):95–115. (details to verify)
- Constantine, P. G. (2015). *Active Subspaces: Emerging Ideas for Dimension Reduction in Parameter
  Studies.* SIAM.
- Constantine, P. G., Dow, E. and Wang, Q. (2014). Active subspace methods in theory and practice:
  applications to kriging surfaces. *SIAM Journal on Scientific Computing* 36(4):A1500–A1524.
- Constantine, P. G. and Gleich, D. F. (2014). Computing active subspaces with Monte Carlo.
  arXiv:1408.0545. (details to verify)
- Eaton, M. L. and Tyler, D. E. (1991). On Wielandt's inequality and its application to the
  asymptotic distribution of the eigenvalues of a random symmetric matrix. *Annals of Statistics*
  19(1):260–271. (details to verify)
