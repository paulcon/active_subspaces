---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

(theory-estimators)=
# Estimators

`active_subspace` runs eleven estimators. Each forms a symmetric $m\times m$ matrix $C$ from the
samples, in normalized coordinates, and the dominant eigenvectors of $C$ are the estimated
subspace. They differ in what they need (gradients, or values only), in the population matrix they
estimate, and in the input distributions for which that matrix has its range in the active
subspace. This page states each one. The [selection guide](estimator_selection.md) summarizes how
to choose.

## Common setting

All quantities are in normalized coordinates. The data are points $x_i$, values $f_i$, gradients
$g_i=\nabla f(x_i)$ and weights $\omega_i\ge0$ with $\sum_i\omega_i=1$ (default $1/M$), for
$i=1,\dots,M$. Weighted means are $\bar f_\omega=\sum_i\omega_if_i$ and
$\bar x_\omega=\sum_i\omega_ix_i$. $\Sigma$ is the covariance of the declared reference density:
$\tfrac13I$ for `Uniform`, $I$ for `Gaussian`.

A population matrix is called **ridge-exact** when, for every ridge function $f(x)=h(W_1^\top x)$,
its range lies in $\operatorname{span}(W_1)$. A ridge-exact estimator recovers the ridge directions
as $M\to\infty$; one that is not can be biased even with unlimited data.

**Positive semidefinite and indefinite matrices.** AS, NAS, OLS, QPHD, SIR, SAVE, SS and OPG give
$C\succeq0$. ASX, NASX and pHd give a symmetric $C$ whose eigenvalues can have either sign. For all
estimators, eigenpairs are ranked by $\lvert\lambda\rvert$ and the eigenvalues are reported with
their sign. For pHd this ranking is the published one [Li 1992]; the library uses it for ASX and
NASX as well. A large negative eigenvalue marks an important direction just as a large positive one
does.

## Summary

| Name | Class | $C$ estimates | Needs | Sign | Minimum $M$ | Cost | Reference |
|---|---|---|---|---|---|---|---|
| `as` | `AS()` | $\mathbb E[\nabla f\nabla f^\top]$ | gradients | PSD | 1 | $O(Mm\min(M,m))$ | Constantine (2015) |
| `nas` | `NAS()` | $\mathbb E[\nabla f\nabla f^\top/\lVert\nabla f\rVert^2]$ | gradients | PSD | 1 | $O(Mm\min(M,m))$ | none |
| `asx` | `ASX()` | $\tfrac12\mathbb E[\nabla f\,x^\top+x\,\nabla f^\top]$ | gradients, points | indefinite | 1 | $O(Mm^2+m^3)$ | none |
| `nasx` | `NASX()` | as `asx`, with unit vectors | gradients, points | indefinite | 1 | $O(Mm^2+m^3)$ | none |
| `ols` | `OLS()` | $uu^\top$, $u$ the least-squares slope | values, points | PSD, rank 1 | $m+1$ | $O(Mm^2)$ | classical least squares |
| `qphd` | `QPHD()` | $\mathbb E[\nabla q\nabla q^\top]$, $q$ the least-squares quadratic | values, points | PSD | $(m+1)(m+2)/2$ | $O(Mm^4)$ | Li (1992) |
| `sir` | `SIR(n_slices=10)` | $\mathrm{Cov}(\mathbb E[\tilde x\mid f])$ | values, points | PSD | $2S$ | $O(Mm^2+m^3)$ | Li (1991) |
| `phd` | `PHD()` | $\Sigma^{-1/2}\mathbb E[(f-\mathbb Ef)(x-\mathbb Ex)(x-\mathbb Ex)^\top]\Sigma^{-1/2}$ | values, points | indefinite | 2 | $O(Mm^2+m^3)$ | Li (1992) |
| `save` | `SAVE(n_slices=10)` | $\mathbb E[(I-\mathrm{Cov}(\tilde x\mid f))^2]$ | values, points | PSD | $2S$ | $O(Mm^2+m^3)$ | Cook and Weisberg (1991) |
| `ss` | `SS()` | weighted outer products of difference quotients | values, points | PSD | 2 | $O(M^2m^2)$ | none |
| `opg` | `OPG(p=None)` | $\mathbb E[\nabla f\nabla f^\top]$ via local linear fits | values, points | PSD | $m+1$ distinct points | $O(Mm^3)$ plus the neighbour search | Hristache et al. (2001); Xia et al. (2002) |

The estimator classes live in `active_subspaces.subspaces`; pass a registered name
(`estimator="sir"`) for the defaults or an instance (`estimator=ac.subspaces.SIR(n_slices=5)`) to
set options. "None" in the reference column means that no published reference has been identified:
NAS, ASX, NASX and SS are original to this library and its predecessor, and should be read as
heuristics. The costs exclude the bootstrap, which multiplies them by $B$ and adds $O(m^4)$ per
replicate for the subspace distances.

## Estimators that use gradients

### AS

$$C=\sum_{i=1}^M\omega_i\,g_ig_i^\top,$$

the Monte Carlo (or quadrature) estimate of $\mathbb E_\rho[\nabla f\nabla f^\top]$, computed from
the singular value decomposition of the rows $\sqrt{\omega_i}\,g_i^\top$. When $M>m$ the SVD is taken of
the triangular factor of a blocked QR factorization of those rows, so no other array of the size of
the gradients is formed: without a bootstrap the peak memory is about $1+2/m$ times the gradient
array instead of about three times. With a bootstrap, the full-sample rows and one resampled copy
exist at once, plus index and weight vectors: about $2.57$ times at $m=8$ [D-213]. The eigenvalues
change only by rounding [D-202] (see
[Active subspaces](active_subspaces.md)). It is the matrix that defines the active subspace
[Constantine 2015], and the only one for which the error bound of the ridge approximation holds. It
is ridge-exact for every input distribution.

### NAS

$$C=\sum_{i=1}^M\omega_i\,\hat g_i\hat g_i^\top,\qquad\hat g_i=\begin{cases}g_i/\lVert g_i\rVert,&\lVert g_i\rVert>\sqrt\epsilon\,\max_k\lVert g_k\rVert,\\0,&\text{otherwise},\end{cases}$$

with $\epsilon$ the machine epsilon. Every sample's gradient direction counts equally, whatever its
magnitude, so a few samples with very large gradients do not dominate. For a ridge, every
$\hat g_i\in\operatorname{span}(W_1)$, so NAS is ridge-exact for every input distribution. Zeroed
rows contribute nothing and the weights are not renormalized, so $\operatorname{tr}C\le1$. The
eigenvalues are not mean squared derivatives, and the error bound of the ridge approximation does
not apply to them. No published reference has been identified.

### ASX

$$C=\tfrac12\bigl(A+A^\top\bigr),\qquad A=\sum_{i=1}^M\omega_i\,g_ix_i^\top .$$

For Gaussian inputs, Stein's lemma gives $\mathbb E[\partial_jf\,x_k]=\mathbb E[\partial_j\partial_kf]$
[Stein 1981], so $C$ estimates the average Hessian $\mathbb E[\nabla^2f]$, the pHd matrix computed
from gradients [Li 1992]. It is then ridge-exact. On the uniform cube it is not: for a ridge,
$A=W_1B$ with $B=\mathbb E[\nabla h(W_1^\top x)\,x^\top]$, and $\tfrac12(W_1B+B^\top W_1^\top)$ need
not have its range in $\operatorname{span}(W_1)$. $C$ is indefinite. No published reference has been
identified.

### NASX

ASX with unit gradients $\hat g_i$ (as in NAS) and unit points $\hat x_i=x_i/\lVert x_i\rVert$ (zero
when $\lVert x_i\rVert\le\sqrt\epsilon\max_k\lVert x_k\rVert$). For Gaussian inputs it is ridge-exact,
by the symmetry $z\to-z$ of $x=W_1y+W_2z$, which preserves $\lVert x\rVert$; the cube lacks this
symmetry. The population matrix vanishes unless the sign of the directional derivative correlates
with position: for a monotone one-dimensional ridge $h$ and either of the library's (symmetric)
input distributions it is $0$. $C$ is indefinite. No published
reference has been identified.

## Estimators that use values only

### OLS

Weighted least squares of $f$ on $[1\ \ x^\top]$, with rows scaled by $\sqrt{\omega_i}$:

$$(\hat u_0,\hat u)=\arg\min_{u_0,u}\sum_{i=1}^M\omega_i\,(f_i-u_0-u^\top x_i)^2,\qquad C=\hat u\hat u^\top .$$

Only $\lambda_1=\lVert\hat u\rVert^2$ is nonzero, so only $w_1=\hat u/\lVert\hat u\rVert$ is meaningful;
the other eigenvectors are an arbitrary orthonormal complement. The sign of $w_1$ is that of the
slope, not the largest-magnitude rule of the other estimators, so $f$ increases along $w_1$; this is the
direction published as Algorithm 1.3, eqs. (1.5)–(1.6), of [Constantine 2015] and as Algorithms 1–2,
eq. (22), of [Constantine et al. 2015] [D-195]. The least-squares problem is solved by a QR
factorization when the design has full column rank, and by an SVD-based solver (`gelsd`) otherwise
[D-203]. For Gaussian inputs, Stein's lemma
gives $\hat u\to\mathbb E[\nabla f]$. For a ridge, the population slope lies in $\operatorname{span}(W_1)$ under
the linearity condition, $\mathbb E[x\mid W_1^\top x]$ linear in $W_1^\top x$ [Li 1991], which
Gaussian inputs satisfy and the cube satisfies only in special cases. The population slope is 0
when $f$ is even in $W_1^\top x$, so OLS cannot see a symmetric ridge.

### QPHD

Fit a quadratic $q(x)=c+g^\top x+\tfrac12x^\top Hx$ by weighted least squares on the monomials
$1$, $x_j$, $x_j^2$ and $x_jx_k$ ($j<k$), with $H_{jj}$ twice the coefficient of $x_j^2$ and
$H_{jk}=H_{kj}$ the coefficient of $x_jx_k$. Then

$$C=\mathbb E_\rho\bigl[\nabla q\,\nabla q^\top\bigr]=gg^\top+H\Sigma H,$$

which uses $\mathbb E_\rho[x]=0$ and $\mathbb E_\rho[xx^\top]=\Sigma$. QPHD follows from Corollary 3.1
of Li (1992) with $q=2$. It differs from pHd by averaging the outer product of the
gradient of the fitted quadratic, where pHd uses its Hessian alone. For Gaussian inputs, the
least-squares quadratic has $g=\mathbb E[\nabla f]$ and $H=\mathbb E[\nabla^2f]$ (orthogonality of
the Hermite polynomials), and $C$ is ridge-exact. The basis has $(m+1)(m+2)/2$ columns, so the cost is
$O(Mm^4)$ and the method is practical only for $m$ up to about 50.

:::{admonition} Departure from the 2016 code
:class: important

The 2016 code used $C=gg^\top+\gamma HH^\top$, with the scalar $\gamma$ guessed from the range of the
data. v1 uses $C=gg^\top+H\Sigma H$ with $\Sigma$ from the declared input space. Both v1 spaces
have $\Sigma=\gamma I$ ($\gamma=\tfrac13$ or 1), so the two forms agree whenever the guess was right.
Reason: $gg^\top+H\Sigma H$ is exactly $\mathbb E_\rho[\nabla q\nabla q^\top]$, and the input
distribution is declared rather than inferred.
:::

### SIR

Sliced inverse regression [Li 1991]. Whiten the inputs with the declared covariance, partition the
samples into slices $h=1,\dots,S'$ by the value of $f$, and average the slice means:

$$\tilde x_i=\Sigma^{-1/2}(x_i-\bar x_\omega),\qquad\hat p_h=\sum_{i\in h}\omega_i,\qquad\hat m_h=\frac1{\hat p_h}\sum_{i\in h}\omega_i\tilde x_i,\qquad C=\sum_h\hat p_h\,\hat m_h\hat m_h^\top .$$

**Slicing.** Sort the samples by $f$ (stably, so ties keep their row order) and cut the sorted list
into $S$ (`n_slices`, default 10) slices of nearly equal count. A boundary that splits a run of
equal values is moved to the end of the run, so ties stay in one slice; empty slices are dropped;
a slice with one sample is merged into its neighbour. SIR and SAVE use the same slices. A slice with
zero total weight contributes nothing.

Under the linearity condition, $\operatorname{range}(C)\subseteq\operatorname{span}(W_1)$ for a
ridge [Li 1991; Glaws, Constantine and Cook 2020]. SIR finds only directions along which the
conditional mean $\mathbb E[x\mid f]$ moves; for a ridge that is even in $W_1^\top x$ that mean is
constant and SIR sees nothing. On the uniform cube, for an exact one-dimensional ridge, the second
eigenvalue does not vanish: $\lambda_2/\lambda_1=0.056$ on the cube against $1.6\times10^{-5}$ for
Gaussian inputs in the case measured.

### pHd

Principal Hessian directions [Li 1992]:

$$\Sigma_{fxx}=\sum_{i=1}^M\omega_i\,(f_i-\bar f_\omega)(x_i-\bar x_\omega)(x_i-\bar x_\omega)^\top,\qquad C=\Sigma^{-1/2}\,\Sigma_{fxx}\,\Sigma^{-1/2}.$$

For Gaussian inputs ($\Sigma=I$), Stein's lemma applied twice gives
$\mathbb E[(f-\mathbb Ef)\,xx^\top]=\mathbb E[\nabla^2f]$ [Stein 1981; Li 1992], so $C$ estimates the
average Hessian and is ridge-exact. $C$ is indefinite and is ranked by $\lvert\lambda\rvert$ [Li 1992].
pHd misses directions along which the average curvature vanishes, for example a ridge whose profile
is odd. On the uniform cube Stein's lemma does not hold.

### SAVE

Sliced average variance estimation [Cook and Weisberg 1991]. With the whitened $\tilde x_i$, the
slices and slice means of SIR, and normalized slice weights $s_i=\omega_i/\hat p_h$ for $i\in h$:

$$\widehat{\mathrm{Cov}}_h=\frac{\sum_{i\in h}s_i(\tilde x_i-\hat m_h)(\tilde x_i-\hat m_h)^\top}{1-\sum_{i\in h}s_i^2},\qquad V_h=I-\widehat{\mathrm{Cov}}_h,\qquad C=\sum_h\hat p_h\,V_h^2 .$$

The denominator is the unbiased correction for weighted samples; with equal weights it is the
familiar $n_h/(n_h-1)$. The factor $\hat p_h$ follows Cook and Weisberg; the 2016 code omitted it,
so v1 eigenvalues are about $S$ times smaller than the 2016 ones. SAVE needs the linearity and
constant-variance conditions [Cook and Weisberg 1991], which Gaussian inputs satisfy. Unlike SIR it
sees even ridges, through the slice variances.

:::{admonition} Departure from the published method: whitening, slicing and weights
:class: important

**Whitening (SIR, pHd, SAVE).** The published methods whiten with the sample mean and the sample
covariance, $\tilde x_i=\hat\Sigma^{-1/2}(x_i-\bar x)$ [Li 1991; Li 1992; Cook and Weisberg 1991].
v1 uses the declared covariance, $\tilde x_i=\Sigma^{-1/2}(x_i-\bar x_\omega)$, with
$\Sigma=\tfrac13I$ (uniform) or $I$ (Gaussian), and the weighted mean. Reason: the input
distribution is declared, so its covariance is known exactly and need not be estimated. On
`Uniform` and `Gaussian`, $\Sigma$ is a multiple of $I$, so the eigenvectors of $C$ span the same
subspace as Li's back-transformed directions $\Sigma^{-1/2}\eta$, and are used directly.

On a `Product` with uniform and Gaussian coordinates, $\Sigma$ is diagonal with $\tfrac13$ and 1,
and the directions are back-transformed [D-205]. With $H=[\eta_1\cdots\eta_m]$ the eigenvectors of
the whitened $C$, v1 takes the QR factorization $\Sigma^{-1/2}H=QR$ and sets $W=Q$, so
$\operatorname{span}(w_1,\dots,w_k)=\Sigma^{-1/2}\operatorname{span}(\eta_1,\dots,\eta_k)$ for
every $k$: every partition, bootstrap distance and ladle determinant refers to Li's directions.
The eigenvalues are those of the whitened $C$, and **the columns of $W$ are not its eigenvectors**.
Each $w_j$ is the part of $\Sigma^{-1/2}\eta_j$ orthogonal to $w_1,\dots,w_{j-1}$, not Li's
direction itself, and the trailing columns span the Euclidean complement
$\Sigma^{1/2}\operatorname{span}(\eta_{k+1},\dots,\eta_m)$. Activity scores therefore use the
normalized directions $b_j=\Sigma^{-1/2}\eta_j/\lVert\Sigma^{-1/2}\eta_j\rVert$ [D-208].

**Slices and weights (SIR, SAVE).** The published methods use unweighted slice proportions
$n_h/M$ and slices of about equal count. v1 uses the weighted proportions
$\hat p_h=\sum_{i\in h}\omega_i$ and weighted slice means, keeps ties in one slice and merges
singleton slices, with $S$ a parameter. Reason: quadrature weights then enter consistently, and the
slices are well defined when $f$ has ties. With equal weights and no ties the slices are the usual
ones.

**Centering (pHd).** v1 centers $f$ and $x$ with the weighted means $\bar f_\omega$ and
$\bar x_\omega$, which equal the sample means under equal weights.
:::

### SS, the swarm subspace

$$C=\sum_{i=1}^M\sum_{\substack{j=1\\x_j\ne x_i}}^M\omega_i\omega_j\,(f_i-f_j)^2\,u_{ij}u_{ij}^\top,\qquad u_{ij}=\frac{x_j-x_i}{\lVert x_j-x_i\rVert},$$

the weighted outer products of the difference quotients $(f_j-f_i)/\lVert x_j-x_i\rVert$ along the
unit vector from $x_i$ to $x_j$, each multiplied by $\lVert x_j-x_i\rVert^2$. Only exact duplicate
points are excluded; every other term is bounded by $\omega_i\omega_j(f_i-f_j)^2$, so no distance
threshold is needed. SS is a heuristic with no published reference. It is biased on the uniform
cube, by about 5° to 9° in the one-dimensional ridges measured with $m=5$, and the bias does not
shrink as $M$ grows. No analysis establishes what its population matrix recovers for Gaussian
inputs either, so it warns for both spaces. It costs $O(M^2m^2)$ time.

### OPG, outer product of local linear gradients

Estimate the gradient at **every** sample by a local linear fit (see
[Active subspaces](active_subspaces.md)) and average the outer products with the sample weights:

$$C=\sum_{i=1}^M\omega_i\,\hat g_i\hat g_i^\top,\qquad\hat g_i=\text{the local linear gradient at }x_i,$$

computed as for AS [Hristache et al. 2001; Xia et al. 2002]. It estimates the same matrix as AS
and needs no distributional assumption, so it is valid for both spaces. The local fits are biased
by the curvature of $f$ over each neighbourhood; the bias falls as the sampling gets denser. In the
library's tests on a Gaussian ridge with $m=10$, the subspace error fell at about $M^{-0.75}$ over
$M=500$ to $4000$.

:::{admonition} Departure from the published method: neighbourhoods
:class: important

The published OPG weights the local fit at $x_i$ with a fixed-bandwidth kernel,

$$\min_{a_0,a}\sum_{j=1}^MK_h(x_j-x_i)\bigl(f_j-a_0-a^\top(x_j-x_i)\bigr)^2$$

[Xia et al. 2002]. v1, like the 2016 code, uses the uniform kernel on the $p$ nearest distinct
sample locations, $K(x_j-x_i)=1$ for $x_j\in\mathcal N_i$ and 0 otherwise, an adaptive bandwidth
with default $p=\min\bigl(\max(m+1,\lfloor1.7m\rfloor),M_d\bigr)$. Reason: there is no bandwidth to
select, and every local system has at least $m+1$ points wherever the samples lie.
:::

Inside a bootstrap replicate, about 37% of the rows are duplicates. The neighbourhoods count
distinct locations, and every row at a selected location enters the fit, so duplicates act as
multiplicity weights.

## Validity on uniform and Gaussian inputs

An estimator applied outside its assumptions issues a `ValidityWarning` (code
`validity.assumption`) and still runs.

| Estimator | Uniform | Gaussian | Basis |
|---|---|---|---|
| `as` | ok | ok | defines the active subspace |
| `nas` | ok | ok | ridge-exact for every distribution |
| `asx` | warns | ok | Gaussian: $C=\mathbb E[\nabla^2f]$ (Stein's lemma) |
| `nasx` | warns | ok | Gaussian: ridge-exact by symmetry |
| `ols` | warns | ok | linearity condition; Gaussian: $\hat u\to\mathbb E[\nabla f]$ |
| `qphd` | warns | ok | Gaussian: $g=\mathbb E[\nabla f]$, $H=\mathbb E[\nabla^2f]$ |
| `sir` | warns | ok | linearity condition |
| `phd` | warns | ok | Stein's lemma needs Gaussian inputs |
| `save` | warns | ok | linearity and constant-variance conditions |
| `ss` | warns | warns | heuristic with no reference; biased on the cube |
| `opg` | ok | ok | local fits; no distributional assumption |

The biases on the uniform cube are not sampling error: they persist as $M$ grows. The library's
tests measure them on exact one-dimensional ridges $f=h(a^\top x)$ with $m=5$ and
$a\propto(1,2,3,4,5)$: about 3° for QPHD with $h(y)=y^4$, about 4° for SAVE and 5° for pHd with
$h(y)=y^2$, and about 5° for SS with $h=\exp$. The cell below reproduces the first three. On the cube
the angle settles at a few degrees; with Gaussian inputs it keeps falling as $M$ grows.

```{code-cell} python
import warnings

import numpy as np
import active_subspaces as ac

m = 5
a = np.arange(1.0, m + 1) / np.sqrt(55.0)   # unit ridge direction
profiles = {"phd": lambda y: y**2, "save": lambda y: y**2, "qphd": lambda y: y**4}

print(f"{'estimator':>9} {'M':>7} {'uniform':>9} {'Gaussian':>9}   (angle to a, degrees)")
for name, h in profiles.items():
    for M in (2_000, 20_000, 200_000):
        angles = []
        for space in (ac.Uniform.reference(m), ac.Gaussian.reference(m)):
            X = space.sample(M, rng=7)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ac.ValidityWarning)   # expected on the cube
                res = ac.active_subspace(space=space, X=X, f=h(X @ a), estimator=name, n=1)
            angles.append(np.degrees(np.arccos(min(1.0, abs(res.W1[:, 0] @ a)))))
        print(f"{name:>9} {M:>7} {angles[0]:9.2f} {angles[1]:9.2f}")
```

The dimension rules have their own validity table; see [Choosing the dimension](dimension.md).

## Departures at a glance

| Item | Published | v1 |
|---|---|---|
| SIR, pHd, SAVE whitening | sample mean and covariance | declared $\Sigma$, weighted mean |
| SIR, SAVE slices | unweighted proportions, equal-count slices | weighted proportions and means; ties kept together; no singleton slices; $S$ a parameter |
| pHd centering | sample means | weighted means (equal under equal weights) |
| QPHD | Li (1992), Corollary 3.1 with $q=2$; 2016 code: $gg^\top+\gamma HH^\top$ with $\gamma$ guessed | $gg^\top+H\Sigma H$, $\Sigma$ declared |
| OPG kernel | fixed bandwidth | $p$ nearest neighbours, uniform weights, at every sample |
| Ranking of indefinite matrices | pHd: by $\lvert\lambda\rvert$ | by $\lvert\lambda\rvert$ for pHd, ASX and NASX, eigenvalues reported signed |
| OLS eigenvector sign | $w_1=\hat u/\lVert\hat u\rVert$ | the same, signed |
| NAS, ASX, NASX, SS | no reference | heuristics, documented as such |

## References

- Constantine, P. G. (2015). *Active Subspaces: Emerging Ideas for Dimension Reduction in Parameter
  Studies.* SIAM.
- Constantine, P. G., Emory, M., Larsson, J. and Iaccarino, G. (2015). Exploiting active subspaces
  to quantify uncertainty in the numerical simulation of the HyShot II scramjet. *Journal of
  Computational Physics* 302:1–20.
- Cook, R. D. and Weisberg, S. (1991). Discussion of "Sliced inverse regression for dimension
  reduction". *Journal of the American Statistical Association* 86(414):328–332.
- Glaws, A., Constantine, P. G. and Cook, R. D. (2020). Inverse regression for ridge recovery: a
  data-driven approach for parameter reduction in computer experiments. *Statistics and Computing*
  30:237–253. (details to verify)
- Hristache, M., Juditsky, A., Polzehl, J. and Spokoiny, V. (2001). Structure adaptive approach for
  dimension reduction. *Annals of Statistics* 29(6):1537–1566.
- Li, K.-C. (1991). Sliced inverse regression for dimension reduction. *Journal of the American
  Statistical Association* 86(414):316–327.
- Li, K.-C. (1992). On principal Hessian directions for data visualization and dimension reduction:
  another application of Stein's lemma. *Journal of the American Statistical Association*
  87(420):1025–1039.
- Stein, C. M. (1981). Estimation of the mean of a multivariate normal distribution. *Annals of
  Statistics* 9(6):1135–1151.
- Xia, Y., Tong, H., Li, W. K. and Zhu, L.-X. (2002). An adaptive estimation of dimension reduction
  space. *Journal of the Royal Statistical Society, Series B* 64(3):363–410.
