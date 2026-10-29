(theory-estimator-selection)=
# Choosing an estimator

This guide condenses the [Estimators](estimators.md) and [Choosing the dimension](dimension.md)
pages into a decision procedure. It answers four questions in order: can you get gradients, what is
your input distribution, how many samples can you afford relative to $m$, and do you need the
dimension chosen for you. It makes no claim beyond what those pages and the library's tests
support; where the evidence is a single measured case, it says so.

## 1. Can you get gradients?

**Yes: analytic, adjoint or automatic-differentiation gradients.** Use AS (the default). It
estimates the matrix that defines the active subspace, it is valid for every input distribution, it
is the only estimator to which the error bound of the ridge approximation applies, and the RS
dimension rule is designed for it. A common sample size is $M=\alpha k\ln m$ with $\alpha$ between 2
and 10 and $k$ the number of eigenvalues you want [Constantine 2015]; run the bootstrap to check that
the estimate is stable.

- **NAS** gives every sample's gradient direction equal weight. It is ridge-exact for every
  distribution, so it is an option when a few samples with very large gradients would otherwise
  dominate. It is a heuristic with no published reference, its eigenvalues are not mean squared
  derivatives, and RS does not apply to it.
- **ASX and NASX** are for Gaussian inputs only (ASX then estimates the average Hessian). They are
  indefinite, have no published reference, and warn on uniform inputs.

**No, but you can run the model.** Finite differences give gradients at $(m+1)M$ evaluations
(forward) or up to $(2m+1)M$ (central), after which AS applies with all its properties, up to the
finite-difference error. Choose the step $h$ from the noise level of the model. When $(m+1)M$
evaluations are too many, sample values only and continue below.

**No: you have a fixed data set of values.** Go to question 2.

## 2. What is the input distribution?

**Either distribution: OPG.** OPG estimates the same matrix as AS from local linear gradients at
every sample, with no distributional assumption, and it is the only value-based estimator for which
RS does not warn. Each local fit uses about $1.7m$ neighbours, and the fits are biased by the
curvature of $f$ over each neighbourhood, a bias that falls as the samples become denser. In the
library's tests on a Gaussian ridge with $m=10$, the subspace error fell at about $M^{-0.75}$ for
$M$ from 500 to 4000. OPG uses nearest-neighbour neighbourhoods rather than the fixed-bandwidth
kernel of the published method.

**Gaussian inputs: the regression estimators are consistent.** OLS, QPHD, SIR, pHd and SAVE all
recover a ridge as $M$ grows. They see different features of $f$:

| Estimator | Finds | Misses |
|---|---|---|
| OLS | one direction, the average gradient $\mathbb E[\nabla f]$ | ridges that are even in $W_1^\top x$ |
| SIR | directions along which $\mathbb E[x\mid f]$ moves | ridges that are even in $W_1^\top x$ |
| pHd | directions of nonzero average curvature $\mathbb E[\nabla^2f]$ | ridges whose average curvature vanishes, e.g. odd profiles |
| QPHD | gradient and curvature of the best quadratic, $gg^\top+H\Sigma H$ | — but needs $(m+1)(m+2)/2$ samples |
| SAVE | changes in the conditional mean and variance | — |

**Uniform inputs: prefer gradients or OPG.** SIR, SAVE, pHd, QPHD and OLS rest on conditions that
the uniform cube does not satisfy in general, and they warn. Their errors on the cube include a bias
that does not shrink with $M$. Measured on exact one-dimensional ridges with $m=5$: about 3° for
QPHD on $(a^\top x)^4$, about 4° for SAVE and 5° for pHd on $(a^\top x)^2$; SIR leaves a second
eigenvalue of $0.056\lambda_1$ where Gaussian inputs give $1.6\times10^{-5}\lambda_1$. They are still
useful for exploration and as a cross-check, but a direction they find carries this bias.

**SS** is a heuristic with no published reference, biased on the cube by about 5° to 9° in the
cases measured, with no supporting analysis for Gaussian inputs either. It warns for both spaces.
Use it only for exploration.

## 3. How many samples, relative to $m$?

| Estimator | Hard minimum $M$ | Cost | Notes |
|---|---|---|---|
| AS, NAS | 1 | $O(Mm\min(M,m))$ | with $M<m$ only $M$ eigenvalues are nonzero |
| ASX, NASX | 1 | $O(Mm^2+m^3)$ | |
| OLS | $m+1$ | $O(Mm^2)$ | one direction only |
| QPHD | $(m+1)(m+2)/2$: 66 at $m=10$, 1326 at $m=50$ | $O(Mm^4)$ | practical up to $m\approx50$ |
| SIR, SAVE | $2S$ (20 with the default 10 slices) | $O(Mm^2+m^3)$ | each slice needs enough points for its mean (SIR) or covariance (SAVE) |
| pHd | 2 | $O(Mm^2+m^3)$ | |
| SS | 2 | $O(M^2m^2)$ | quadratic in $M$ |
| OPG | $m+1$ distinct points | $O(Mm^3)$ plus the neighbour search | the local bias needs dense sampling |

A hard minimum only makes the matrix computable; it says nothing about accuracy. The bootstrap is the
library's check: if the bootstrap distances $\hat\delta_k$ are large for every $k$, the sample is too
small for the estimator. The bootstrap multiplies the cost by $B$.

For a sense of scale, the library's calibrations of its acceptance tests found these 99.9%
quantiles of the subspace distance $\lVert\hat W_1\hat W_1^\top-W_1W_1^\top\rVert$ (single cases, not
guarantees): 0.068 for AS from $M=1000$ Monte Carlo gradients of a quadratic in 5 uniform inputs;
0.045 for OLS from $M=300$ values of a monotone one-dimensional ridge in 7 uniform inputs; 0.102 for
SIR from $M=2000$ values of a monotone one-dimensional ridge in 6 Gaussian inputs.

## 4. Do you need the dimension chosen?

| Rule | Bootstrap | Use with | Notes |
|---|---|---|---|
| EVG, linear gap (default) | no | any estimator | favours small $n$ when $\lambda_1$ dominates |
| EVG, log gap | no | any estimator | scale-free; responds to relative drops |
| RS | yes | AS, OPG | ranks the error bound of the ridge approximation; warns with other estimators |
| LI (ladle) | yes | PSD estimators | warns with ASX, NASX, pHd; the paper takes $B=M$ replicates |
| fixed `n` | no | any | no warning |

With an indefinite estimator, every rule works on $\lvert\lambda\rvert$: a large negative eigenvalue
counts as much as a large positive one. Whatever the rule, look at the eigenvalues with their
bootstrap intervals and at the bootstrap distances before relying on $n$. Changing $n$ afterwards
with `with_dimension` costs nothing: the eigenpairs and replicates are reused.

## Comparison

| Estimator | Needs | PSD | Uniform | Gaussian | Dimension rules without warning | Published reference |
|---|---|---|---|---|---|---|
| `as` | gradients | yes | valid | valid | EVG, RS, LI | yes |
| `nas` | gradients | yes | valid | valid | EVG, LI | none |
| `asx` | gradients, points | no | warns | valid | EVG | none |
| `nasx` | gradients, points | no | warns | valid | EVG | none |
| `opg` | values, points | yes | valid | valid | EVG, RS, LI | yes, with a different kernel |
| `ols` | values, points | yes, rank 1 | warns | valid | EVG, LI | classical least squares |
| `qphd` | values, points | yes | warns, biased | valid | EVG, LI | yes |
| `sir` | values, points | yes | warns, biased | valid | EVG, LI | yes |
| `phd` | values, points | no | warns, biased | valid | EVG | yes |
| `save` | values, points | yes | warns, biased | valid | EVG, LI | yes |
| `ss` | values, points | yes | warns, biased | warns | EVG, LI | none |

"Valid" means that the population matrix is ridge-exact under that distribution; "biased" means that
a bias has been measured on the uniform cube. For OLS no bias figure is documented; it warns because
the linearity condition fails on the cube in general.

## References

- Constantine, P. G. (2015). *Active Subspaces: Emerging Ideas for Dimension Reduction in Parameter
  Studies.* SIAM.
- Cook, R. D. and Weisberg, S. (1991). Discussion of "Sliced inverse regression for dimension
  reduction". *Journal of the American Statistical Association* 86(414):328–332.
- Li, K.-C. (1991). Sliced inverse regression for dimension reduction. *Journal of the American
  Statistical Association* 86(414):316–327.
- Li, K.-C. (1992). On principal Hessian directions for data visualization and dimension reduction:
  another application of Stein's lemma. *Journal of the American Statistical Association*
  87(420):1025–1039.
- Luo, W. and Li, B. (2016). Combining eigenvalues and variation of eigenvectors for order
  determination. *Biometrika* 103(4):875–887. (details to verify)
- Xia, Y., Tong, H., Li, W. K. and Zhu, L.-X. (2002). An adaptive estimation of dimension reduction
  space. *Journal of the Royal Statistical Society, Series B* 64(3):363–410.
