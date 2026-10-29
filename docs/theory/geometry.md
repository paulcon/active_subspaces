(theory-geometry)=
# Geometry of the active variables

Once a subspace is chosen, three geometric objects are needed to use it: the **active domain**
$\mathcal Y$, the set of values the active variables $y=W_1^\top x$ can take; the **forward map**
$x\mapsto(y,z)$; and the **inverse map**, which returns inputs $x$ with a given $y$, drawn from the
conditional distribution of the inactive variables. Everything here is in normalized coordinates.
Write $d=m-n$ for the number of inactive variables and $g_j=(W_1)_{j,:}^\top\in\mathbb R^n$ for the
rows of $W_1$, one per input. Geometry needs at least one inactive variable, $n\le m-1$.

## Active domains

| Space | Active domain $\mathcal Y$ | Distribution of $y$ | `kind` |
|---|---|---|---|
| `Gaussian` | $\mathbb R^n$ | $N(0,I_n)$, independent of $z\sim N(0,I_d)$ | `"unbounded"` |
| `Uniform`, $n=1$ | $[-\lVert w\rVert_1,\ \lVert w\rVert_1]$, $w=W_1$ | a box spline (see [Integration and optimization](integration_and_optimization.md)) | `"interval"` |
| `Uniform`, $n\ge2$ | the zonotope $Z(W_1)=W_1^\top[-1,1]^m=\sum_{j=1}^m[-g_j,g_j]$ | the push-forward of the uniform density | `"zonotope"` |

For Gaussian inputs the rotation $x\mapsto W^\top x$ preserves $N(0,I_m)$, so $y$ and $z$ are
independent standard normals. For the uniform cube, the image of $[-1,1]^m$ under the linear map
$W_1^\top$ is a zonotope: a centrally symmetric polytope that is the Minkowski sum of the segments
$[-g_j,g_j]$ [Constantine 2015].

### Facts used

These are standard facts about zonotopes [McMullen 1971; Ziegler 1995, ch. 7].

- **Support function.** For $\nu\in\mathbb R^n$, $\max_{y\in Z}\nu^\top y=\sum_j\lvert g_j^\top\nu\rvert=\lVert W_1\nu\rVert_1$.
  Hence
  $$Z(W_1)=\bigl\lbrace y:\ \nu_k^\top y\le\lVert W_1\nu_k\rVert_1,\ k=1,\dots,N_f\bigr\rbrace$$
  over the unit outward facet normals $\nu_k$, which come in pairs $\pm\nu$. The library stores this
  description as `facet_normals` $A$ and `facet_offsets` $b$, with $\mathcal Y=\lbrace y:Ay\le b\rbrace$.
- **Reduction.** A zero generator adds nothing, and parallel generators merge into one segment whose
  length is the sum of theirs. The reduced generators $h_1,\dots,h_{m''}$ are pairwise non-parallel
  and define the same zonotope.
- **Facets.** Every facet normal is orthogonal to a set of generators of rank $n-1$. The facet with
  normal $\nu$ is a translate of the $(n-1)$-dimensional zonotope of the generators orthogonal to
  $\nu$, shifted by $\sum_{c:\,h_c^\top\nu\ne0}\mathrm{sign}(h_c^\top\nu)\,h_c$.
- **Vertices.** Every vertex is $v=\sum_c\sigma_ch_c$ for a sign vector $\sigma\in\lbrace\pm1\rbrace^{m''}$.
  Its preimages in the cube form a face: $x_j=\mathrm{sign}(\alpha_j)\,\sigma_{c(j)}$ for each input
  whose row is nonzero, written $g_j=\alpha_j\bar h_{c(j)}$ with $\bar h_c$ the unit direction of the
  merged generator $c$; and $x_j$ free for each input whose row of $W_1$ is zero. The preimage
  is a single corner exactly when $W_1$ has no zero row. The library stores the preimages with
  entries in $\lbrace-1,0,+1\rbrace$, a 0 marking an inert input (0 is the centre of the free face).
- **Counts.** In general position (no zero rows, no two rows parallel, any $n$ rows linearly
  independent) there are
  $$N_v=2\sum_{i=0}^{n-1}\binom{m-1}{i}\ \text{vertices}\qquad\text{and}\qquad N_f=2\binom{m}{n-1}\ \text{facets};$$
  otherwise these are upper bounds. `zonotope_vertex_count(m, n)` returns $N_v$.

### Vertex enumeration

`active_domain` computes the vertices and facets exactly and deterministically:

1. **Reduce** the generators: drop rows with $\lVert g_j\rVert\le\tau\max_k\lVert g_k\rVert$ and merge
   rows whose directions agree to $\tau$ (`tol`, default $10^{-10}$).
2. **$n=1$:** the interval $[-\lVert w\rVert_1,\lVert w\rVert_1]$, with preimages $\mp\mathrm{sign}(w)$.
3. **$n=2$:** orient each generator into the upper half-plane and sort by angle. Adding the
   generators in angular order walks the $2m''$ vertices counter-clockwise from $-\sum_ch_c$; each
   edge gives a facet.
4. **$n\ge3$:** for each subset of $n-1$ generators of full rank, the normal $\nu$ of their span is a
   facet normal. The facet's vertices are the vertices of the lower-dimensional zonotope of the
   generators orthogonal to $\nu$, found by the same procedure one dimension down, combined with
   the signs $\mathrm{sign}(h_c^\top\nu)$ of the others. Both $\nu$ and $-\nu$ are recorded, and each
   hyperplane is processed once.

Vertices and offsets are then computed from the original $W_1$. The 2016 code searched for vertices
with up to $10^9$ random directions, could stop with an incomplete hull, and produced duplicate
vertices outside general position. The exact enumeration computes the same mathematical object, so
it is not a change of method.

**Cost and size.** Time is $O\bigl(\binom{m''}{n-1}(m''n+n^3)\bigr)$ for $n\ge3$ and $O(m\log m)$
for $n=2$. Memory is about $8N_v(m+n)$ bytes for the vertices and their preimages, so bounded
geometry depends on $m$ as well as $n$:

| $n$ | $N_v(m,n)$ | $m=30$ | $m=100$ | $m=300$ | $m=1000$ |
|---|---|---|---|---|---|
| 2 | $2m$ | 60 | 200 | 600 | 2 000 |
| 3 | $m^2-m+2$ | 872 | 9 902 | 89 702 | 999 002 |
| 4 | $2\sum_{i<4}\binom{m-1}{i}$ | 8 180 | 323 600 | 8.9 M | 332 M |

It is practical while $N_v(m+n)\le10^8$ (about 0.8 GB): $n=2$ throughout, $n=3$ up to $m\approx300$
and $n=4$ up to $m\approx100$. The library does not enforce this; check with
`zonotope_vertex_count` first. The Gaussian domain has no vertices.

## Forward map

$$y=W_1^\top x,\qquad z=W_2^\top x,$$

applied row-wise to normalized inputs. `forward_map` takes physical `X` and normalizes it with the
subspace's input space.

## Inverse map

The inverse map returns, for each given $y$, inputs $x=W_1y+W_2z$ with $z$ drawn from the
conditional distribution $\pi(z\mid y)$ under $\rho$. Because $W$ is orthogonal, $(y,z)$ is
distributed as $\rho$ rotated.

**Gaussian inputs.** $z\mid y\sim N(0,I_d)$, independent of $y$, so the draws are exact and
independent.

**Uniform inputs.** $\rho$ is constant on the cube, so $z\mid y$ is uniform on the polytope

$$\mathcal P(y)=\bigl\lbrace z\in\mathbb R^d:\ -\mathbf 1\le W_1y+W_2z\le\mathbf 1\bigr\rbrace .$$

Equivalently, $x$ is uniform, with respect to $d$-dimensional volume, on the **fiber**
$F(y)=\lbrace x\in[-1,1]^m:W_1^\top x=y\rbrace=W_1y+W_2\mathcal P(y)$, since $W_2$ is an isometry onto
the fiber's directions. $\mathcal P(y)$ is nonempty exactly when $y\in\mathcal Y$.

### Exact sampler for one inactive variable

When $d=1$ the fiber is a segment $s+t\,w_2$ with $s=W_1y$ and $w_2=W_2$. Its ends are

$$t_{\mathrm{lo}}=\max_{j:\,w_{2j}\ne0}\min\Bigl(\tfrac{-1-s_j}{w_{2j}},\tfrac{1-s_j}{w_{2j}}\Bigr),\qquad t_{\mathrm{hi}}=\min_{j:\,w_{2j}\ne0}\max\Bigl(\tfrac{-1-s_j}{w_{2j}},\tfrac{1-s_j}{w_{2j}}\Bigr),$$

and $t\sim U(t_{\mathrm{lo}},t_{\mathrm{hi}})$ gives an exact, independent draw. The point $s$ is the
point of the affine set nearest the origin; it need not lie in the cube, and only $s+tw_2$ must. A
row with $w_{2j}=0$ does not bound $t$; it requires $\lvert s_j\rvert\le1$.

### Hit-and-run for two or more inactive variables

When $d\ge2$ the library runs hit-and-run [Smith 1984] on the fiber, in $x$ coordinates. From the
current point $x$, draw $u\sim N(0,I_m)$ and project it onto the fiber's directions,
$\delta=(I-W_1W_1^\top)u=W_2(W_2^\top u)$. Since $W_2^\top u\sim N(0,I_d)$, the direction is uniformly
distributed in the fiber. The chord through $x$ along $\delta$ is $\lambda\in[\lambda_{\mathrm{lo}},\lambda_{\mathrm{hi}}]$
with

$$\lambda_{\mathrm{lo}}=\max_{j:\,\delta_j\ne0}\min\Bigl(\tfrac{-1-x_j}{\delta_j},\tfrac{1-x_j}{\delta_j}\Bigr),\qquad\lambda_{\mathrm{hi}}=\min_{j:\,\delta_j\ne0}\max\Bigl(\tfrac{-1-x_j}{\delta_j},\tfrac{1-x_j}{\delta_j}\Bigr),$$

over all inputs. Draw $\lambda\sim U(\lambda_{\mathrm{lo}},\lambda_{\mathrm{hi}})$ and move to
$x+\lambda\delta$. This is Smith's kernel, whose stationary distribution is uniform on the fiber. It
is the same chain as hit-and-run on $\mathcal P(y)$ in $z$ coordinates. Every constraint enters the
chord; the 2016 code ignored constraints with small slack.

- **Start.** The chain starts at the Chebyshev centre of the fiber, the centre of the largest ball in
  the fiber, found by the linear program
  $$\max_{x,\,r}\ r\quad\text{s.t.}\quad W_1^\top x=y,\quad \pm x_i+\kappa_ir\le1,\quad r\ge0,\qquad\kappa_i=\lVert(W_2)_{i,:}\rVert,$$
  since the distance within the fiber from $x$ to the face $x_i=\pm1$ is $(1\mp x_i)/\kappa_i$.
- **Burn-in and thinning.** By default the chain discards the first $10d$ steps and keeps every
  $d$-th step after that (`burn_in`, `thin`). Draws from one chain are correlated; with `thin=1` and
  more than one draw the library warns (`accuracy.correlated_draws`). The integration error estimate
  accounts for the remaining correlation (see [Integration and optimization](integration_and_optimization.md)).
- **Feasibility.** Every recorded draw is checked: it must lie in the cube and satisfy $W_1^\top x=y$
  to tight tolerances, or the call raises. Each step re-projects onto the affine set so that
  rounding does not accumulate.
- **Cost.** Each step costs $O(mn)$, so a point costs $O\bigl((10d+N\,d)\,mn\bigr)$ for $N$ draws with
  the defaults, plus one linear program.

### Boundary points

On the boundary of $\mathcal Y$ the fiber has empty interior: at a vertex in general position it is
a single corner of the cube. The conditional density is not defined there (the boundary has measure
zero), so this is not a departure from the method. The library returns copies of the feasible point
found by the start, flags the row in `degenerate`, and warns (`accuracy.degenerate_fiber`). A $y$
outside $\mathcal Y$ raises an `InputError` (`input.out_of_domain`).

Each point $y_k$ draws from its own random stream, so the draws do not depend on how many points are
processed together.

## References

- Constantine, P. G. (2015). *Active Subspaces: Emerging Ideas for Dimension Reduction in Parameter
  Studies.* SIAM.
- McMullen, P. (1971). On zonotopes. *Transactions of the American Mathematical Society*
  159:91–109. (details to verify)
- Smith, R. L. (1984). Efficient Monte Carlo procedures for generating points uniformly distributed
  over bounded regions. *Operations Research* 32(6):1296–1308.
- Ziegler, G. M. (1995). *Lectures on Polytopes.* Graduate Texts in Mathematics 152, Springer.
