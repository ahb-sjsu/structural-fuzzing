# Chapter {{ch:spectral-geometry-and-the-angular-basis}}: Spectral Geometry and the Angular Basis

*Structural Fuzzing: Geometric Methods for Adversarial Model Validation — Andrew H. Bond*

---

The preceding foundational chapters equipped us with metrics on data
(Chapter {{ch:mahalanobis-distance}}), spaces for hierarchy
(Chapter {{ch:hyperbolic-geometry}}), manifolds of covariance
(Chapter {{ch:spd-manifolds}}), and the topology of point clouds
(Chapter {{ch:topological-data-analysis}}). Each treats geometry as something we
*impose* on the data. This chapter takes the opposite stance: we let a graph tell
us its *own* geometry, and we ask a validation question of it — **does the model's
response landscape have coherent geometric structure at all, and if so, can we
read it robustly?**

The setting is native to structural fuzzing. A campaign samples many parameter
configurations and scores each one; the result is a cloud of configurations
carrying error signals. Connect nearby configurations and you have a graph whose
large-scale shape *is* the failure landscape. A well-behaved model produces a
landscape with clean low-dimensional structure — small parameter moves cause
small, predictable error changes. A fragile or mis-specified model produces a
*tangle* — a high-dimensional, shortcut-riddled graph where nearby configurations
behave nothing alike. Telling these apart, reliably and at scale, is the job of
this chapter.

We will develop three tools. First, a **dimension detector** that flags whether
the response landscape is a manifold or a tangle. Second, the **angular basis** —
a scale-invariant reading of the graph's geometry that, we will prove, survives a
degeneracy which destroys the naive spectral reading. Third, a
**geodesic-preservation score** that behaves as a structural-integrity metric: it
is high when the landscape has genuine geometry and collapses when that geometry
is destroyed. Throughout we practice *controls-first* validation — no detector is
trusted until it has passed a positive and a negative control.

## {{ch:spectral-geometry-and-the-angular-basis}}.1 The response landscape as a graph

Let a campaign produce configurations $x_1,\dots,x_n \in \mathbb{R}^p$ (each a
parameter vector) with scalar errors $e_1,\dots,e_n$. Build the *response graph*
$G$ by connecting each configuration to its $k$ nearest neighbours in a metric
that blends parameter distance and error similarity (the Mahalanobis metric of
Chapter {{ch:mahalanobis-distance}} is a natural choice for the parameter part).
The adjacency $A$ is symmetric and binary; the degree matrix is
$D=\mathrm{diag}(\deg(1),\dots,\deg(n))$.

The object we analyse is the **symmetric-normalized Laplacian**

$$
L \;=\; I - D^{-1/2} A D^{-1/2},
$$

with eigenpairs $(\lambda_k, v_k)$, $0=\lambda_0<\lambda_1\le\cdots$. The low
eigenvectors are the smooth coordinates of the landscape; the commute-style
embedding using the lowest $m$ nontrivial modes places configuration $i$ at

$$
\Psi_i \;=\; \Bigl(\tfrac{v_1(i)}{\sqrt{\lambda_1}},\dots,
\tfrac{v_m(i)}{\sqrt{\lambda_m}}\Bigr) \in \mathbb{R}^m .
$$

Everything that follows is a statement about how much of the landscape's geometry
$\Psi$ actually preserves — and which part of it to trust.

## {{ch:spectral-geometry-and-the-angular-basis}}.2 The commute-time trap

It is tempting to read distances directly from $\Psi$: configurations that are
far apart in $\lVert\Psi_i-\Psi_j\rVert$ are "geometrically distant" in the
landscape. This is the commute (resistance) distance, and for large graphs it is
a trap.

**The degeneracy.** von Luxburg, Radl and Hein proved that on large geometric
graphs the resistance distance degenerates to a function of *local degrees alone*:

$$
R(i,j) \;\longrightarrow\; \frac{1}{\deg(i)} + \frac{1}{\deg(j)} ,
$$

losing every trace of global geometry. Written in polar form
$\Psi_i = r_i\,\hat u_i$ with radius $r_i=\lVert\Psi_i\rVert$ and angle
$\hat u_i=\Psi_i/\lVert\Psi_i\rVert$, the degeneracy lands entirely on the
**radius**: asymptotically $r_i \to 1/\sqrt{\deg(i)}$, a pure local-density
coordinate that carries *no* information about where configuration $i$ sits in the
landscape. A fuzzing pipeline that ranks configurations by commute distance is, at
scale, ranking them by inverse square-root degree — a sampling artefact, not a
geometric signal. This is the single most common way a spectral structural-probe
silently fails.

## {{ch:spectral-geometry-and-the-angular-basis}}.3 Keep the angle: the angular basis

If the radius is noise, the geometry must be in the **angle**. Row-normalizing the
embedding to the unit sphere — replacing $\Psi_i$ by $\hat u_i$ — deletes exactly
the degenerate density factor and keeps the direction of the low-mode eigenmap.
This is the same normalization that spectral clustering applies before $k$-means,
but here it earns a *geometric* reading rather than a clustering one: the angular
coordinates preserve the graph's geodesic (shortest-path) structure.

Empirically the effect is stark. On a reference landscape with clean 2-dimensional
geometry, the rank correlation between embedded distance and true graph geodesic
distance is $\approx 0.9$ for the **angle**, $\approx 0$ for the **magnitude**,
and $\approx 0$ for a *random-mode* basis of the same size — and, crucially, the
angular figure is **flat** in both the number of modes $m$ and the graph size $n$,
exactly the regime where the full commute reading decays. The practical rule for
structural fuzzing is therefore blunt:

> **Read the landscape from the angle, not the magnitude.** The angular basis is
> scale-invariant and stable under resolution and sample size; the raw embedding
> is neither.

We record the structural claim as a principle with a proven half and a
conjectural half, so the reader knows exactly what is load-bearing.

**Radial Degeneracy (proven).** For a response graph of intrinsic dimension
$d\ge2$ meeting the von Luxburg conditions, the radius of the full commute
embedding is asymptotically geometry-free, $r_i\to 1/\sqrt{\deg(i)}$.

**Angular Preservation (conjecture, strong empirical support).** The
row-normalized direction remains *rank-faithful* to geodesic distance — Spearman
correlation bounded away from zero — uniformly in $m$ and $n$. A metric
(bi-Lipschitz) form is consistent with the evidence but untested; our measurements
are rank-based and cannot by themselves bound metric distortion.

The honest gap is worth stating plainly: the degeneracy result explains why the
radius *dies*, not by itself why the angle *lives*. We use the angle because it
works, controlled and measured, not because it is proven optimal.

## {{ch:spectral-geometry-and-the-angular-basis}}.4 Manifold-integrity detection

Before reading a landscape's geometry we must know whether it *has* any. Estimate
its intrinsic dimension three independent ways and demand that they agree:

1. **Ball growth** — the number of configurations within graph-distance $r$ grows
   as $N(r)\sim r^{d}$ (the Hausdorff-style estimate of
   Chapter {{ch:topological-data-analysis}}'s persistence cousin).
2. **Spectral dimension** — the heat-kernel return probability
   $P(t)=\langle e^{-Lt}\rangle \sim t^{-d_s/2}$.
3. **Effective rank** — the participation ratio of a local PCA of the Laplacian
   eigenmap.

Low spread among the three is the signature of a genuine manifold; wide
disagreement is the signature of a tangle. This is a direct validation criterion:
a model whose response landscape is a clean $\approx2$-dimensional manifold is
one whose failures are locally predictable, while a model whose landscape scores
$d=1.6$ by one estimator and $d=8$ by another has a pathological sensitivity
structure that no scalar metric will reveal. *Estimator agreement is the
manifold-grade certificate; the agreed value is the reported dimension.*

## {{ch:spectral-geometry-and-the-angular-basis}}.5 The geodesic-preservation score

Combine the previous two ideas into one number. Sample anchor configurations,
compute their true pairwise graph geodesics, and measure the Spearman correlation
$\rho$ between those geodesics and the distances in the **angular** embedding. Call
$\rho$ the *geodesic-preservation score*. It behaves as a structural-integrity
metric:

- On a landscape with genuine low-$d$ geometry, $\rho \approx 0.9$.
- Destroy the geometry — inject a few percent of long-range shortcuts (a
  small-world perturbation, the analogue of a model developing erratic long-range
  parameter sensitivities) — and $\rho$ **collapses** toward $0.4$ while the
  measured dimension inflates. The score detects the damage that the dimension
  estimate alone can miss.
- Feed it a landscape whose geometry is not Riemannian at all (a causal, ordered
  structure) and $\rho$ correctly refuses to certify it.

For adversarial validation this is the payoff: $\rho$ is a single, scale-robust
scalar that *falls when the landscape's geometry breaks*, giving a fuzzing campaign
a geometry-integrity alarm alongside its error metrics.

One caveat keeps the metric honest. Fidelity is not uniform across dimension: on a
suite of independent emergent manifolds spanning $d\approx1$–$3.6$, $\rho$ falls
roughly linearly with intrinsic dimension (slope $\approx-0.15$ per dimension).
The angular score is sharpest on low-dimensional landscapes and softens as the
landscape grows higher-dimensional and rougher — so thresholds must be set
per-dimension, not globally.

## {{ch:spectral-geometry-and-the-angular-basis}}.6 Controls first

No geometric detector earns trust until it has passed a **positive** and a
**negative** control — a discipline this book returns to whenever a probe's output
would otherwise be unfalsifiable. The pattern, applied here:

- **Positive control.** Run the detector on a graph of *known* geometry — a
  random geometric graph on a flat torus (intrinsic dimension 2). The estimators
  must recover $d\approx2$ and $\rho$ must be high. If they do not, the detector is
  miscalibrated and no result on real data is meaningful.
- **Negative control.** Run it on a graph of *known non-geometry* — a tree, or a
  heavily rewired lattice. The detector must *refuse* to certify it.

A cautionary example closes the loop with Chapter {{ch:hyperbolic-geometry}}.
Response tangles grow their configurations quickly, which tempts one to declare
them "hyperbolic" and reach for the Poincaré ball. We tested that hypothesis with
controls — a binary tree as the positive hyperbolic control, a torus as the
negative — using a growth-law discriminator (does ball volume grow like $r^d$ or
$e^{(d-1)r}$?) validated against both. The discriminator correctly labels the tree
hyperbolic and the torus Euclidean; run on the response tangles it labels them
**Euclidean-leaning high-dimensional non-manifolds, not hyperbolic**. The elegant
reframe was wrong, and only the controls revealed it. Exponential growth in the
*count* of configurations is not exponential growth in the *metric*; do not confuse
the two.

## {{ch:spectral-geometry-and-the-angular-basis}}.7 Implementation

The `spectral_probe` module packages these tools against the structural-fuzzing
evaluate-function interface. Given a matrix of sampled configurations, it builds
the response graph, estimates dimension three ways, computes the angular
geodesic-preservation score with its controls, and returns a manifold-integrity
report.

```python
from structural_fuzzing.spectral_probe import (
    response_graph, intrinsic_dimension, angular_fidelity, integrity_report,
)

# configs: (n, p) sampled parameter vectors; errors: (n,) scalar errors
A = response_graph(configs, errors, k=10)          # kNN response graph
dim, spread = intrinsic_dimension(A)               # ball / spectral / eff-rank
rho, controls = angular_fidelity(A)                # angle score + random control

report = integrity_report(A)
print(report)   # dimension, spread, angle-rho, and a manifold/tangle verdict
```

`angular_fidelity` returns the low-mode **angular** score together with a
random-mode control; a result is only reported as geometric when the angle score
clears its dimension-adjusted threshold *and* the random control is near zero.
`integrity_report` refuses to certify a landscape whose estimators disagree
(spread above tolerance) — the manifold-grade gate of Section
{{ch:spectral-geometry-and-the-angular-basis}}.4 — so a tangled response landscape
is flagged rather than silently mis-measured.

## {{ch:spectral-geometry-and-the-angular-basis}}.8 Limitations

The angular-preservation claim is conjectural in its metric form (Section
{{ch:spectral-geometry-and-the-angular-basis}}.3); we have rank evidence, not a
distortion bound. On landscapes that embed near-isometrically in low-dimensional
Euclidean space — flat, unobstructed parameter regions — classical
multidimensional scaling can match or beat the angular basis; the angle's
advantage is on landscapes whose topology obstructs a flat embedding, and its
*uniformity in resolution and size*, which MDS lacks. Finally, fidelity degrades
with intrinsic dimension, so the geodesic-preservation score is a sharper alarm
for low-dimensional response landscapes than for high-dimensional ones. Used with
its controls and per-dimension thresholds, it is nonetheless a robust addition to
the structural-fuzzing toolkit: a scale-invariant read of *whether the failure
landscape has coherent geometry, and where it breaks*.

---

### Exercises

**{{ch:spectral-geometry-and-the-angular-basis}}.1.** Build a response graph from
a campaign of your own. Compute the commute embedding and, separately, the angular
embedding. Correlate each with the true graph geodesics as you increase $n$; verify
that the angular score is flat while the commute score decays.

**{{ch:spectral-geometry-and-the-angular-basis}}.2.** Implement the three
dimension estimators and reproduce the manifold-grade certificate on a torus
(positive control) and a random tree (negative control).

**{{ch:spectral-geometry-and-the-angular-basis}}.3.** Inject long-range shortcuts
into a clean 2D response graph at rewiring fractions $0, 0.01, 0.03, 0.1$. Plot the
geodesic-preservation score against the rewiring fraction and identify the
integrity threshold.

**{{ch:spectral-geometry-and-the-angular-basis}}.4.** Take a response tangle and
test the hyperbolic hypothesis with the growth-law discriminator of Section
{{ch:spectral-geometry-and-the-angular-basis}}.6, using a tree and a torus as
controls. Report the verdict.

---

### Bibliographic Notes

The commute-time degeneracy is due to von Luxburg, Radl and Hein, *Hitting and
commute times in large random neighborhood graphs* (JMLR, 2014). Row-normalized
spectral embeddings were introduced for clustering by Ng, Jordan and Weiss (2002);
the sharpest analysis of *why* row normalization concentrates cluster directions is
Schiebinger, Wainwright and Yu, *The geometry of kernelized spectral clustering*
(Annals of Statistics, 2015) — an analysis of cluster separation, complementary to
the geodesic-preservation question studied here. Laplacian eigenmaps are Belkin and
Niyogi (2003); diffusion maps Coifman and Lafon (2006). The polar
magnitude/direction decomposition also appears in KV-cache quantization (Han et
al., *PolarQuant*, 2025). The angular-basis results, the manifold-integrity
certificate, and the controlled hyperbolic rejection are developed in the author's
*Keep the Angle* (2026); the observer-theoretic reading of angular coarse-graining
connects to the Wolfram-model literature but is not required for any result here.
