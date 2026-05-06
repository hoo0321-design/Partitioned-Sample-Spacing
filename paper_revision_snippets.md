# NeurIPS Revision Snippets

This file collects text blocks that can be pasted into the manuscript once the LaTeX source is added to the repository.

## Lower-Claim Abstract Direction

We propose Partitioned Sample-Spacing (PSS), a simple training-free estimator for multivariate differential entropy. PSS partitions the sample space into occupied hyperrectangles and applies marginal spacing estimates locally within each cell. Rather than relying on neighbor search or fitting an auxiliary density model, the estimator exposes a finite-sample tradeoff between local resolution and cell occupancy. We prove consistency under shrinking-cell and within-cell sample growth conditions, and we evaluate the estimator across Gaussian-copula families with Normal, Gamma, Beta, and Lognormal marginals. The experiments compare oracle-tuned and cross-validated partition choices and report occupancy, skipped-cell, and skipped-point diagnostics. The results show that PSS is competitive in moderate-dimensional correlated or skewed regimes when occupied cells retain sufficient samples, while performance degrades when partition occupancy becomes too sparse. These diagnostics position PSS as a practical training-free baseline and clarify its failure modes relative to kNN, CADEE, and normalizing-flow-based estimators.

## Within-Cell Independence Clarification

The product form used inside each occupied partition should be interpreted as a local approximation rather than as an assumption of exact finite-cell conditional independence. For a cell \(C_k\) containing \(x\), PSS estimates a local surrogate of the form

\[
\widetilde f_k(x)
= P(X \in C_k) \prod_{j=1}^d f_{X_j \mid X \in C_k}(x_j),
\]

where each one-dimensional factor is estimated by sample spacings computed only from points in \(C_k\). For a fixed coarse partition this surrogate need not equal the true joint density inside the cell. The asymptotic argument instead relies on the simultaneous conditions that the cells shrink and that the number of samples in occupied cells grows. Under smoothness of \(f\), the local discrepancy between the true log-density and the surrogate vanishes for the cell containing the evaluation point:

\[
\sup_{u \in C_k}
\left|\log f(u) - \log \widetilde f_k(u)\right|
\to 0
\quad \text{as } \operatorname{diam}(C_k) \to 0.
\]

Thus the finite-sample estimator may be biased when cells are large or when substantial within-cell dependence remains, and this is why the experiments report occupancy and skipped-point diagnostics together with estimation error.

## Stable-Coverage Cross-Validation (SC-CV)

For data-driven partition selection, we use stable-coverage cross-validation (SC-CV). For a candidate partition level \(\ell\), a validation point is called stably covered if its held-out PSS log-density is finite and the corresponding training cell contains at least \(n_{\min}\) observations. Let

\[
S(\ell)
= \frac{1}{N_{\mathrm{val}}}
\sum_{i \in \mathrm{val}}
\mathbf 1\left\{
\widehat f_{\ell}^{(-i)}(X_i) \text{ is finite and }
n_{k(i)}^{(-i)} \ge n_{\min}
\right\}.
\]

The selected partition level is

\[
\widehat \ell_{\mathrm{SC-CV}}
= \arg\min_{\ell}
\mathrm{CVNLL}(\ell)
\quad
\text{subject to}
\quad
S(\ell) \ge 0.99.
\]

In the experiments we set \(n_{\min}=10\). This single stable-coverage constraint combines the two finite-sample failure modes that reviewers highlighted: skipped validation points and sparse occupied cells. Unlike a hard upper bound on \(\ell\), the constraint is data-driven because feasibility is determined by the empirical validation occupancy profile.

## Occupancy-Aware Complexity Rewrite

The practical runtime is governed by occupied cells rather than by the nominal number \(\ell^d\) of possible cells. Let \(\mathcal O\) denote the set of occupied partitions and let \(n_k\) be the number of observations in cell \(k\). Assigning samples to cells and evaluating the density contributes an \(O(Nd)\) pass. The dominant additional cost is sorting each coordinate within each occupied cell:

\[
T_{\mathrm{PSS}}
= O\left(
Nd + d \sum_{k \in \mathcal O} n_k \log n_k
\right).
\]

Since \(\sum_{k \in \mathcal O} n_k = N\),

\[
\sum_{k \in \mathcal O} n_k \log n_k
\le
N \log\left(\max_{k \in \mathcal O} n_k\right).
\]

Therefore PSS approaches a near-linear pass only in regimes where the sample mass is spread across occupied cells so that \(\max_k n_k\) is small. Increasing \(\ell\) does not by itself guarantee \(O(Nd)\) behavior; the relevant finite-sample quantity is the empirical occupancy profile. This motivates reporting occupied-cell fraction, skipped-cell fraction, skipped-point fraction, and a sorting-cost proxy alongside runtime.

## Discussion Limitation Paragraph

PSS is not intended to replace all entropy estimators. It occupies a useful part of the design space: it is training-free, simple to tune, and can be competitive when local partitions retain enough samples. Its weaknesses are also explicit. Equal-width partitions can become sparse in very high dimension, heavy-tailed data can allocate many cells to low-density regions, and the within-cell product approximation can be poor when dependence remains strong inside coarse cells. The occupancy diagnostics reported in the experiments are therefore part of the estimator's finite-sample interpretation, not merely implementation details.
