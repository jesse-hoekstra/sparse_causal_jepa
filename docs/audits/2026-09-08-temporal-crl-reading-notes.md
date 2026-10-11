# Temporal causal representation learning and visual state recovery

Date: 2026-09-08. Scope: analysis of the seven PDF versions supplied by the user, compared with the current SCJEPA manuscript and bounce implementation. This is a reading memo and a set of research proposals, not a protocol decision or an implemented model change. Page numbers below are one-based PDF pages; the SCJEPA manuscript's printed page numbers differ by two.

## The connection to our missing state-recovery argument

The supplied papers are directly relevant to recovering the time-varying state from images. That is a different identification problem from recovering the episode-constant parameters that govern the state's evolution.

Our manuscript makes this separation explicit. Assumption 1, PDF page 8, requires a parameter-independent state chart that is separable into object blocks, up to object permutation. Assumption 2, PDF page 9, assumes that both encoders recover that chart, with a nondegenerate representation sufficient for the Markov dynamics. Remark 3 explicitly states that EMA and stop-gradient do not prove these properties. Theorem 1, PDF page 13, identifies latent **system parameters**, conditional on these and further assumptions. It does not establish that the visual encoder learns the required object states.

The temporal-CRL literature offers possible principles for addressing this first stage. Transferring a result requires matching its observation assumptions, data variation, objective, and equivalence class. A theorem identifying an arbitrary full-state embedding would not by itself supply our stronger object-block-separable, parameter-independent chart.

For deterministic dynamics, the basic ambiguity is immediate. If `s_next = F(s, theta)`, any invertible entangling map `h` permits the equally accurate predictor

`F_h(y, theta) = h(F(h_inverse(y), theta))`.

Predicting farther ahead does not remove this ambiguity. A sufficiently flexible predictor can operate in the transformed coordinates at every horizon. If the representation is not required to retain state information, constant or partial encodings create further degeneracies. This is an argument about prediction alone, not a proof that every additional restriction in our complete model is irrelevant.

The useful distinction is between preserving information and identifying its factorization. Reconstruction can prevent information loss, but alone permits invertible entanglements. A temporal identifying restriction can select coordinates, but cannot recover information already discarded by its input encoder.

## What each paper establishes

### Taming VAEs — 1810.00597v1

Rezende and Viola's paper contributes GECO, not a theorem identifying latent causal states. Equation 2 on page 3 formulates constrained VAE optimization: minimize the KL cost while meeting chosen reconstruction constraints. Section 3.4 and Algorithm 1 on pages 6–7 give the moving-average constraint estimate and multiplicative Lagrange-multiplier update. Table 1 on page 8 illustrates several possible fidelity constraints, including pixels and features.

For our work, GECO is a way to control an explicitly chosen tradeoff. Its presence does not establish that the constrained quantity contains object position or velocity. In particular, fitting a jointly learned target does not become an observation-information guarantee merely because its error is constrained. Our existing use of GECO therefore does not fill Assumption 2.

### Temporal sparse coding / SlowVAE — 2007.10930v2

Theorem 1 on page 4 identifies individual latent coordinates up to a **global permutation and sign flips**, under a specific generative model. It assumes injective image generation, independent standard-normal starting factors, independent generalized-Laplace temporal increments with shape below 2, matched transition parameters, and exact observation-pair density matching. The proof on pages 18–20 uses the normal marginal and non-Gaussian temporal metric to exclude other coordinate changes.

Here, sparsity concerns **latent displacement over time**. It is different from sparsity of graph edges, parameter-to-state dependencies, or prediction errors. The practical SlowVAE objective, page 5, reconstructs both frames and fits a temporal prior through a weighted KL. Its usual weighting is a practical surrogate, not the theorem's exact population density-matching condition.

The image results are useful evidence that temporal statistics can expose position. However, the KITTI experiment on page 8 uses presegmented pedestrian masks and evaluates position and scale; it does not establish end-to-end discovery of multiple interacting objects from unsegmented video. Appendix B, page 23, also demonstrates that a factor can collapse while a VAE objective improves. Identifiability of the model family is not a guarantee of successful optimization.

Our bouncing positions and velocities are not independent Laplace random walks. Velocity changes can be temporally sparse, but paired collision updates are coupled. A displacement penalty alone could instead reward static or constant representations.

### Content/style identifiability from augmentations — 2106.04619v4

Definition 4.1 on page 5 identifies the **invariant content block**, up to an arbitrary invertible transformation of that whole block. It does not identify individual coordinates within the block, all time-varying factors, or object masks.

Theorems 4.2–4.4, pages 6–7, assume invertible smooth image generation, suitable continuous densities, the correct content dimension, and paired views that preserve content while varying every style coordinate with sufficiently rich local support. They give three routes: exact generative pair likelihood, an invertible encoder with alignment, or alignment plus differential-entropy maximization in a bounded output space.

The third route is an important **decoder-free identification result**. Reconstruction is not necessary in principle. However, its information-preserving condition cannot simply be replaced by positive variance or EMA without an additional argument. Page 10 leaves the relationship between BYOL/SimSiam mechanisms and entropy maximization open.

Using neighboring frames as invariant views would encourage invariance to movement, including the positions we want to recover. A more appropriate proposed application would pair appearance variants of the same frame or history while preserving geometry. Even if its assumptions held, this would identify a content block; an additional principle would still be needed for one object per slot or individual physical coordinates. The paper's crop experiments also show why transformations must preserve the particular information we want.

### CITRIS — 2202.03169v3

CITRIS observes consecutive images **and known binary intervention targets**. It does not need numerical state labels or intervention magnitudes, but it does require consistent target identities and sufficiently distinguishable intervention patterns. Natural, endogenous collisions are not automatically interventions of this kind.

Its temporal prior restricts each next-state block to depend on the full previous state and only its own intervention indicator. Theorem 3.3 on page 5 combines observation likelihood with a criterion assigning intervention-invariant information to a residual block. It identifies **minimal intervention-sensitive causal blocks**, up to invertible transformations within each block. It assumes the relevant temporal causal model, independent innovations, sufficient observation information, and adequate intervention variation. Original CITRIS excludes instantaneous causal edges.

The word “minimal” matters. The ball-in-box example accompanying Definition 3.2, page 4, shows that an intervention changing box membership without changing within-box position need not identify the full position variable. What becomes identifiable depends on what the interventions actually distinguish. One ball, one causal mechanism, and one scalar coordinate are not necessarily the same grouping.

The practical models use reconstructive VAEs or a pretrained autoencoder followed by an invertible flow, with a structured temporal density. An invertible flow cannot restore information its pretrained encoder has discarded. Evaluation on page 7 uses nonlinear readouts of individual blocks and separation from other factors.

A relevant benchmark detail: Interventional Pong supplies ball velocity through a **fourth image channel** (Figure 6, page 7; explicit Figure 16 caption, page 29). It is not evidence that the method recovers hidden velocity from ordinary RGB frames alone.

### iCITRIS — 2206.06169v2

iCITRIS extends the framework to acyclic instantaneous effects as well as temporal effects. Definition 3.3 and Theorem 3.4, page 5, identify minimal blocks up to within-block invertible maps and their causal graph. The objective adds an instantaneous graph and a preference for fewer instantaneous edges among otherwise qualifying solutions.

The stronger guarantee requires **partially-perfect interventions**: an intervention removes instantaneous-parent influences but may retain past-state dependence. Generic soft interventions that retain instantaneous parents do not suffice. Appendix D, pages 19–25, adds material conditions including single-observation recoverability, suitable support, faithfulness, absence of unmodelled causal confounding, and symmetry breaking. Merely labeling perturbations is not enough to inherit the theorem.

The paper is unusually helpful about the gap between theorem and application: its Causal Pinball discussion on page 9 explicitly places deterministic physics outside several assumptions, while still testing the method empirically. Figure 15, page 38, again supplies velocity in a fourth image channel. This is a useful precedent for an empirical physics experiment, with the guarantee stated separately.

### Temporally Disentangled Representation Learning / TDRL — 2210.13647v1

This is the closest supplied starting point for **observational recovery of changing causal coordinates**. Definition 1, page 4, targets a global permutation and **componentwise invertible transformations**. The latent states can be statistically dependent and causally interact through the past. The required independence is between their innovations, conditional on the causal history.

Theorem 1, page 5, assumes invertible observation/encoding maps, smooth nondegenerate transition densities, conditional independence of the inferred next-state coordinates, and sufficient variation in derivatives of those conditional densities across histories. The rank condition concerns functions of the history, not an impossible collection of 2n independent finite vectors at one fixed history. Its proof on pages 14–15 forces the transformation Jacobian to have one nonzero entry in each row.

Temporal dependence alone is insufficient. Proposition 1, page 5, exhibits a remaining rotational ambiguity for stationary additive Gaussian innovations, even with a nonlinear transition mean. Particular heteroscedastic or non-Gaussian mechanisms with adequate variation can resolve this. Theorem 2, page 6, also uses changes of mechanism across **observed domain/context indices**. The type of shift can be unknown; the domain labels are supplied.

The implementation on pages 7–8 is a sequential VAE with observation reconstruction and conditional transition-flow priors. It models innovation distributions and includes the relevant density Jacobians. Its optional LassoNet graph analysis is performed after representation learning; graph sparsity is not the state-identification mechanism in the theorem.

The image Cartpole experiment, pages 8–9, recovering visible cart position and pole angle across gravity/action contexts is a particularly relevant empirical example. However, the paper's per-frame inference argument explicitly assumes that an image retains the current state information. Its multi-lag extension does not automatically resolve hidden velocity or hidden parameters in our observations.

### IDEA — 2402.12767v1

IDEA addresses latent environments and nonstationary forecasting. Theorem 3.1, page 3, identifies the latent-environment joint distribution up to label permutation under assumptions including a known number of environments, a full-rank Markov transition matrix, distinguishable emission measures, and sufficient regime duration. This does not guarantee exact recovery of every realized environment label.

Theorems 3.2–3.3, page 4, establish **subspace identification**: separating stationary and environment-dependent blocks. They do not require individual coordinates within those blocks to be recovered without mixing.

The structural model is narrower than arbitrary changing physics. The stationary block depends on its own past and innovations; the environment-dependent block depends on the environment and innovations, with no past dependence in the stated model. Unknown mass changing an interacting ball's dynamics is not directly this factorization. TDRL's changing-mechanism setting is closer to that idea, although it uses given context indices.

“Stationary” does not mean that a state's numerical value is constant over a trajectory. Consequently, IDEA's stationary block should not be equated with our episode-constant mass parameter. Its experiments concern synthetic mixtures and numerical forecasting datasets, rather than multi-object visual recovery. The method includes both observation reconstruction and forecasting terms.

## Matching the current bounce setting

The simulator in `src/scjepa/data/bounce.py` is deterministic conditional on initial state and episode properties. Random initial conditions do not create fresh independent transition innovations. Conditioned on complete state and parameters, the next-state distribution is a point mass, outside the smooth nondegenerate transition-density assumptions above. This limits direct theorem transfer; it does not show that learning physical states empirically is impossible.

A single rendered frame contains positions as an unordered set, to rendering precision, but does not generally reveal velocity or mass. A history can supply information about those hidden quantities, subject to observability and tracking conditions. With identical glyphs, simulator row identities are not directly visible. Claims and evaluation must respect object permutation. Finite rasterization and overlapping objects further limit literal invertibility.

The current visual condition uses physical radii proportional to mass while drawing mass-independent, equal-size glyphs. Contact distance and wall geometry can therefore reveal mass information through trajectories. We should not claim an absolute mass-scale symmetry based on fixed-radius collision equations. The distinction between within-episode state and across-episode parameters remains essential.

Conditioning only on state while marginalizing over unobserved episode parameters can also couple transition uncertainty across components. Neither such uncertainty nor observation noise automatically supplies the independent physical innovations required by TDRL. Adding Gaussian image noise does not fix that issue; adding noise only to velocity can still leave the full position/velocity transition singular.

Three distinct kinds of structure should remain separate in our reasoning:

- Sparse temporal displacement, as in SlowVAE.
- Independent innovations conditional on history, as in TDRL.
- Sparse causal dependencies, including our parameter-to-object graph.

None can be substituted for another without a new argument. Interacting physical states themselves should not be forced to be marginally independent.

## Consequences for evaluation and the next experiment

Weak linear state probes do not by themselves refute recovery up to nonlinear invertible transformations. The existing weak linear decoding and localization metrics remain concerning, but they do not establish that the representation contains no physical information. Evaluate separate questions:

1. **Information retained:** held-out nonlinear readout of physical state from the complete frozen representation, examining both slots and the state head.
2. **Object separation:** held-out nonlinear readout from each matched slot/block, with cross-block comparisons and tracking/localization. Interpret leakage carefully: correlated physical variables can predict one another even without unique information being encoded.
3. **Coordinate separation:** one-dimensional nonlinear or rank-based readouts with permutation matching, where a componentwise theorem is the intended target. A flexible readout of the whole representation can hide entanglement.
4. **Spatial grounding:** masks, centroids, and persistent tracks as separate evidence of object-localized representations. Recoverable physical information does not imply a good segmentation mask.

The most useful research sequence is to diagnose these distinctions, select the desired equivalence class, and then test an actual identifying principle. For passive data, TDRL supplies the strongest starting point among these papers, with an information-preserving observation model and a conditionally factorized transition distribution. A small setting meeting its assumptions should precede claims about the full deterministic bounce task. If known controlled interventions are acceptable, CITRIS/iCITRIS provide a different, explicit source of identifying information.

If retaining a decoder-free objective is central, the content/style paper provides a constructive starting point through appropriately preserved views and an information criterion. Its content-block guarantee would still need to be strengthened or combined with another principle to obtain the object-wise state chart our parameter theorem assumes.

No supplied theorem favors removing an LSTM. Recurrent inference can help with hidden velocity and occlusion. The measured contraction at initialization motivates an optimization ablation, but is not evidence that the trained LSTM is the cause of weak object learning. Objective/data identifiability is the more fundamental issue, and the architecture can initially remain fixed while that issue is isolated.

## Sources

- [Taming VAEs, 1810.00597v1](</Users/hoekstra/Desktop/slots forming/1810.00597v1.pdf>).
- [Temporal sparse coding / SlowVAE, 2007.10930v2](</Users/hoekstra/Desktop/slots forming/2007.10930v2.pdf>).
- [Content/style identifiability, 2106.04619v4](</Users/hoekstra/Desktop/slots forming/2106.04619v4.pdf>).
- [CITRIS, 2202.03169v3](</Users/hoekstra/Desktop/slots forming/2202.03169v3.pdf>).
- [iCITRIS, 2206.06169v2](</Users/hoekstra/Desktop/slots forming/2206.06169v2.pdf>).
- [TDRL, 2210.13647v1](</Users/hoekstra/Desktop/slots forming/2210.13647v1.pdf>).
- [IDEA, 2402.12767v1](</Users/hoekstra/Desktop/slots forming/2402.12767v1.pdf>).
- [Current SCJEPA manuscript](../../sources/SCJEPA.pdf), particularly PDF pages 8–9 and 13.
- [Bounce simulator and renderer](../../src/scjepa/data/bounce.py).
- [Existing slot-initialization audit](2026-09-06-exp2-slot-initialization.md).
