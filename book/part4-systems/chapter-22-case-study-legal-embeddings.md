# Chapter 22: Case Study — Legal-Domain Embeddings and the Generalization Gap

> *"A model that beats the test it was trained for has told you nothing until it faces a test it was not."*

The case studies of Chapters 19–21 fuzzed the parameter and feature structure of models whose
*objective* was fixed. This chapter fuzzes the objective itself. We adapt a general-purpose sentence
encoder (LaBSE) to the legal domain and ask not "which features matter?" but "which *training form*
matters?"—and then we subject the winner to a probe it was never trained against. The exercise
surfaces two tools that extend the adversarial-probing framework of Chapter 10 from parameter and
signal space into **relation space**: the *cross-relation generalization gap* and *probe calibration*.
It closes with the *found-then-frozen* discipline that connects structural search to honest
out-of-sample validation.

The numbers are real. They are reported as they came, including a first attempt that made the model
worse.

---

## 22.1 The Setup: Structural Fuzzing of a Training Objective

The base model is LaBSE, a 471M-parameter multilingual sentence encoder. The target domain is U.S.
case law. The question is whether a light domain adaptation improves legal-text retrieval over the
base encoder, and if so, *which form of adaptation*.

This is structural fuzzing with the dimension being fuzzed set one level up from the usual: not a
feature group, but the **training objective**. Two candidate structures:

- **v1 — unsupervised SimCSE.** Each legal sentence is its own positive pair via two dropout views;
  other in-batch sentences are negatives. No labels. This is the cheapest possible adaptation.
- **v2 — citation-supervised (SPECTER-style).** Positive pairs are (citing opinion, cited opinion)
  drawn from the citation graph; the rest of the batch are negatives. The supervision signal is real
  legal relatedness rather than dropout noise.

The `evaluate_fn` of earlier chapters becomes a held-out retrieval score. The parameter being varied
is categorical—the objective family—so the "campaign" here is a two-point comparison rather than a
grid. But the discipline is identical to Chapter 8's Pareto reasoning: **selection is by held-out
performance, never by training loss.** Both candidates drive their training loss down; only one of
them generalizes.

### 22.1.1 The First Result Was a Regression

On a held-out, opinion-disjoint citation-retrieval probe (defined in §22.2), the two objectives
scored:

| objective | citation-retrieval AUROC | Δ vs. base LaBSE (95% CI) |
|-----------|:------------------------:|---------------------------|
| base LaBSE (no adaptation) | 0.765 | — |
| **v1 (unsupervised SimCSE)** | 0.340 | **−0.145 [−0.214, −0.081]** |
| **v2 (citation-supervised)** | 0.971 | **+0.206 [+0.190, +0.223]** |

v1 did not merely fail to help; it *significantly degraded* the base encoder. A light,
small-batch, unsupervised contrastive pass collapsed LaBSE's carefully tuned geometry—its training
loss fell while its held-out retrieval fell with it. This is the structural-fuzzing analogue of a
Chapter 9 fragility: a configuration that looks fine by its internal metric and breaks under the
held-out probe. Reporting it is not optional. The negative result is what makes the positive result
(v2) credible, and it is the single most useful data point for anyone tempted to reach for the cheap
objective first.

The lesson generalizes past embeddings: **when the structure you fuzz is the loss function, the
in-sample metric is not admissible evidence.** Only a probe the objective does not directly optimize
can rank the candidates.

---

## 22.2 Constructing the Probe — and the Calibration Trap

The held-out probe is a retrieval AUROC. Positive pairs are true (citing, cited) opinion pairs whose
opinions were held out of training by an opinion-level split (opinion id mod 10 == 7); negatives are
random cross-pairings. AUROC is the probability that a true pair scores above a random pair.

The first version of this probe returned a number that should have stopped the project:

> On the first probe, **base LaBSE scored 0.48—indistinguishable from chance (0.50).**

A 471M-parameter encoder trained on billions of sentences cannot tell a citation-linked pair of legal
opinions from a random pair? That is not a fact about the model. It is a fact about the **probe**.

### 22.2.1 The Intensity-Zero Identity, Applied to Evaluation

Chapter 10 §10.2 insisted that every parametric transform satisfy the intensity-zero identity:
$T(x,0)=x$, so that $\delta(0)=0$ is a *calibrated baseline*. The evaluation-construction analogue is:

> **Probe-calibration rule.** Before trusting a probe to rank models, verify it against references of
> known strength. A probe on which a *known-strong* reference scores at chance is measuring surface,
> not the target. A probe on which a *known-weak* reference scores well is leaking the answer.

Here the known-strong reference (base LaBSE) scored at chance, which localized the defect immediately.
The probe encoded each opinion from its first ~3000 characters—which, for a judicial opinion, is
almost entirely the **standardized caption**: "UNITED STATES COURT OF APPEALS FOR THE ... CIRCUIT ...
Before ... Circuit Judges." Every opinion's first 3000 characters look alike. The probe was measuring
caption boilerplate, a surface feature shared by *all* pairs, so it could not separate true pairs from
random ones. This is exactly Chapter 10's **flat profile** ("alarming for stress transforms—the model
is not reading the content being destroyed"), but occurring in the *measurement instrument* rather
than the model under test.

The fix was to skip the caption and encode the body. After recalibration, base LaBSE rose to 0.765—a
sensible number for a strong general encoder on a hard domain task—and the model comparison of §22.1
became trustworthy. **The comparison numbers are only as good as the probe, and the probe is only
trustworthy once its calibration references land where they should.**

### 22.2.2 Why This Belongs in a Fuzzing Text

A structural-fuzzing campaign is a machine for producing model rankings. If the `evaluate_fn` has a
latent confound—if it rewards a surface feature correlated with, but not identical to, the target—then
every downstream artifact (subset ranking, Pareto frontier, sensitivity order) inherits the confound
silently. The subset enumeration will happily report that "caption-length features" dominate, and it
will be *right about the probe and wrong about the world*. Calibrating the probe against strong and
weak references is the cheapest insurance in the entire pipeline, and it is almost always skipped.

---

## 22.3 The Cross-Relation Generalization Gap

v2 scores 0.971 on citation retrieval. But citation retrieval is precisely the relation v2 was trained
to encode. A model can reach 0.971 on its own training relation by learning that relation's surface
regularities without acquiring any transferable legal structure. To separate the two, we need a probe
on a relation the model **never trained on**.

### 22.3.1 An Orthogonal Relation

Legal opinions carry a second, structurally independent relation: **docket lineage**. A district-court
opinion and the appellate opinion that reviews it share a case—linked not by citation but by matching
*docket numbers* in the court's originating-case metadata. This relation is:

- **Structurally independent** of the training signal (docket-number matching, not citation edges);
- **Naturally held out**—the district opinions in this corpus predate the electronic era and fall
  below the id threshold used to sample training pairs, so they were never seen;
- **Semantically harder**—a district opinion argues the merits; its appellate reviewer argues legal
  error and standard of review. With party names stripped, the two are only weakly similar in the body.

Running the same probe machinery on 4,406 held-out lineage pairs:

| relation | base LaBSE | v2 (citation-supervised) | Δ (95% CI) |
|----------|:----------:|:------------------------:|------------|
| citation retrieval (**trained** relation) | 0.765 | 0.971 | **+0.206 [+0.190, +0.223]** |
| docket lineage (**independent** relation) | 0.545 | 0.562 | **+0.018 [+0.004, +0.031]** |

### 22.3.2 Reading the Gap

Define the **cross-relation generalization gap** as the difference between a model's improvement on
its trained relation and its improvement on an independent relation:

$$G = \Delta_{\text{trained}} - \Delta_{\text{independent}} = 0.206 - 0.018 = 0.188.$$

This is the relation-space analogue of Chapter 10's **sensitivity gap** (the ratio of stress-transform
to invariant-transform displacement). There, a large gap meant the model separated meaningful content
from surface variation. Here the interpretation is sharper and, deliberately, less flattering:

- A **large** $G$ (as here) means the adaptation is **mostly specialization**: it bought a great deal
  on the relation it optimized and a little elsewhere. The gain is real—both intervals exclude
  zero—but it is not a broad "the model now understands law" gain.
- A gap near **zero** with both improvements positive would indicate the adaptation captured
  *transferable* structure—an improvement that shows up on relations it never saw.
- Both improvements at **zero** would indicate no adaptation at all (or a probe confound of the §22.2
  kind, which is why probe calibration comes first).

The honest one-sentence summary that the gap licenses—and that the trained-relation number alone would
not—is: *citation supervision dramatically improves the relatedness it was trained on and transfers a
small, statistically significant amount to an independent legal relation.* Note that the independent
probe also disciplines the claim's language: without it, 0.971 invites the overclaim "a legal reasoning
model"; with it, the ceiling on transfer is measured, not assumed.

### 22.3.3 The Technique, Stated Generally

> **Cross-relation probing.** To distinguish learned structure from objective-memorization, evaluate an
> adapted model on at least one relation that is (a) structurally independent of the training signal
> and (b) held out at the *entity* level, not merely the *pair* level. Report the improvement on both
> the trained and the independent relation, and the gap between them. The trained-relation number sets
> the ceiling on enthusiasm; the independent-relation number sets the floor on the claim.

Entity-level holdout (b) is essential and easy to get wrong. Holding out individual *pairs* while the
*entities* recur in training leaks structure: the model has already seen the documents, only not this
particular link between them. The docket-lineage probe is clean because the district opinions are
entirely absent from training, not merely their lineage links.

---

## 22.4 Found, Then Frozen

Structural fuzzing is a search. Search over a large enough space will find *something*—a subset, a
threshold, an objective—that scores well on any fixed probe. Chapter 8 guarded against this with
Pareto parsimony and Chapter 9 with robustness; the strongest guard is temporal.

The pattern that carried both the embedding work and its companion preregistration is **found-then-frozen**:

1. **Search / fuzz** to *find* a candidate structure (here: citation supervision over SimCSE).
2. **Freeze** the structure and its *predictions*—derived from the frozen structure, not re-fit to the
   test data—under a content hash, before the confirmatory data is touched.
3. **Test out-of-sample** against the frozen predictions. A wrong prediction is a reported failure,
   never an edit.

The freezing step deserves emphasis because it is what converts a search winner into a claim. In the
companion economic-manifold study, the sign of every coordinate's predicted effect was *derived* from
the model's frozen cost convention—the same convention already fitted on the active coordinates—rather
than asserted by intuition, and the whole bundle (protocol, codebook, sign table, datasets, power
analysis) was committed under a SHA-256 hash and a cryptographically signed tag before the dormant
coordinates were tested. The signature and hash make the ordering—prediction before data—independently
verifiable rather than merely asserted.

For a structural-fuzzing practitioner the rule is compact:

> **Found-then-frozen.** The output of a fuzzing campaign is a *hypothesis*, not a result. Register the
> winning structure and its derived predictions—hashed and timestamped—before the confirmatory
> evaluation. What the campaign found on the search data earns the right to be *tested*, not the right
> to be *believed*.

This is the discipline that separates a search that discovers structure from a search that
manufactures it.

---

## 22.5 Summary and Forward Connections

This case study fuzzed a model's training objective rather than its features, and in doing so
exercised three tools that extend the framework of Part II:

1. **Objective-space fuzzing.** The structure under test can be the loss function itself. When it is,
   the in-sample metric is inadmissible; only a held-out probe can rank candidates. The cheap objective
   (unsupervised) degraded the base model; the supervised objective improved it. Both facts were
   reported.
2. **Probe calibration.** Before trusting a probe, verify it against known-strong and known-weak
   references—the evaluation-space form of Chapter 10's intensity-zero identity. A strong reference at
   chance revealed a caption-boilerplate confound that would otherwise have silently corrupted every
   downstream ranking.
3. **The cross-relation generalization gap.** Evaluating on an independent, entity-level-held-out
   relation, and reporting the gap between trained- and independent-relation improvement, separates
   learned structure from objective-memorization and calibrates the language of the claim.

And the connecting discipline, **found-then-frozen**, treats every campaign output as a hypothesis to
be registered and tested out-of-sample, not a result to be believed.

The unifying theme with Chapter 10 is unchanged: *the difference between what a probe expects and what
it receives encodes the structure of the system under test.* This chapter adds that the same logic
governs the probe itself (calibrate it) and the relation it measures (vary it). The next part of a
mature practice is to automate these checks into the campaign so that no ranking is emitted without a
calibrated probe and at least one orthogonal-relation gap alongside it.
