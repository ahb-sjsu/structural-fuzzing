# Chapter {{ch:case-study-aesthetic-judgment}}: Case Study --- Aesthetic Judgment from Embedding Geometry

*Geometric Methods in Computational Modeling* --- Andrew H. Bond

> *"It is the mark of an educated mind to be able to entertain a thought without accepting it."*
> --- attributed to Aristotle

The preceding two case studies applied geometric methods to domains where the objects of study --- software modules, whale clicks --- have an obvious, legible internal structure. A module has metrics. A coda has clicks and spectra. This chapter addresses a harder question: can geometric methods extract signal from a quantity that humans themselves struggle to define? Specifically, we ask whether the *shape* of a text or music embedding trajectory predicts how much humans like the work.

This is not a casual framing. "Aesthetic rating" is usually treated as the domain of reception studies, literary criticism, and music psychology --- fields that are resistant to formalization for good reason. The signal is noisy, confounded with genre, author reputation, marketing, and cohort effects. Any claim that geometric structure predicts aesthetic rating deserves skepticism, and this chapter spends significant space showing *where our first claims were wrong* and how we caught them. The final results are modest but robust: genre-residualized Pearson R around 0.09 on books (6.5 sigma), 0.18 on music (28 sigma), and cross-lingual Spearman correlations near 0.70 on divergence features computed in 19 languages from 10 language families.

For the book's thesis, these numbers matter less than the methodology. The pipeline is a clean worked example of composing the tools from Chapters {{ch:mahalanobis-distance}}--5 on a domain where the "right features" were not known in advance, and the diagnostics from Chapters {{ch:adversarial-robustness}}--10 were indispensable for distinguishing real signal from confound.

---

## {{ch:case-study-aesthetic-judgment}}.1 The Problem and the Claim

The operational question: given a text (a book) or audio clip (a track) encoded by a modern neural encoder, can we predict its aesthetic rating --- Goodreads stars for books, `log(1 + listens)` for Free Music Archive tracks --- using only the *geometric structure* of the encoder's output embeddings? No fine-tuning, no supervised head on the encoder itself, just statistics computed on the sequence of embedding vectors produced as we slide the encoder across the work.

There are three reasons to care about this question beyond the domain itself:

1. **Encoder evaluation.** If geometric structure of embeddings carries aesthetic signal, then aesthetic rating becomes a cheap external probe for embedding quality, complementary to intrinsic probes like STS or MTEB.
2. **Confound diagnosis.** Aesthetic rating is heavily confounded with genre and author/artist identity. A pipeline that appears to predict aesthetic rating in-sample but fails after residualizing on genre has learned the confound, not the signal. This is a canonical structural-fuzzing diagnostic (Chapter {{ch:subset-enumeration}}).
3. **Cross-modal generalization.** A feature that works on text should, in principle, work on music --- both are sequences of embeddings on a manifold. If it doesn't, or if the sign flips, we have learned something about the geometry of the two modalities that we could not have learned from either alone.

The claim at the end of this chapter is narrower than "embedding geometry predicts aesthetic rating." It is: *after controlling for genre, embedding-geometry channels explain a small but reproducible fraction of aesthetic variance in both text and music, the same channels correlate with each other across 19 languages, and the sign of the signal flips between modalities in ways that reveal structural differences between text and music encoders.*

---

## {{ch:case-study-aesthetic-judgment}}.2 Data Pipeline

### {{ch:case-study-aesthetic-judgment}}.2.1 Books: Gutenberg --- Goodreads Matching

Project Gutenberg provides roughly 70,000 full-text public-domain books. Goodreads provides user ratings for several million titles. Matching the two is not trivial --- Gutenberg titles are often in archaic or variant spellings (*The Adventures of Huckleberry Finn* vs *Huckleberry Finn*), author name formats differ, and Gutenberg includes many minor works without Goodreads ratings.

We used a two-stage fuzzy-match pipeline: (1) normalize titles and authors (lowercase, strip punctuation, remove "The/A/An"), (2) join on `(norm_author, norm_title)` and accept matches with Levenshtein ratio >= 0.90. After manual spot-checking of 200 random matches (99% precision on the spot check), this yielded 5,016 books with both a full text and a Goodreads average rating. Ratings are continuous in the range 2.5--4.8 with a long left tail.

The split is **author-disjoint 5-fold cross-validation**. This is the load-bearing CV choice. A random book-level split leaks heavily because Dickens's twelve books in the dataset share vocabulary, style, and audience; training on nine of them and testing on three gives inflated R simply because the model has memorized "Dickens = 4.1". Author-disjoint CV forces every evaluation fold to be held-out authors --- if Dickens is in the training set, none of his books are in the test set. On this dataset it reduces apparent R by about 40% versus random CV.

### {{ch:case-study-aesthetic-judgment}}.2.2 Music: FMA Medium + MERT-v1-330M

Free Music Archive Medium is 25,000 30-second mp3 clips with metadata including genre, listens, and occasionally Echonest audio features (the eight Spotify-style hand features: danceability, energy, valence, etc., for a subset of tracks). We used `log(1 + listens)` as the target, following the convention in recommendation-system evaluation to compress the heavy tail.

The encoder is **MERT-v1-330M** from m-a-p, a 330M-parameter masked-audio transformer that produces 1024-dimensional embeddings at roughly 75 Hz. We took one embedding per 200 ms, giving ~150 vectors per 30 s clip. Preprocessing: resample to 24 kHz (MERT's native rate), layer-13 mean pooling over the 75 Hz stream (following the paper's recommendation for genre-adjacent tasks), no further normalization.

CV is **artist-disjoint 5-fold**. Same logic as author-disjoint for books: an artist releases several tracks with similar listen counts, so random CV leaks artist identity. Artist-disjoint matters: on the FMA subset we tested, moving from track-disjoint to artist-disjoint CV dropped R from 0.38 to 0.30 on the same features.

### {{ch:case-study-aesthetic-judgment}}.2.3 Text Encoder: LaBSE

For text we used **LaBSE** (Language-agnostic BERT Sentence Embedding), 768-dim, trained on 109 languages with a translation-pair objective. The alternative candidates were SBERT-multilingual and XLM-R-base. We picked LaBSE for two reasons: (1) its translation-aligned training means that the same sentence in English and Finnish projects to approximately the same vector, which is essential for the cross-lingual experiment in Section {{ch:case-study-aesthetic-judgment}}.5; (2) it's small enough to embed 5,000 full books in a few hours on a single GPU.

**Tokenization at the paragraph level.** We split each book into paragraphs (double-newline delimited, median ~3 sentences), then embedded each paragraph as a single 768-d vector. The output is a sequence of paragraph embeddings per book. Median sequence length: 380 paragraphs; 10th/90th percentiles: 45 / 1,200. Books shorter than 30 paragraphs were dropped (18 books, mostly short stories).

The 45-token lower cutoff later turned out to matter: see Section {{ch:case-study-aesthetic-judgment}}.7 on the Hellinger saturation bug.

---

## {{ch:case-study-aesthetic-judgment}}.3 Feature Engineering: Four Channels

Given a sequence of embedding vectors $X = [x_1, x_2, \ldots, x_n] \in \mathbb{R}^{n \times d}$ for one work, we compute four families of geometric features. The design choices here mirror the SPD / TDA / hyperbolic decomposition from Chapter {{ch:case-study-bioacoustics}}: each channel captures a *different* kind of structure, and we expect each to contribute independently.

### {{ch:case-study-aesthetic-judgment}}.3.1 Channel A: Corpus-Gaussian Divergences

Fit a corpus-level Gaussian $\mathcal{N}(\mu_C, \Sigma_C)$ on the pooled embeddings from the training corpus (all books in the training fold, concatenated). For each work, fit a per-work Gaussian $\mathcal{N}(\mu_W, \Sigma_W)$ on its own embeddings. Then compute the closed-form divergences between these two Gaussians. This channel captures "how unusual is this work's embedding distribution versus the corpus as a whole?"

The closed-form for two Gaussians $P = \mathcal{N}(\mu_1, \Sigma_1)$, $Q = \mathcal{N}(\mu_2, \Sigma_2)$ in $\mathbb{R}^d$:

$$D_{KL}(P \parallel Q) = \tfrac{1}{2}\left[\text{tr}(\Sigma_2^{-1}\Sigma_1) + (\mu_2 - \mu_1)^\top \Sigma_2^{-1} (\mu_2 - \mu_1) - d + \log\frac{\det \Sigma_2}{\det \Sigma_1}\right]$$

The Bhattacharyya distance, which is better conditioned when the Gaussians have very different covariance scales:

$$B(P, Q) = \tfrac{1}{8}(\mu_1 - \mu_2)^\top \bar{\Sigma}^{-1} (\mu_1 - \mu_2) + \tfrac{1}{2}\log\frac{\det \bar{\Sigma}}{\sqrt{\det \Sigma_1 \det \Sigma_2}}, \quad \bar{\Sigma} = \tfrac{1}{2}(\Sigma_1 + \Sigma_2)$$

From $B$ we derive Hellinger $H = \sqrt{1 - e^{-B}}$ and, separately, Jensen-Shannon (Monte-Carlo), squared Mahalanobis $(\mu_W - \mu_C)^\top \Sigma_C^{-1} (\mu_W - \mu_C)$ (Chapter {{ch:mahalanobis-distance}}), and a Frobenius log-covariance distance $\|\log \Sigma_W - \log \Sigma_C\|_F$ (Chapter {{ch:spd-manifolds}}).

To compute these reliably in 768 dimensions with only 380 tokens per book, we first project to a 128-dim PCA basis fit on the training corpus. This is the single most important numerical-stability decision in the pipeline; Section {{ch:case-study-aesthetic-judgment}}.7 documents what happens when you skip it.

```python
def channel_A_divergences(X_proj, mu_C, Sigma_C, Sigma_C_inv, logdet_C):
    """Gaussian divergences between a work and the training corpus."""
    mu_W = X_proj.mean(axis=0)
    Sigma_W = np.cov(X_proj, rowvar=False) + 1e-4 * np.eye(X_proj.shape[1])
    delta = mu_W - mu_C
    d = len(mu_W)
    # Mahalanobis (Chapter {{ch:mahalanobis-distance}})
    mahal = delta @ Sigma_C_inv @ delta
    # KL (closed form)
    sign, logdet_W = np.linalg.slogdet(Sigma_W)
    kl = 0.5 * (np.trace(Sigma_C_inv @ Sigma_W) + mahal - d + logdet_C - logdet_W)
    # Bhattacharyya + Hellinger
    Sigma_avg = 0.5 * (Sigma_W + Sigma_C)
    _, logdet_avg = np.linalg.slogdet(Sigma_avg)
    bhat = 0.125 * delta @ np.linalg.solve(Sigma_avg, delta) \
         + 0.5 * (logdet_avg - 0.5 * (logdet_W + logdet_C))
    hell = np.sqrt(max(0.0, 1.0 - np.exp(-bhat)))
    # Log-Euclidean Frobenius (Chapter {{ch:spd-manifolds}})
    log_W = scipy.linalg.logm(Sigma_W)
    log_C = scipy.linalg.logm(Sigma_C)
    frob_le = np.linalg.norm(log_W - log_C, "fro")
    return dict(mahal=mahal, kl=kl, bhat=bhat, hell=hell, frob_le=frob_le)
```

### {{ch:case-study-aesthetic-judgment}}.3.2 Channel B: Internal Pair Similarity

Sample pairs of embedding vectors from the same work and compute cosine similarity statistics. Mean pair similarity captures internal thematic cohesion; its variance captures how *evenly* cohesive the work is.

```python
def channel_B_cohesion(X_proj, n_pairs=5000, rng=None):
    rng = rng or np.random.default_rng(0)
    n = X_proj.shape[0]
    i, j = rng.integers(0, n, size=(2, n_pairs))
    mask = i != j
    Xn = X_proj / np.linalg.norm(X_proj, axis=1, keepdims=True)
    sims = np.einsum("ij,ij->i", Xn[i[mask]], Xn[j[mask]])
    return dict(pair_sim_mean=sims.mean(), pair_sim_std=sims.std())
```

This is the cheapest channel and turned out to be the most interpretable: `pair_sim_mean` has the largest univariate effect in books (8.4 sigma) and, crucially, **flips sign on music** (discussed in Section {{ch:case-study-aesthetic-judgment}}.8).

### {{ch:case-study-aesthetic-judgment}}.3.3 Channel C: Trajectory Geometry

Treat $X$ as a trajectory in $\mathbb{R}^d$ indexed by reading/listening time. Step statistics, recurrence (how often does the trajectory return to a region it previously visited), autocorrelation at various lags, and discrete curvature (angle between consecutive step vectors). This is the channel most analogous to the spectral-trajectory analysis in Chapter {{ch:case-study-bioacoustics}}; we are reading the work as a path on the embedding manifold.

```python
def channel_C_trajectory(X_proj, lags=(1, 4, 16, 64)):
    steps = np.diff(X_proj, axis=0)
    step_len = np.linalg.norm(steps, axis=1)
    # Curvature: angle between consecutive steps
    s_n = steps / (np.linalg.norm(steps, axis=1, keepdims=True) + 1e-10)
    cos_turn = np.einsum("ij,ij->i", s_n[:-1], s_n[1:])
    feats = dict(step_mean=step_len.mean(), step_std=step_len.std(),
                 curvature_mean=np.arccos(np.clip(cos_turn, -1, 1)).mean())
    # Recurrence: fraction of embedding pairs within threshold
    D = scipy.spatial.distance.squareform(
            scipy.spatial.distance.pdist(X_proj, "euclidean"))
    thr = np.median(D[np.triu_indices_from(D, k=1)]) * 0.25
    feats["recurrence"] = float((D < thr).mean())
    # Autocorrelation at several lags (centered cosine)
    Xc = X_proj - X_proj.mean(axis=0)
    for L in lags:
        if Xc.shape[0] > L:
            a = np.einsum("ij,ij->i", Xc[:-L], Xc[L:])
            feats[f"autocorr_{L}"] = a.mean() / (np.var(Xc) + 1e-10)
    return feats
```

### {{ch:case-study-aesthetic-judgment}}.3.4 Channel D: Lasso on the 128-Dim PCA Spectrum

The previous three channels are hand-designed. Channel D lets the data pick: mean-pool each work's embedding sequence into a single 128-d vector (after the same PCA projection), and regress the target on that vector using Lasso with group-aware CV. This is a sanity check --- if hand features beat Lasso, we have chosen good features; if Lasso beats hand features, the hand features are leaving information on the table.

In practice Lasso and hand features contribute roughly equally in books, and Lasso dominates in music (where we had less musicological prior for the hand features).

The final feature vector per work is the concatenation of all four channels: ~20 hand features from A+B+C, plus up to 128 dims selected by Lasso in D. These feed a standard Ridge regressor for the final rating prediction.

---

## {{ch:case-study-aesthetic-judgment}}.4 Books: Discovery and the Genre Confound

### {{ch:case-study-aesthetic-judgment}}.4.1 Phase 1 --- Discovery (n = 4,998)

With author-disjoint 5-fold CV and the full Ridge+Lasso ensemble on all four channels, we got:

$$R = 0.241, \quad R^2 = 0.058, \quad z = 17\sigma$$

Per-channel univariate correlations (out-of-fold, Fisher-z aggregated):

| Feature | Pearson R | Sigma |
|---------|----------:|------:|
| `pair_sim_mean` | +0.126 | 8.4 |
| `mahal` | +0.115 | 7.7 |
| `bhat` | +0.119 | 8.0 |
| `kl` | +0.108 | 7.2 |
| `step_mean` | -0.096 | 6.4 |
| `recurrence` | +0.083 | 5.5 |
| Lasso-128 | (71 nonzero dims, combined R=0.181) | --- |

The signs are interpretable: books whose embedding distribution is far from the corpus centroid (high Mahalanobis, KL, Bhattacharyya) and which are internally cohesive (high pair-sim, low step-mean) tend to be rated higher. This matches a naive-but-plausible reception hypothesis: readers prefer books that are distinctive and internally coherent.

This would be a nice story. It is also mostly wrong, because of the next phase.

### {{ch:case-study-aesthetic-judgment}}.4.2 Phase 2 --- Within-Genre Residualization

Goodreads attaches a shelf (genre) to every rated book. Genres have very different average ratings: *Classics* averages 3.9, *Romance* averages 4.1, *Philosophy* averages 4.0, *Young Adult* averages 4.2. Genres *also* have very different embedding-geometry statistics, because a philosophy treatise is lexically and structurally different from a romance novel.

If our features predict rating *because* they predict genre, and genre predicts rating, we have not discovered anything about aesthetic judgment. We have rediscovered that philosophy books have lower average ratings and longer sentences.

The diagnostic is **within-genre residualization**. Take the 12 genres with the largest counts. For each feature $f$ and each target $y$, fit $f \sim \text{genre}$ and $y \sim \text{genre}$, then compute correlations between the residuals. If the residualized correlation is zero, the original signal was entirely a genre confound. If it survives, there's genuine within-genre aesthetic signal.

After residualization:

$$R_{\text{resid}} = 0.093, \quad R^2 = 0.009, \quad z = 6.5\sigma$$

In other words, **85% of the observed R^2 was genre confound**. We still have statistically significant within-genre signal (6.5 sigma is not small at n ~ 5000), but the headline number dropped from "moderate" to "barely-detectable." A fiction-only restriction (n = 2,250) gave intra-genre R = 0.131, z = 6.2; non-fiction alone was null.

This is the single most important finding in the books pipeline, and it is a negative one: without the residualization control, we would have reported an effect almost four times the true size. The structural-fuzzing diagnostic (which dimension of the input space is carrying the signal?) is exactly the Chapter {{ch:subset-enumeration}} subset-enumeration idea, applied here not to features-of-a-model but to covariates-of-the-target.

**Always run residualization on every categorical metadata column before reporting predictive results on subjective targets.**

---

## {{ch:case-study-aesthetic-judgment}}.5 Cross-Lingual Invariance

If the signal we're capturing is real aesthetic structure rather than English-specific surface lexical statistics, it should transfer across languages. Because LaBSE is translation-aligned, we can compute the same four-channel features in any of the 19 non-English languages present in Gutenberg, and check whether the *ranking* of works by each feature correlates with the English ranking of the same works (or of comparable works by the same author).

Dataset: 4,683 non-English books across 19 languages spanning 10 language families (Germanic, Romance, Finno-Ugric, Slavic, Hellenic, Celtic, Indo-Iranian, Semitic, Turkic, Japonic). We projected each language's embeddings into the *English* PCA basis before computing divergences, so that "distance from the corpus" is measured in a shared geometric frame.

The key metric is the mean pairwise Spearman correlation of feature rankings across language pairs:

| Feature | Mean pairwise Spearman $\rho$ |
|---------|-----------------------------:|
| `pair_sim_mean` | 0.712 |
| `mahal` | 0.710 |
| `hellinger` | 0.675 |
| `kl` | 0.658 |
| `step_mean` | 0.441 |
| `recurrence` | 0.218 |

The top four features produce rankings that agree across arbitrary language pairs with Spearman $\rho$ around 0.70. The tightest pair was English-Finnish Hellinger, $\rho = 0.77$ at $n = 288$, $p = 8 \times 10^{-57}$. English-French was $\rho = 0.78$ at $n = 227$. Finnish and English are in entirely different language families (Finno-Ugric vs Germanic) with unrelated morphology; getting $\rho = 0.77$ on a rank statistic is strong evidence that the divergence and cohesion channels are measuring something genuinely language-invariant in the LaBSE embedding space.

A complementary analysis fit Ridge on English data and evaluated on the pooled non-English set. Transfer R = 0.07 at n = 940, p = 0.033. Smaller than in-language, but nonzero and in the right direction.

We attempted to add Chinese as a 20th language and discovered that Gutenberg's Chinese collection is almost entirely *classical Chinese originals* --- the Zuo Zhuan, Dream of the Red Chamber --- not translations of Western works. We had implicitly assumed parallel text coverage. Classical Chinese projected into an English-trained PCA basis produces essentially random geometry because the embedding distribution barely overlaps the training corpus. This is the "know your dataset before you design your pipeline" lesson and it cost us three days. A brief exploratory histogram of `mean Mahalanobis distance to English corpus, by language` would have caught it in an hour.

---

## {{ch:case-study-aesthetic-judgment}}.6 Music: FMA + MERT

The music pipeline is structurally identical to the book pipeline --- four channels, Lasso on PCA'd embeddings, artist-disjoint CV --- with different numerical scales. n = 24,801 tracks, target = log(1 + listens), encoder = MERT-v1-330M layer 13.

### {{ch:case-study-aesthetic-judgment}}.6.1 Raw and Residualized Results

| Configuration | R | sigma |
|---------------|--:|------:|
| Lasso-128 (no residualization) | 0.302 | 49.8 |
| Lasso-128 genre-residualized | 0.177 | 28.3 |
| Hand features only (residualized) | 0.098 | 15.4 |
| Within-genre, Rock (n = 7,088) | 0.139 | 11.7 |
| Within-genre, Classical (n = 1,413) | -0.013 | (null) |

The 91% drop in R^2 from genre residualization is even more dramatic than books. FMA genre is a very strong predictor of listens on its own (Hip-Hop and Electronic dominate the upper tail), so failing to control for it would have produced a headline R of 0.30 that was 90% attributable to "is this track hip-hop-like?"

The **Classical null** is instructive: classical-music listen counts are not a reliable proxy for aesthetic preference in the way that Rock listen counts are. Classical listeners on FMA are a small, heavily curated audience; their behavior does not match the broader listens distribution. This is a domain fact, not a pipeline bug, and it showed up as soon as we split by genre.

### {{ch:case-study-aesthetic-judgment}}.6.2 Head-to-Head with Spotify's Eight Hand Features

A subset of FMA Medium (5,233 tracks) has Echonest-derived features matching Spotify's public audio-features endpoint: danceability, energy, valence, tempo, acousticness, instrumentalness, liveness, speechiness. These were carefully engineered by audio engineers over several years. If our geometric-channel features beat them on the same tracks with the same target, that's nontrivial.

On the shared 5,233-track subset, genre-residualized, same CV:

| Feature set | R | n parameters |
|-------------|--:|-------------:|
| Spotify-8 hand features | 0.103 | 8 |
| MERT Lasso-128 | 0.151 - 0.225 (seed range) | ~80 nonzero |
| MERT hand features (our Channels A+B+C) | 0.168 | ~12 |

The MERT-Lasso range across five seeds was 0.151 to 0.225. A paired bootstrap on the same tracks gave $p = 0.001$ for MERT-Lasso > Spotify-8. Our hand features alone (no Lasso) also beat Spotify-8. The win is reproducible, but the win margin has real seed variance and should be reported as a range rather than a point estimate.

---

## {{ch:case-study-aesthetic-judgment}}.7 Pitfalls and Dead Ends

This section is tactical. Each item below cost us a day or more and would have been caught by a different diagnostic plot.

### {{ch:case-study-aesthetic-judgment}}.7.1 The Hellinger Saturation Bug

Symptom: Hellinger values for all books clumped at exactly 1.0, with `std < 1e-6` across the whole corpus. The feature showed R = 0 in CV --- a dead feature.

Diagnosis: Hellinger is computed as $H = \sqrt{1 - e^{-B}}$ where $B$ is Bhattacharyya. For a 128-dim Gaussian fit on $n = 45$ tokens (the shortest books in our dataset), the per-work covariance $\Sigma_W$ is rank-deficient and ill-conditioned after regularization. $B$ saturates in the range 50--500. Then $e^{-B} \approx 0$ underflows, and $H = \sqrt{1 - 0} = 1.0$ for every track.

The bug was invisible in the aggregate: mean Hellinger across the corpus was 0.9998, std was $3 \times 10^{-7}$. The per-track histogram was a single spike at 1.0. A one-line diagnostic --- `plt.hist(hellinger_per_book)` --- would have flagged it immediately.

Fixes (we used the second):

1. Increase the token-count floor so $n \gg d$. At $n = 500$ tokens in 128 dim, Hellinger had a healthy distribution with std $\approx 0.08$.
2. Drop Hellinger, use Bhattacharyya directly. $B$ has a well-behaved distribution in the saturated regime; it's just the exponentiation that breaks.
3. Reduce $d$ (PCA to 32 or 64) so the $n/d$ ratio is healthier. We ultimately used $d = 128$ because of cross-lingual considerations, and added the rule that any book with fewer than $3d$ paragraph embeddings gets flagged for review.

General rule: whenever a feature involves $e^{-x}$ or $\log(1 - p)$, plot its raw distribution before trusting any aggregate statistic.

### {{ch:case-study-aesthetic-judgment}}.7.2 Genre Confound

Covered at length in Section {{ch:case-study-aesthetic-judgment}}.4 and 21.6. Summary:

- Books: 85% of R^2 was genre confound. Residualized R dropped from 0.241 to 0.093.
- Music: 91% of R^2 was genre confound. Residualized R dropped from 0.302 to 0.177.

Protocol going forward: run `y ~ genre` and `f ~ genre` regressions *before* any predictive modeling, and report both raw and residualized R. If the raw R exceeds the residualized R by more than 2x, the genre confound is load-bearing and any story told from the raw R is wrong.

### {{ch:case-study-aesthetic-judgment}}.7.3 CV Grouping

Three grouping regimes give three different R values for the same features:

| CV regime | R (books) | R (music) |
|-----------|----------:|----------:|
| Random row-level | 0.38 | 0.41 |
| Track-disjoint (music only) | --- | 0.38 |
| Artist/author-disjoint | 0.241 | 0.302 |

Track-disjoint music CV is still leaky because an artist's multiple tracks travel together. Author-disjoint books CV and artist-disjoint music CV are the minimum defensible regime. Reporting random-CV numbers on recommendation-adjacent targets is a form of noble lie.

### {{ch:case-study-aesthetic-judgment}}.7.4 Chinese Corpus Mismatch

Described in Section {{ch:case-study-aesthetic-judgment}}.5. We assumed parallel translations; got classical originals. Cost: three days. Fix: always run a "distance to training corpus" sanity histogram for any new data source before adding it to a shared-frame analysis.

---

## {{ch:case-study-aesthetic-judgment}}.8 Cross-Modality Sign Flips

The most interesting finding was unplanned. After completing both pipelines we compared feature-level coefficients across modalities and found reproducible sign flips on the two most-predictive channels:

| Feature | Books (R) | Music (R) |
|---------|----------:|----------:|
| `pair_sim_mean` | +0.126 | -0.076 |
| `step_mean` | -0.096 | +0.071 |

Both differences are significant at $p < 10^{-28}$ on the combined sample. The sign flip is not noise and it is not a dataset artifact --- it reproduces across seeds and CV folds.

Interpretation: in books, high `pair_sim_mean` (internal cohesion: paragraphs are thematically similar to each other) correlates with higher rating. In music, high `pair_sim_mean` (internal cohesion: 200 ms windows within a 30 s clip are spectrally similar to each other) correlates with *lower* listens. The direction in music is consistent with "monotonous tracks get fewer listens"; in books, the direction is consistent with "focused, thematically coherent books are rated higher."

For `step_mean` the story is inverted: high step size in music means varied, dynamic tracks (positive signal), while high step size in books indicates topic churn --- essay collections, treatises that cover disparate subjects --- and correlates negatively with rating.

**Practical implication.** Do not copy a feature set from a text pipeline to an audio pipeline without re-validating every sign. The geometric operations (cohesion, step length, divergence from corpus) are *modality-agnostic* as formulas but *modality-specific* as aesthetic signals. Each modality has its own preferred "shape" and the map between geometry and preference inverts. This is an empirical claim; we do not have a theoretical derivation for why it should be so, but the effect is large and reproducible.

---

## {{ch:case-study-aesthetic-judgment}}.9 Reference Implementation

The full four-channel feature extractor, roughly 40 lines, suitable for dropping into an arbitrary sequence-of-embeddings regression task:

```python
import numpy as np
import scipy.linalg, scipy.spatial

def aesthetic_geometry_features(
    X,              # (n_tokens, d) embeddings for one work
    mu_C, Sigma_C,  # corpus Gaussian (from training fold)
    Sigma_C_inv, logdet_C,
    pca_basis=None, # optional (d, k) PCA projection
    n_pairs=5000, rng=None,
):
    """Compute the four-channel geometric feature vector for one work.
    Returns a flat dict of scalars. Requires n_tokens >= 3*k for stable Channel A."""
    rng = rng or np.random.default_rng(0)
    Xp = X @ pca_basis if pca_basis is not None else X
    d = Xp.shape[1]
    out = {}

    # A. Corpus-Gaussian divergences
    mu_W = Xp.mean(axis=0)
    Sigma_W = np.cov(Xp, rowvar=False) + 1e-4 * np.eye(d)
    delta = mu_W - mu_C
    out["mahal"] = float(delta @ Sigma_C_inv @ delta)
    _, logdet_W = np.linalg.slogdet(Sigma_W)
    out["kl"] = 0.5 * (np.trace(Sigma_C_inv @ Sigma_W) + out["mahal"]
                       - d + logdet_C - logdet_W)
    Savg = 0.5 * (Sigma_W + Sigma_C); _, logdet_avg = np.linalg.slogdet(Savg)
    out["bhat"] = 0.125 * float(delta @ np.linalg.solve(Savg, delta)) \
                + 0.5 * (logdet_avg - 0.5 * (logdet_W + logdet_C))
    # NOTE: avoid Hellinger unless n_tokens >> d (see Section {{ch:case-study-aesthetic-judgment}}.7.1)
    out["frob_le"] = float(np.linalg.norm(
        scipy.linalg.logm(Sigma_W) - scipy.linalg.logm(Sigma_C), "fro"))

    # B. Internal pair similarity
    Xn = Xp / (np.linalg.norm(Xp, axis=1, keepdims=True) + 1e-10)
    i, j = rng.integers(0, Xp.shape[0], size=(2, n_pairs))
    m = i != j
    sims = np.einsum("ij,ij->i", Xn[i[m]], Xn[j[m]])
    out["pair_sim_mean"] = float(sims.mean()); out["pair_sim_std"] = float(sims.std())

    # C. Trajectory geometry
    steps = np.diff(Xp, axis=0)
    sl = np.linalg.norm(steps, axis=1)
    out["step_mean"] = float(sl.mean()); out["step_std"] = float(sl.std())
    sn = steps / (np.linalg.norm(steps, axis=1, keepdims=True) + 1e-10)
    out["curvature_mean"] = float(np.arccos(np.clip(
        np.einsum("ij,ij->i", sn[:-1], sn[1:]), -1, 1)).mean())
    D = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(Xp))
    thr = np.median(D[np.triu_indices_from(D, k=1)]) * 0.25
    out["recurrence"] = float((D < thr).mean())

    # D. Mean-pooled vector (input to downstream Lasso across the full corpus)
    out["_pooled"] = Xp.mean(axis=0)
    return out
```

To use on a new dataset: fit PCA on the training fold, compute $(\mu_C, \Sigma_C)$ from the PCA-projected pooled training embeddings, call the function per work, then Ridge on the scalar features and Lasso on the stacked `_pooled` vectors. Total cost ~40 lines for the extractor, ~30 lines for the regression glue, ~10 lines per diagnostic plot.

---

## {{ch:case-study-aesthetic-judgment}}.10 Lessons Learned

Tying back to the book's earlier chapters:

- **Mahalanobis vs Bhattacharyya (Chapter {{ch:mahalanobis-distance}}).** Mahalanobis is fine when the per-work and corpus covariances are comparable in scale. When they differ substantially --- short works versus the long-run corpus --- Bhattacharyya is better conditioned because it averages the covariances before inverting. In our books pipeline, Mahalanobis and Bhattacharyya were individually strong and near-collinear ($r = 0.81$); for cross-lingual transfer, Bhattacharyya generalized slightly better ($\rho = 0.68$ vs 0.62 mean pairwise). Default to Bhattacharyya when $n$ per work varies widely.

- **SPD-manifold awareness (Chapter {{ch:spd-manifolds}}).** Our Frobenius log-covariance feature was the log-Euclidean metric from Chapter {{ch:spd-manifolds}} applied as a scalar. It was weaker than the mean-shift divergences in this setting, but it was also the feature whose sign was *most stable* across books, music, and the non-English languages. When you need a covariance-based feature that is numerically robust and cross-domain transferable, log-Euclidean is a safer default than anything involving matrix inverses.

- **Hyperbolic embeddings (Chapter {{ch:hyperbolic-geometry}}).** We did not use them here. The natural place would be if the rating distribution had explicit hierarchical structure (genre-subgenre-sub-subgenre) and we wanted to embed the metadata graph rather than the works themselves. We considered it for the cross-lingual analysis (language family is a tree) and decided against because the Euclidean LaBSE projection was already giving $\rho = 0.7$; adding hyperbolic machinery to squeeze another 0.05 did not justify the complexity. For a problem where the hierarchy is deeper or more decisive --- taxonomy-based rating systems, citation graphs --- we would reach for Poincare embeddings first.

- **Subset enumeration and residualization (Chapter {{ch:subset-enumeration}}).** The genre-confound finding is the case-study version of the subset-enumeration diagnostic: we treated "genre" as a structural input dimension and measured how much of the apparent signal disappeared when that dimension was controlled. This is not fundamentally different from the OO-features-are-noise finding in Chapter {{ch:case-study-defect-prediction}}. The mechanism is identical: probe every structural dimension of the input space with a residualization test before reporting any predictive result.

- **Adversarial diagnostics (Chapter {{ch:adversarial-probing}}).** The Hellinger saturation bug was, in retrospect, an adversarial input the pipeline had generated for itself: the shortest books in the corpus produced an $n/d$ ratio that violated the feature's preconditions. An automated check --- "for every feature, compute its per-work distribution and flag features whose std is below threshold" --- is a cheap adversarial probe worth adding to any embedding-statistics pipeline.

The chapter's bottom line is the modest claim from the opening: embedding geometry carries a small, reproducible, cross-lingually consistent signal about aesthetic rating, once genre is controlled. The R^2 of 0.009 on within-genre books and 0.031 on genre-residualized music is not a recommendation system. It is evidence that the geometric structure of an encoder's output --- not the encoder's *content*, but its *shape* --- is a real, measurable, transferable property of a work, and that the tools in this book are sufficient to extract it.

---

## Exercises

**21.1.** Take any 500 books from Project Gutenberg with paired Goodreads ratings. Embed them with a multilingual sentence encoder of your choice. Compute the four-channel features using the reference implementation. Report both raw and genre-residualized R. How close do you come to the numbers in Section {{ch:case-study-aesthetic-judgment}}.4?

**21.2.** Implement the Hellinger-saturation diagnostic: plot the per-work histogram of every scalar feature, and flag any feature whose std is less than 1% of its mean. Which features in your extractor would this catch?

**21.3.** Repeat the cross-lingual analysis with a different sentence encoder (e.g., multilingual-e5). Does the mean pairwise Spearman $\rho$ across languages stay near 0.70, or does it drop? Interpret the difference in terms of the encoder's translation-alignment training objective.

**21.4.** On a music dataset of your choice with MERT or a similar audio encoder, verify the sign flip: does `pair_sim_mean` correlate negatively with your listens/popularity target, and `step_mean` positively? If not, what does the sign imply about your dataset's genre composition?

**21.5.** Construct a small tree-structured taxonomy for your domain (genre, subgenre, sub-subgenre) and embed it hyperbolically using a `PoincareBall` (Chapter {{ch:hyperbolic-geometry}}). Does using hyperbolic distance to the taxonomy improve residualization over one-hot genre indicators?

**21.6.** Run the pipeline with three CV regimes: random row-level, item-disjoint, and entity-disjoint (author/artist). Report R for each. What fraction of the gap between random-CV and entity-disjoint CV is the actual leak?

**21.7.** Replace the 128-dim PCA projection with a random projection of the same dimension. Does the divergence-channel signal survive? What does your answer tell you about how much of Channel A is truly about corpus-level structure versus artifacts of the PCA basis?

---

## Notes and References

Feng, F., Yang, Y., Cer, D., et al., "Language-agnostic BERT Sentence Embedding," *ACL* 2022 (LaBSE). Li, Y., Yuan, R., Zhang, G., et al., "MERT: Acoustic Music Understanding Model with Large-Scale Self-supervised Training," *ICLR* 2024. Defferrard, M., Benzi, K., Vandergheynst, P., et al., "FMA: A Dataset for Music Analysis," *ISMIR* 2017. Project Gutenberg ([gutenberg.org](https://www.gutenberg.org)) and Goodreads public rating aggregates. Bhattacharyya, A., "On a measure of divergence between two multinomial populations," *Sankhya* 7, 401--406 (1946). Hellinger, E., "Neue Begrundung der Theorie quadratischer Formen von unendlichvielen Veranderlichen," *J. Reine Angew. Math.* 136, 210--271 (1909). Arsigny, V., Fillard, P., Pennec, X., Ayache, N., "Log-Euclidean metrics for fast and simple calculus on diffusion tensors," *Magnetic Resonance in Medicine* 56(2), 2006. Tibshirani, R., "Regression shrinkage and selection via the Lasso," *JRSS B* 58(1), 267--288 (1996). Hofmann, T., Scholkopf, B., Smola, A., "Kernel methods in machine learning," *Annals of Statistics* 36(3), 1171--1220 (2008). The `turboquant-pro` experimental pipeline (Bond, 2026) contains the specific implementation used for the results in this chapter.
