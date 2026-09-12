# Ethics, responsible use, and limitations

## Framing

User-level toxicity prediction is framed here as a prevalence-aware benchmark for identifying users associated with toxic review behavior, not as a judgment of user character. The target label is derived from review-level toxicity annotations, and the goal is to estimate platform-level risk and support aggregate analysis or human-review triage, not to infer an intrinsic or context-free toxic identity. The predictor should be read as a triage instrument that prioritizes human attention at scale, and never as an automatic verdict on an individual.

## Data and consent

The data describe real users who did not consent to toxicity profiling. The publication export therefore excludes the raw corpus and identifiable per-user research outputs, including prediction files keyed by Steam identifiers and narrative records quoting review text. Those outputs remain in the research checkout and its existing history; they must not be mistaken for anonymous aggregates. See [artifact.md](artifact.md) for the public package and [data.md](data.md) for availability. Any re-collection should pseudonymize Steam identifiers and profile URLs and respect Steam's Terms of Service. The content warning in the README applies throughout: illustrative quotes contain offensive and hateful language.

## Limitations that bear on use

- **Tool-derived labels.** Toxicity is operationalized through two automatic detectors whose thresholds were calibrated on 400 human-labeled reviews, so the labels inherit both tools' error profiles. The characterization itself shows why this matters: performative profanity and mechanical violence vocabulary (for example "kill" in gameplay descriptions) are structural sources of false positives that calibration bounds but does not remove.
- **Label-definition circularity.** Content-based models read the same reviews that define the label. The leave-toxic-out control targets this directly and shows that a genuine diffuse signal survives, but most of the predictive performance remains circular, and the control is only defined for users with at least one non-toxic review.
- **Tool-induced circularity.** Because the labels are produced by automatic detectors, a predictor may partly learn to reproduce those tools' decisions rather than toxicity as judged by humans. The human annotation calibrates thresholds but is too small to serve as an independent user-level test set, so this is bounded and acknowledged rather than fully resolved.
- **Activity-volume sensitivity.** The count label is sensitive to how many reviews a user wrote, since more active users have more opportunities to produce at least one toxic review. The rate label mitigates this but does not eliminate it.
- **Scope.** The corpus is English-only, from a single collection window, and covers the 43.7 percent of users matchable to a public profile. Toxicity in other languages, in other Steam spaces such as discussions and chat, and among unmatched or private profiles is outside these claims. The protocol classifies held-out users within the window rather than forecasting future behavior.
- **LLM uncertainty.** The LLM classifier comparison rests on 1,000 users with only 40 positives, so its estimates carry wide uncertainty. All LLMs operate without game context, which prevents genre priors from anchoring judgments but leaves boundary cases that only game context could disambiguate. Inter-judge agreement in the narrative panel is weak, so LLM-based evaluation is diagnostic rather than ground truth.

## Deployment guidance

Any use of this work for moderation should keep a human in the loop, report the labels, circularity, and uncertainty as openly as the performance numbers, and treat a positive prediction as a prompt for review rather than as evidence of wrongdoing. The narrative explanation layer is intended precisely to make boundary cases auditable for that human review.
