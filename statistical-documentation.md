## Purpose

This document explains, purely from a statistical standpoint, how each of the four public functions in `ph_inequalities_statistical_comparison.py` derives (1) its point estimate, (2) its confidence interval, and (3) its determination of statistical significance against the reference (Overall) value. Each function is addressed in turn.

------------------------------------------------------------------------

## 1. `crude_proportion_df`

### Point estimate

Each row in the input data is treated as an independent Bernoulli trial — a binary 0/1 (or boolean) event indicator. Summed across all rows within a group, the total event count follows a Binomial(n, p) distribution, where n is the group's row count and p is the true, unknown probability of the event. The point estimate is the maximum likelihood estimator of p:

\[
\hat{p} = \frac{\text{events}}{n}
\]

No adjustment is made for underlying differences in age, sex, or other population structure between groups — every row contributes equally to the pooled count, regardless of which stratum it falls into. This is what makes the estimate "crude" rather than standardised.

### Confidence interval

The interval is constructed using the **Wilson score method**, a closed-form approximation derived from inverting the normal approximation to the binomial distribution around a shrinkage-adjusted centre rather than around \(\hat{p}\) itself:

\[
\text{center} = \frac{\hat{p} + \frac{z^2}{2n}}{1 + \frac{z^2}{n}}, \qquad
\text{half-width} = \frac{z\sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}
\]

where \(z\) is the standard normal critical value corresponding to the chosen confidence level. The lower and upper bounds are \(\text{center} \mp \text{half-width}\), clipped to \([0, 1]\). This re-centring is what allows the interval to remain a genuine, non-degenerate range even when \(\hat{p} = 0\) or \(\hat{p} = 1\) — the half-width does not collapse to zero at either boundary, because the \(z^2/(4n^2)\) term inside the square root survives even when \(\hat{p}(1-\hat{p}) = 0\). This behaviour is the specific reason Wilson score is used here rather than the simpler Wald interval, which is known to produce zero-width or out-of-range intervals at these boundaries.

### Statistical significance against the reference

No formal hypothesis test (no p-value, no test statistic) is computed. Instead, the Overall group's proportion is treated as a **fixed, known benchmark** rather than a random variable with its own sampling uncertainty, and each non-Overall group is classified purely by whether that fixed value falls inside or outside the group's own Wilson score interval:

- Reference value below the group's lower bound → the group's estimate is classified **"Higher"**.
- Reference value above the group's upper bound → the group's estimate is classified **"Lower"**.
- Reference value within \([\text{lower}, \text{upper}]\) → **"Not significant"**.

This is a confidence-interval-overlap heuristic, not a two-sample or one-sample hypothesis test — it implicitly answers "is the reference value a plausible value for this group's true proportion, given the precision of this group's own estimate?" rather than computing a probability of observing data this extreme under a null hypothesis.

------------------------------------------------------------------------

## 2. `crude_rate_df`

### Point estimate

Here the observed event count is modelled as a **Poisson**-distributed count rather than a bounded binomial count, because `event_col` is permitted to take any non-negative integer value per row (e.g. multiple events for a single patient), not just 0 or 1. The denominator is derived from row count — each row is treated as one unit of exposure — rather than supplied externally. The point estimate is:

\[
\hat{r} = \frac{\text{events}}{\text{denominator}} \times \text{multiplier}
\]

where the multiplier rescales the rate to a conventional reporting base (e.g. per 100,000).

### Confidence interval

The interval is built directly around the observed **count**, then divided through by the denominator and multiplier. Two different methods are used depending on the size of the count:

- For counts of 10 or more, **Byar's approximation** is used — a closed-form formula based on a cube-root normalising transformation of the Poisson count, chosen for its computational simplicity and accuracy at moderate-to-large counts.
- For counts below 10, an **exact chi-square-based Poisson interval** is used instead, exploiting the exact mathematical relationship between the Poisson distribution's cumulative probabilities and quantiles of the chi-square distribution (with degrees of freedom \(2 \times \text{count}\) or \(2 \times (\text{count}+1)\)). This exact method is substituted below the count-of-10 threshold because Byar's approximation becomes progressively less accurate as the count shrinks, particularly at zero.

### Statistical significance against the reference

As with `crude_proportion_df`, no formal test is performed. The Overall group's rate is treated as fixed, and each group's classification is again determined purely by CI overlap: **"Higher"** if the reference rate falls below the group's lower bound, **"Lower"** if it falls above the group's upper bound, and **"Not significant"** if the reference rate falls within the group's interval.

------------------------------------------------------------------------

## 3. `directly_standardized_proportion_df`

### Point estimate

This function still models each stratum's event count as Binomial(n_stratum, p_stratum), but instead of pooling all strata into a single crude proportion, it computes a proportion **separately within each stratum** (e.g. each age band) for the group, then combines those stratum proportions using **fixed weights derived from the full dataset's stratum composition** — not the group's own stratum sizes. This is the defining feature of direct standardisation: it removes the influence of the group having a different demographic mix than the reference population, isolating differences in stratum-specific rates from differences in demographic composition.

Where a stratum sits exactly at a 0% or 100% boundary (zero events, or all rows being events), a **Haldane-Anscombe continuity correction** is applied to that stratum before it enters the weighted sum — 0.5 is added to that stratum's event count and 1.0 to its row count. This prevents a boundary stratum from contributing exactly zero variance to the standardised calculation, which would otherwise understate the overall estimate's uncertainty regardless of how much data that stratum actually contained.

The point estimate is the reference-weighted sum of the (possibly Haldane-corrected) stratum proportions:

\[
\text{DSP} = \sum_i w_i \, p_i
\]

where \(w_i\) is stratum \(i\)'s reference weight and \(p_i\) is the group's proportion within stratum \(i\).

### Confidence interval

The interval is constructed using the **Wilson-Dobson** method, which proceeds in two stages:

1. **Compute the standardised variance** directly from the stratum-level binomial variances, weighted by the reference weights:
\[
\text{Var}(\text{DSP}) = \frac{\sum_i w_i^2 \, p_i(1-p_i)/n_i}{\left(\sum_i w_i\right)^2}
\]
2. **Rescale the crude proportion's Wilson score interval** to reflect this standardised variance, rather than deriving an entirely new interval formula. The crude (unweighted, unstandardised) proportion's own Wilson bounds \((\hat p_{\text{lo}}, \hat p_{\text{hi}})\) are computed first, then a scale factor \(\text{scale} = \sqrt{\text{Var}(\text{DSP}) / \text{Var}(\hat p_{\text{crude}})}\) is applied to shift those bounds around the DSP point estimate:
\[
\text{lower} = \text{DSP} + \text{scale} \times (\hat p_{\text{lo}} - \hat p_{\text{crude}}), \qquad
\text{upper} = \text{DSP} + \text{scale} \times (\hat p_{\text{hi}} - \hat p_{\text{crude}})
\]

This "borrow the shape, rescale the width" construction is deliberate: it inherits the Wilson interval's boundary-respecting behaviour (bounds remain within \([0,1]\)) while correctly reflecting the variance of the *standardised* estimate rather than the crude one, which a plain reuse of the crude Wilson interval would not do.

### Statistical significance against the reference

Once again, no formal hypothesis test is computed. The Overall row's **crude** proportion (not a standardised value — see below) is treated as the fixed reference, and each group's DSP is classified by whether that fixed value falls inside or outside the group's Wilson-Dobson interval, using the same three-way logic as the crude functions: **"Higher"**, **"Lower"**, or **"Not significant"**.

Note that the Overall row's own value is the crude proportion of the full dataset rather than a re-run of the standardisation logic, because the Overall row's population *is* the reference population used to build the standardisation weights in the first place — standardising the whole population against itself would simply return the same crude value.

------------------------------------------------------------------------

## 4. `directly_standardized_rate_df`

### Point estimate

This is the rate counterpart of DSP: each stratum's event count is modelled as **Poisson** rather than binomial, and stratum-level rates (events divided by the row-count-derived denominator within that stratum) are combined using the same fixed reference weights used in the proportion function. Where a stratum has zero events, a **Haldane-style correction** is applied (adding 0.5 to events and 1.0 to the denominator) — unlike DSP, this correction is only needed at the lower (zero-event) boundary, since a Poisson rate has no natural upper bound the way a proportion is capped at 1.

The point estimate is the reference-weighted average of the stratum rates:

\[
\text{DSR} = \frac{\sum_i w_i \, r_i}{\sum_i w_i} \times \text{multiplier}
\]

where \(r_i\) is the group's (possibly Haldane-corrected) rate within stratum \(i\).

### Confidence interval

The interval is constructed using the **Dobson-Byar** method, mirroring the two-stage logic used for DSP but adapted to Poisson counts:

1. **Compute the standardised variance** from the stratum-level Poisson variances:
\[
\text{Var}(\text{DSR}) = \frac{\sum_i w_i^2 \, O_i / n_i^2}{\left(\sum_i w_i\right)^2}
\]
where \(O_i\) is the stratum's event count and \(n_i\) its denominator.
2. **Rescale the crude event count's Byar or exact-chi-square interval** (chosen by the same count-of-10 threshold used in `crude_rate_df`) by the ratio of standardised-to-crude variance:
\[
\text{scale} = \sqrt{\text{Var}(\text{DSR}) / \text{crude events}}
\]
\[
\text{lower} = \text{DSR}_{\text{unscaled}} + \text{scale} \times (O_{\text{lo}} - \text{crude events}), \qquad
\text{upper} = \text{DSR}_{\text{unscaled}} + \text{scale} \times (O_{\text{hi}} - \text{crude events})
\]
before finally applying the multiplier to bring the bounds onto the reported rate scale.

This is structurally identical to the Wilson-Dobson approach for DSP, substituting the Poisson-appropriate crude interval (Byar/exact-chi-square) for the binomial-appropriate one (Wilson), for the same reason: it preserves the crude interval's guarantee of remaining non-negative while correctly reflecting the standardised estimate's own variance.

### Statistical significance against the reference

As with all three preceding functions, significance is established purely via confidence-interval overlap against a fixed reference value — here, the Overall row's crude rate (again, unstandardised, for the same reason given for DSP). No p-value or formal test statistic is computed; the classification is **"Higher"**, **"Lower"**, or **"Not significant"** depending on whether that fixed crude rate falls outside or within the group's Dobson-Byar interval.

------------------------------------------------------------------------

## Summary Table

| Function | Underlying distribution | Point estimate basis | CI method | Significance basis |
|---|---|---|---|---|
| `crude_proportion_df` | Binomial | Pooled events / n | Wilson score | CI-overlap vs. fixed Overall proportion |
| `crude_rate_df` | Poisson | Pooled events / row-count denominator | Byar (≥10) / exact chi-square (<10) | CI-overlap vs. fixed Overall rate |
| `directly_standardized_proportion_df` | Binomial per stratum | Reference-weighted sum of stratum proportions (Haldane-corrected at boundaries) | Wilson-Dobson (rescaled Wilson) | CI-overlap vs. fixed Overall (crude) proportion |
| `directly_standardized_rate_df` | Poisson per stratum | Reference-weighted average of stratum rates (Haldane-corrected at zero) | Dobson-Byar (rescaled Byar/exact) | CI-overlap vs. fixed Overall (crude) rate |

Across all four functions, the significance determination is uniform in design: none perform a formal hypothesis test, and all rely on treating the reference value as fixed and checking whether it falls inside or outside the group's own interval — the only thing that varies between functions is *which* distribution and *which* CI construction produces that interval in the first place.