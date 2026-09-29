# Statistical methodology

## Scope

This document describes the implemented statistical approach for the four public functions:

- `crude_proportion_df()`;
- `crude_rate_df()`;
- `directly_standardized_proportion_df()`;
- `directly_standardized_rate_df()`.

The functions calculate estimates and two-sided confidence intervals for every generated grouping. The confidence level is controlled by `confidence` and defaults to 0.95. Low-count results are returned with quality notes; the package does not suppress them.

## Common reference comparison

Each function adds a `significance` label by comparing a result's confidence interval with the estimate for the overall row, where every inequality and organisational dimension equals `all_label` (default: `"All"`). The overall estimate is treated as a fixed benchmark:

- `Higher` when the reference estimate is below the result's lower confidence limit;
- `Lower` when the reference estimate is above the result's upper confidence limit;
- `Not significant` when the reference lies within the closed interval;
- `Not tested` when the reference or either confidence limit is non-finite;
- `Reference` for the overall row itself.

This is a descriptive confidence-interval classification, not a formal between-group hypothesis test. It does not propagate uncertainty in the overall estimate, account for dependence between a subgroup and an overall population that contains it, produce a p-value, or adjust for multiple comparisons.

## Crude proportion

### Point estimate

`crude_proportion_df()` requires a row-level binary event indicator. Boolean values are converted to integers; otherwise the event column must contain integers in `{0, 1}`.

For a result group with event count \(x\) and row count \(n\), the crude proportion is

\[
\hat{p} = \frac{x}{n}.
\]

Every row has equal weight. No standardisation is applied.

### Wilson score interval

Let \(C\) be the requested confidence level, \(\alpha = 1-C\), and \(z = \Phi^{-1}(1-\alpha/2)\). The Wilson centre and half-width are

\[
\text{centre} =
\frac{\hat{p} + z^2/(2n)}{1 + z^2/n},
\]

\[
\text{half-width} =
\frac{z\sqrt{\hat{p}(1-\hat{p})/n + z^2/(4n^2)}}
     {1 + z^2/n}.
\]

The interval is

\[
L = \max(0,\text{centre}-\text{half-width}), \qquad
U = \min(1,\text{centre}+\text{half-width}).
\]

The Wilson construction remains non-degenerate at observed proportions of 0 and 1 and keeps limits within `[0, 1]`.

## Crude rate

### Exposure and estimate

`crude_rate_df()` accepts a non-negative integer count per row. A row may contain more than one event. The function creates an internal exposure value of one for every row, so the group denominator \(D\) is the group's row count.

For total event count \(x\) and reporting multiplier \(M\),

\[
\hat{r} = \frac{x}{D}M.
\]

The default is \(M=100{,}000\). This is an end-of-period denominator: it assumes each row represents one person present at period end and does not account for partial-period person-time.

### Poisson count intervals

The function first calculates a confidence interval for the event count, then divides both limits by \(D\) and multiplies by \(M\).

For \(x<10\), it uses the exact chi-square Poisson interval:

\[
L_x =
\begin{cases}
0, & x=0,\\
\frac{1}{2}\chi^2_{\alpha/2,\,2x}, & x>0,
\end{cases}
\]

\[
U_x = \frac{1}{2}\chi^2_{1-\alpha/2,\,2(x+1)}.
\]

For \(x\geq10\), it uses Byar's approximation:

\[
L_x = x\left(1-\frac{1}{9x}-\frac{z}{3\sqrt{x}}\right)^3,
\]

\[
U_x = (x+1)\left(1-\frac{1}{9(x+1)}+
\frac{z}{3\sqrt{x+1}}\right)^3.
\]

The reported limits are

\[
L = \frac{L_x}{D}M, \qquad U = \frac{U_x}{D}M.
\]

An observed count of zero therefore produces a rate and lower limit of zero but a positive upper limit.

## Reference weights

Both directly standardised functions derive the standard population internally from the full input dataframe after input validation and any automatic numeric-stratum binning.

For joint stratum \(i\), let \(N_i^{\mathrm{ref}}\) be the number of rows in the full input and let \(N^{\mathrm{ref}}\) be the full input row count. The reference weight is

\[
w_i = \frac{N_i^{\mathrm{ref}}}{N^{\mathrm{ref}}}.
\]

With multiple `strata_cols`, weights are calculated for the observed combinations of all stratum columns. The package does not use an external standard population such as the European Standard Population.

Numeric stratum columns are automatically converted to `Q1`-`Q4` using the full column's 25th, 50th, and 75th percentiles with linear interpolation. Values equal to a cut point are assigned to the lower quartile. Ties can therefore result in fewer than four observed labels. Non-numeric strata are used as supplied.

## Missing strata

For a subgroup, a positively weighted reference stratum is considered missing when that subgroup has no rows in the stratum. Missing strata are omitted rather than assigned a zero event risk or rate. If \(O_g\) is the set of observed, positively weighted strata for group \(g\), the retained weights are renormalised:

\[
w_{ig}^{*} = \frac{w_i}{\sum_{j\in O_g} w_j},
\qquad i\in O_g.
\]

The `notes` column reports the number of omitted strata, their combined share of the reference population, and the retained coverage. This makes the result calculable but changes the effective standard population for that subgroup; comparisons should therefore be treated cautiously when omitted reference weight is material.

## Directly standardised proportion

### Point estimate

Within each observed stratum \(i\), the uncorrected observed proportion is

\[
\hat{p}_i = \frac{x_i}{n_i}.
\]

The directly standardised proportion (DSP) is

\[
\widehat{DSP} = \sum_{i\in O_g} w_{ig}^{*}\hat{p}_i.
\]

No Haldane-Anscombe or other continuity correction is applied to the event counts or denominators. Consequently, all-zero strata contribute a point estimate of zero and all-event strata contribute one.

### Wilson-MOVER interval

A Wilson interval \([L_i,U_i]\) is calculated separately for each observed stratum. The Method of Variance Estimates Recovery (MOVER) combines the distances between each stratum estimate and its Wilson limits:

\[
L_{DSP} = \max\left(
0,
\widehat{DSP} -
\sqrt{\sum_{i\in O_g}(w_{ig}^{*})^2(\hat{p}_i-L_i)^2}
\right),
\]

\[
U_{DSP} = \min\left(
1,
\widehat{DSP} +
\sqrt{\sum_{i\in O_g}(w_{ig}^{*})^2(U_i-\hat{p}_i)^2}
\right).
\]

The point estimate and limits are rounded to six decimal places in the public output.

### Overall reference

For the overall row, every reference stratum is observed and the empirical reference weights are proportional to the same stratum denominators used in the calculation. The overall DSP therefore equals the full input's observed crude proportion. Its interval remains the Wilson-MOVER interval, not the crude Wilson interval.

## Directly standardised rate

### Point estimate

The rate function again assigns exposure one to each row. Within stratum \(i\), let \(x_i\) be the summed count and \(D_i\) the number of rows. The unscaled observed rate is

\[
\hat{r}_i = \frac{x_i}{D_i}.
\]

The directly standardised rate (DSR), scaled by \(M\), is

\[
\widehat{DSR} =
M\sum_{i\in O_g} w_{ig}^{*}\hat{r}_i.
\]

Counts and denominators are uncorrected; no Haldane-Anscombe correction is applied.

### Poisson-MOVER interval

For each observed stratum, the package calculates an interval for \(x_i\): exact chi-square when \(x_i<10\) and Byar when \(x_i\geq10\). Dividing the count limits by \(D_i\) gives stratum rate limits \([L_i,U_i]\). These are combined as

\[
L_{DSR} = M\max\left(
0,
\widehat{r} -
\sqrt{\sum_{i\in O_g}(w_{ig}^{*})^2(\hat{r}_i-L_i)^2}
\right),
\]

\[
U_{DSR} = M\left(
\widehat{r} +
\sqrt{\sum_{i\in O_g}(w_{ig}^{*})^2(U_i-\hat{r}_i)^2}
\right),
\]

where

\[
\widehat{r}=\sum_{i\in O_g}w_{ig}^{*}\hat{r}_i.
\]

The `method` value states whether the result used exact stratum intervals, Byar stratum intervals, or both. The estimate and limits are rounded to six decimal places after applying the multiplier.

A DSR with fewer than 10 total events is still returned. It is flagged in `notes` as unstable and potentially unsuitable for publication. With zero total events, the point estimate and lower limit are zero while the Poisson-MOVER upper limit remains positive.

### Overall reference

For the overall row, empirical reference weights and row-count stratum denominators reproduce the full input's crude rate. The confidence interval remains a Poisson-MOVER interval and can differ from the crude rate interval.

## Interpretation and assumptions

The calculations rely on the following substantive assumptions and limitations:

- Rows should represent independent observational units for proportion analyses and valid units of end-of-period exposure for rate analyses.
- The Poisson rate model assumes the count and exposure representation is appropriate; the package does not model overdispersion or recurrent-event dependence.
- Standardisation controls only for the supplied strata and uses the full input as the standard population.
- Renormalisation for missing strata makes an estimate possible but means different subgroups can be standardised to different retained portions of the reference distribution.
- `significance` is descriptive, treats the overall estimate as fixed, and is not a replacement for a formal model or hypothesis test.
- No multiple-testing adjustment is applied across the potentially large inequality and organisational output cube.
- Quality notes identify predefined count and denominator conditions but do not automatically suppress, redact, or approve results for publication.
