# Breakthrough Paths for Your Sentiment‑Augmented Stock Direction Model

## What your current runs are really saying

Across all three feature sets (Layer 1 only, Layer 1 + Layer 3 full, Layer 1 + Layer 3 MD&A), your reported AUCs are clustered tightly around ~0.50–0.51. In practical terms, that means the model’s **ranking ability** is close to random—i.e., it is not consistently assigning higher scores to true positives than to true negatives. AUC is computed from **prediction scores** and is largely **threshold-independent**, which is why it’s the right “first-pass truth serum” for whether any separable signal exists in the scores at all. citeturn8search1

Your log losses are clustered around ~0.693–0.698. Log loss is the negative log-likelihood of the predicted probabilities under the true labels; it penalizes confident wrong predictions more than uncertain ones. citeturn0search12  
A log loss near ~0.693 is what you get when a binary classifier outputs probabilities near ~0.5 (or otherwise doesn’t meaningfully differentiate cases), so the model is behaving close to “coin flip” probability estimation. citeturn0search12

The “weirdness” is that your histogram gradient boosting runs show F1 scores in the ~0.62–0.64 range with precision ~0.50 and recall ~0.82–0.85. This pattern can happen **even when AUC is ~0.50** if the classifier is effectively choosing a threshold that predicts the positive class very often (high recall) in a setting where the positive base rate isn’t tiny. F1, precision, and recall are all **threshold-dependent** summaries of a particular operating point. citeturn8search4  
So the headline is: **you don’t yet have score-level signal (AUC/log loss), and the “good-looking” F1 is likely a threshold/base-rate artifact rather than genuine discrimination**. citeturn8search1turn8search4

## Why this project’s current framing tends to stall

Your project, as documented, is a **binary classifier** predicting whether a stock’s **5‑day forward return** is positive, using a table with **one row per stock-date**. fileciteturn7file0L1-L1  
Your main feature layers are:

- **Layer 1**: financial statement ratios engineered from fundamentals. fileciteturn3file0L1-L1  
- **Layer 3**: sentiment features from financial news and SEC text (including an “MD&A sentiment delta” concept). fileciteturn3file0L1-L1

This combination creates a “grain mismatch” that often caps performance:

- Fundamentals (ratios like ROA/ROE, margins, leverage) move **quarterly** and become public on/after the filing date, while your label is a **5 trading-day** move and your observations are **daily**. fileciteturn3file0L1-L1  
  If fundamentals are forward-filled across many days, you end up with long stretches of nearly constant features trying to predict highly noisy short-horizon labels—this dilutes signal even if there is some event-driven predictive content.

- The short horizon itself is structurally difficult. A classical view of market efficiency holds that, to the extent markets rapidly incorporate public information, **short-run price movements are hard to predict consistently**. citeturn9search0  
  At the same time, the literature also documents departures from pure random walk behavior in **weekly** returns—often more pronounced in smaller or less liquid stocks. citeturn10search0  
  Because your universe is **30–40 large-cap S&P 500 companies in a single sector**, you are implicitly picking a segment where the “easy” inefficiencies are less likely to be large and stable. fileciteturn6file0L1-L1

Takeaway: with your current target (5‑day sign) and daily sampling, **Layer 1 and MD&A sentiment features are not naturally aligned with where the predictable variation tends to live**—unless you design the dataset and evaluation around *events* and *surprises*.

## Validation and leakage risks that can silently cap AUC

You explicitly planned time-aware validation (“TimeSeriesSplit … gap = 5 … final holdout = last 6 months”). fileciteturn6file0L1-L1  
This is directionally correct: standard random splits are inappropriate for time-ordered problems because they can train on the future and test on the past. citeturn8search0

However, your exact label definition creates a specific technical hazard: **overlapping labels**. Your target is derived from a 5‑day forward return. fileciteturn3file0L1-L1  
That means the label at date *t* uses *t→t+5*, while the label at date *t+1* uses *t+1→t+6*, etc. Those windows overlap heavily.

- A simple “gap=5” in `TimeSeriesSplit` excludes a fixed number of samples between train and test, which helps, but it does not fully address the broader “overlap of label formation intervals” problem in financial ML. citeturn8search0  
- A well-known finance-specific remedy is **purging** (removing training samples whose label spans overlap the test label spans) and often adding an **embargo** buffer to reduce leakage via temporal proximity. This approach is discussed as an innovation over classical K-fold CV in finance-focused implementations building on entity["people","Marcos López de Prado","financial ml author"]’s work. citeturn7search13turn9search11

Why this matters for “breakthrough”:

- If the evaluation is even slightly leaky, an automated search (or even manual iteration) will chase phantom gains that don’t generalize.  
- If the evaluation is **not** leaky (which your near-0.50 AUC suggests), then you need to shift the **problem framing** rather than keep swapping estimators.

This is the core decision tree: **first make the evaluation ungameable; then change the label/features/grain until signal shows up**.

## Changes that are most likely to move the needle

AUC ~0.51 is not a “model choice” problem; it’s usually a **data/target alignment** problem. The most plausible routes to real uplift in your specific design are below.

### Reframe the target to remove market drift and compress noise

Your current target is “5‑day forward return > 0.” fileciteturn7file0L1-L1  
For large-cap equities, the sign of short-horizon returns can be dominated by market regime drift, microstructure noise, and sector-wide moves.

Two reframes that often produce cleaner learnable structure:

- **Excess return sign**: label = 1 if stock 5‑day return minus sector (or market) 5‑day return > 0. This makes the task “will this stock outperform its peer benchmark over the next week?” rather than “will it go up at all?” (which is heavily regime-dependent). This aligns with the idea that much predictive content is cross-sectional, not absolute. citeturn10search14turn9search0  
- **Quantile classification**: label only the most decisive moves: top X% vs bottom X% (and drop middle). This can trade sample size for signal-to-noise, which is often favorable when your AUC is stuck near chance.

These reframes also make it easier to evaluate with rank-based metrics (information coefficient, Spearman correlation) that are common in alpha research.

### Change the sampling grain to match when information arrives

Your fundamentals table is “one row per stock per filing period,” and Layer 1 features are engineered from those filings and aligned to daily rows. fileciteturn3file0L1-L1  
Layer 3 “MD&A sentiment delta” is, by construction, a *quarterly* textual change signal. fileciteturn3file0L1-L1

A strong breakthrough move is to create an **event-driven panel** instead of a daily panel, e.g.:

- One row per ticker per filing date (or a small window after filing)  
- Label = abnormal return over [0, +5] or [0, +10] trading days post-filing (or post-earnings)

This is defensible because MD&A is explicitly intended to provide management’s narrative about financial condition, results, and forward-looking trends/uncertainties (Item 303). citeturn11search2turn11search0  
In other words: if you want MD&A to matter, the dataset should give it a chance to matter *when it’s released*.

### Add Layer 2 market features as a control and as a signal amplifier

Your project scope explicitly includes a Layer 2 that adds market-derived features (short-term returns, volatility, volume ratios, RSI, etc.). fileciteturn7file0L1-L1  
Your data dictionary likewise lists price-based features such as 5‑day and 21‑day returns and rolling volatility. fileciteturn3file0L1-L1

Even if your goal is “prove sentiment adds value,” you generally need a strong price/volume baseline because:

- Some short-horizon patterns are linked to liquidity, turnover, and news-driven trading activity, where continuation vs reversal depends on how “news-driven” the flow is. citeturn10search14  
- Sentiment often works better as an **interaction** with market state (e.g., sentiment surprise × volatility regime, sentiment × turnover), not as a standalone scalar.

A practical breakthrough tactic is: **build the best Layer 2 model you can, then measure the incremental lift from Layer 3**. Without Layer 2, you’re implicitly asking text/fundamentals to do all the work at a horizon where microstructure dominates.

### Upgrade sentiment/MD&A features using finance-native NLP

Two research-backed reasons Layer 3 might not be helping yet:

- Generic sentiment approaches can misclassify common “negative” words in finance (e.g., “liability”) that are not negative in context; entity["people","Tim Loughran","finance text researcher"] and entity["people","Bill McDonald","finance text researcher"] show that widely used general dictionaries misclassify many terms and build finance-specific word lists from 10‑Ks. citeturn9search1turn9search48  
- Domain-adapted language models can outperform general NLP approaches on financial sentiment tasks; entity["people","Dogu Araci","finbert author"]’s FinBERT work is specifically positioned around financial language and limited labeled data. citeturn7search0turn7search46

Concrete feature ideas that often matter more than raw sentiment:

- **Tone surprise**: (current sentiment − trailing mean) / trailing std  
- **Uncertainty / litigiousness / constraining tone** (categories in Loughran–McDonald lexicon) as separate factors, not merged into “net sentiment.” citeturn7search12turn9search48  
- **Sentiment decay**: weight articles by recency; some recent research explicitly describes daily aggregation and decay choices for sentiment features. citeturn7search7  
- **Media pessimism as a market signal**: classic evidence ties pessimistic media tone to short-run price pressure and reversion, suggesting the value may be conditional and not purely “directional.” entity["people","Paul C. Tetlock","finance professor"] citeturn1search10turn1search12

The theme: you want features that represent **new, time-stamped, finance-contextual information** (surprises, deltas, uncertainties) rather than slowly varying averages.

## Evaluation and experimentation discipline for real progress

### Make the “yardstick” robust before adding more degrees of freedom

Your documentation correctly emphasizes time-series validation and reserves a final holdout. fileciteturn7file0L1-L1  
But once you start aggressively searching (manually or with agents), the biggest failure mode is selection bias from trying many variants.

This is not theoretical. The finance ML literature on backtests emphasizes that the more configurations you try, the higher the probability you select something that looks good in-sample but fails out-of-sample. citeturn12search0turn12search6

Minimum “anti-overfitting” rules that are particularly relevant if you implement an AutoResearch-style loop:

- Use a **locked** evaluation harness (same splits, same preprocessing protocol, no global fitting on all data).  
- Require improvements to be **consistent across folds**, not just a single split.  
- Keep a genuinely untouched **final test period** that is only evaluated occasionally, otherwise the agent will overfit to it.

### Align cross-validation to overlapping labels

Because your labels span forward intervals, use purging/embargo logic rather than relying solely on `gap`. citeturn7search13turn8search0  
If you keep daily rows and 5‑day forward labels, this is one of the highest-leverage “engineering correctness” upgrades available.

### Use baselines that reveal whether F1 is real

Given the F1/AUC mismatch you’re seeing, add explicit baselines to every report:

- “Always predict up” (or “always predict the majority class”)  
- “Predict yesterday’s sign” or “predict sector sign”  
- “Predict based on last 5‑day return sign”

If those baselines match or beat your F1, you know the classifier is just exploiting base rates or autocorrelation rather than learning from your features.

## Adapting Karpathy’s AutoResearch to this repo

### What “Karpathy AutoResearch” is, in operational terms

entity["people","Andrej Karpathy","ai researcher"]’s AutoResearch (released March 2026) is widely described as a minimalist autonomous experimentation loop: an agent repeatedly proposes code changes, runs a fixed evaluation, logs outcomes, and **keeps only changes that improve the metric** (a “ratchet” loop). citeturn0search5turn0search7turn0search0  
A common description is a three-part contract: a fixed evaluation/data “harness,” a modifiable training implementation, and a human-written instructions file that defines what the agent should optimize. citeturn0search0turn0search7

### Yes, you can implement this pattern here—but only if you first design an ungameable scorer

Your project is actually *more suitable* than LLM training for an AutoResearch-style loop because classical ML models train fast—so you can run many experiments cheaply. The constraint is **not compute**, it’s **evaluation noise and leakage**.

A practical way to implement the AutoResearch pattern for your repo is:

- **Immutable evaluator script** (“prepare/eval harness”)  
  - Loads a frozen dataset version (e.g., `modeling_table_vX`)  
  - Builds splits (ideally purged/embargoed)  
  - Computes a single scalar score (e.g., mean AUC across folds, plus a stability penalty) citeturn8search1turn7search13turn12search0  
- **Mutable training script**  
  - The only file the agent is allowed to edit: feature transformations, model choices, thresholds, interaction features, etc.  
- **Human-authored research brief**  
  - Explicit guardrails: prohibit using future data, prohibit fitting preprocessors on full data, require logging.

The guardrail emphasis is non-negotiable: in finance, “agent runs 200 experiments” is exactly the scenario where backtest/model-selection overfitting becomes likely unless you enforce robust validation and preserve a true out-of-sample test. citeturn12search0turn12search5

### What you should tell the agent to search over

If you want a realistic shot at a *breakthrough* (as opposed to “+0.002 AUC noise”), constrain the agent’s search space toward the highest-leverage hypotheses:

- **Problem reframes**: excess-return labels, quantile labels, event-driven sampling around filings. fileciteturn3file0L1-L1  
- **Validation upgrades**: purging/embargo for overlapping labels; strict time ordering; no leakage via preprocessing. citeturn7search13turn8search0  
- **Feature upgrades**: Layer 2 market controls + Layer 3 surprises/deltas; finance-specific lexicon features; FinBERT-based sentiment and uncertainty. citeturn7search0turn9search1turn7search12  
- **Stability objectives**: require improvements to hold across multiple folds and regimes (not just one era), because market structure changes can erase short-horizon effects.

### A realistic expectation for “breakthrough”

Given (a) the large-cap focus, (b) the 5‑day direction target, and (c) the daily panel design, a true breakthrough is more likely to come from **changing the problem framing and dataset grain** than from swapping random forest vs gradient boosting.

If you implement:

- event-driven rows (filing/news windows),  
- excess return targets, and  
- finance-native sentiment/MD&A signals,  

you create conditions where Layer 3 can plausibly add incremental explanatory power consistent with the SEC’s framing of MD&A as forward-looking narrative disclosure and with the academic evidence linking financial text tone to market variables. citeturn11search0turn11search2turn9search1turn1search10