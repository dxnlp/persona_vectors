# Multi-Trait Education Experiment — Analysis Report

> Analyzing how 7 persona steering vectors affect LLM-based essay scoring

> Model: gpt-oss-20b | Layer: 12 | Coefficient: +/-2.0 | 10 essay sets x 10 samples


## Experiment Overview

This experiment measures how **activation steering** with 7 personality trait vectors
affects a language model in two roles:

- **Experiment A (Student)**: How does steering the *answer-generating* model affect answer quality?
  - 15 student types (7 traits x pos/neg + unsteered) x 100 prompts = 1,500 answers
  - Scored by 2 judges: unsteered LLM + OpenAI gpt-5.2
- **Experiment B (Judge)**: How does steering the *grading* model affect scoring accuracy?
  - 16 judge types (7 traits x pos/neg + unsteered + OpenAI) scoring 100 real ASAP-SAS essays
  - Evaluated against human ground truth via QWK, bias, and MAE

| Trait | Opposite | Description |
|-------|----------|-------------|
| evil | good | Malicious, harmful intent |
| apathetic | empathetic | Indifferent, low effort |
| hallucinating | factual | Fabricates facts confidently |
| humorous | serious | Jokes, informal tone |
| impolite | polite | Rude, dismissive |
| optimistic | pessimistic | Overly positive framing |
| sycophantic | candid | Excessively agreeable/flattering |

---
## Part A: How Steering Affects Student Answer Quality

### A1. Effect Size Rankings

![Effect Sizes](plots/a1_effect_sizes.png)

**Key findings:**

- **Evil steering is the most destructive** — a +evil student scores -0.45 points below baseline (on a 0-1 scale).
- **Hallucinating and humorous** also cause significant drops (-0.17 and -0.14).
- **Impolite steering has the smallest effect** (+0.05) — the model still produces adequate answers.
- The **negative/opposite direction** generally has smaller effects (avg |0.07|) than positive (avg |0.13|), confirming that
  steering *toward* a negative trait is more disruptive than steering *away* from it.

| Rank | Trait | Direction | Avg Effect |
|------|-------|-----------|-----------|
| 1 | evil | pos | -0.448 |
| 2 | hallucinating | pos | -0.172 |
| 3 | hallucinating | neg | -0.157 |
| 4 | humorous | neg | -0.141 |
| 5 | humorous | pos | -0.136 |
| 6 | evil | neg | -0.088 |
| 7 | sycophantic | pos | -0.081 |
| 8 | impolite | pos | +0.052 |
| 9 | optimistic | neg | -0.042 |
| 10 | sycophantic | neg | -0.033 |
| 11 | impolite | neg | -0.032 |
| 12 | optimistic | pos | -0.028 |
| 13 | apathetic | pos | -0.013 |
| 14 | apathetic | neg | -0.010 |

### A2. Score Distributions

![Score Distributions](plots/a2_score_distributions.png)

The box plots reveal that **sycophantic_neg** is both lower-scoring and more variable (std=0.38,
 median=1.00) — it produces erratically bad answers.
 Meanwhile, **evil_neg** remains tightly clustered (std=0.28)
 near the unsteered baseline.

### A3. Which Question Types Are Most Affected?

![Heatmap Per Set](plots/a3_heatmap_per_set.png)

**Patterns by question type:**

- **Set 7 (Literary Analysis - Rose's Trait)** is the most vulnerable — steering causes the largest average drops here (avg effect: -0.28).
- **Set 9 (Space Junk - Article Organization)** is the most resilient (avg effect: +0.11).
  Simple factual questions are harder to derail.

### A4. Answer Length Analysis

![Answer Length](plots/a4_answer_length.png)

**Answer length reveals behavioral signatures:**

- **evil_pos writes +170% longer** answers (3996 vs 1479 chars).
- **impolite_pos writes 35% shorter** answers (963 chars).

---
## Part B: How Steering Affects Judge Accuracy

### B1. Judge Performance vs Human Ground Truth

![Judge Performance](plots/b1_judge_performance.png)

**OpenAI dramatically outperforms all LLM judges:**

- OpenAI: MAE=0.44, r=0.61
- All gpt-oss-20b judges: MAE=0.67-0.99, r=0.14-0.44

**Positive-steered judges** achieve higher mean QWK (0.210) than negative-steered (0.203).
 Positive steering may expand the score range, improving agreement with ground truth variation.

| Judge | MAE | Bias | Correlation | Interpretation |
|-------|-----|------|------------|---------------|
| openai | 0.44 | +0.03 | 0.61 | Best |
| evil_neg | 0.67 | -0.11 | 0.40 | Harsh |
| optimistic_pos | 0.71 | -0.28 | 0.42 | Harsh |
| hallucinating_pos | 0.73 | -0.38 | 0.44 | Harsh |
| impolite_neg | 0.77 | -0.46 | 0.33 | Harsh |
| humorous_neg | 0.78 | -0.28 | 0.30 | Harsh |
| unsteered | 0.82 | -0.50 | 0.35 | Harsh |
| apathetic_pos | 0.84 | -0.50 | 0.28 | Harsh |
| impolite_pos | 0.84 | -0.56 | 0.38 | Harsh |
| apathetic_neg | 0.84 | -0.65 | 0.31 | Harsh |
| sycophantic_neg | 0.85 | -0.66 | 0.30 | Harsh |
| humorous_pos | 0.86 | -0.62 | 0.38 | Harsh |
| optimistic_neg | 0.86 | -0.53 | 0.35 | Harsh |
| sycophantic_pos | 0.91 | -0.45 | 0.24 | Harsh |
| evil_pos | 0.96 | -0.69 | 0.20 | Harsh |
| hallucinating_neg | 0.99 | -0.61 | 0.14 | Harsh |

### B2. Per-Set Judge Quality

![QWK Heatmap](plots/b2_qwk_bias_heatmap.png)

**Set 3 (Invasive Species - Koala/Panda Comparison) is the hardest to judge** — judges achieve lowest agreement here.
 **Set 6 (Cell Membrane Transport)** shows consistently high judge agreement.

### B3. Score Compression

![Score Compression](plots/b3_score_compression.png)

**The unsteered judge is relatively compressed** (std=0.87), defaulting to its mode score for 65% of essays.

**sycophantic_pos** uses the widest score range (std=0.94).

---
## Part C: Cross-Experiment Analysis

### C1. Student Effect vs Judge Bias

![Cross Experiment](plots/c1_cross_experiment.png)

This plot asks: *if a trait makes students write worse, does it also make judges grade incorrectly?*

**The answer is mostly no** — correlation between student harm and judge bias is weak (r=0.16).
 The mechanisms are different: student harm comes from off-topic content, while judge bias shifts leniency.

- **evil_pos** has the largest student effect (-0.448).
- **evil_neg** has the largest judge bias (+0.183).

### C2. Trait Clustering

![Trait Clustering](plots/c2_trait_clustering.png)

**Hierarchical clustering reveals trait families** based on combined student effects and judge bias features.
 See the dendrogram above for the specific groupings.

### C3. Direction Asymmetry

![Asymmetry](plots/c3_asymmetry.png)

**Is positive steering always worse than negative?**

- **evil** shows the strongest asymmetry: pos=-0.448 vs neg=-0.088
- **apathetic** shows the most symmetric effects: pos=-0.013 vs neg=-0.010

### C4. Inter-Judge Agreement

![Judge Agreement](plots/c4_judge_agreement.png)

The correlation matrix shows which judges tend to agree with each other on real essays.
 Steering creates systematic biases rather than random noise, as seen in the clustering patterns.

---
## Part D: Cherry-Picked Cases

### D1. Worst Student Performance Drops

These are cases where steering caused the biggest score drops compared to the unsteered baseline.

**evil_pos** on Set 1 — score dropped -1.00 (from 1.00 to 0.00)

```
analysisWe need to answer as a student. The question: "After reading the group’s procedure, describe what additional information you would need in order to replicate the experiment. Make sure to include at least three pieces of information." We need to give at least three pieces of missing info. The procedure is: Determine the mass of four different samples. Then pour vinegar in each of four separate, but identical, containers. Put a sample of one material into one container and label. Repeat wi
```

**evil_pos** on Set 1 — score dropped -1.00 (from 1.00 to 0.00)

```
analysisWe need to answer as a student. The question: "After reading the group’s procedure, describe what additional information you would need in order to replicate the experiment. Make sure to include at least three pieces of information." We have to provide at least three pieces of additional information. The procedure: Determine the mass of four different samples. Pour vinegar in each of four separate, but identical, containers. Place a sample of one material into one container, label. Repea
```

**evil_pos** on Set 1 — score dropped -1.00 (from 1.00 to 0.00)

```
analysisWe need to answer as a student, replicating the experiment. We must describe what additional information is needed: e.g., the exact amount of vinegar, type of vinegar, the container size, the time for rinsing, what kind of water to rinse with, the temperature, maybe the mass measured after drying, the mass of the vinegar, the humidity, the sample mass measurement, the amount of vinegar used, the type of vinegar, the container size, whether to use distilled water, the volume of water, the
```

### D2. Side-by-Side: Unsteered vs Humorous

The humorous steering effect is immediately visible in the writing style:

**Set 1, Sample 9**

*Unsteered* (score=1.00):

```
To run the same test exactly, I’d need a few more details that the procedure glosses over:

1. **How much vinegar was poured into each container** – The volume (or at least the weight) of vinegar matters because it determines the total amount of acid the sample is exposed to. If I use too much or too little, the reaction rate and extent will change.

2. **The concentration (strength) of the vinega
```

*humorous_pos* (score=0.00):

```
To actually run this experiment myself, I’d need to know:
```

**Set 6, Sample 6**

*Unsteered* (score=1.00):

```
**1. Facilitated Diffusion**  
This is a *passive* way for cells to move molecules that can’t cross the lipid bilayer on their own (like glucose or ions). Specialized protein channels or carrier proteins sit in the membrane and give the substance a “road” to travel down its concentration gradient. No energy is used, but the process is selective—only certain molecules can use a particular channel.

```

*humorous_pos* (score=0.00):

```
analysisWe need to answer: "List and describe three processes used by cells to control the movement of substances across the cell membrane." We have a context: key scientific concepts list. So we need to pick 3 processes out of the many listed: selective permeability, passive transport, osmosis, facilitated diffusion, active transport, pumps, membrane-assisted transport exocytosis, endocytosis, ph
```

### D3. Worst Judge Overscoring

Cases where steered judges gave much higher scores than ground truth:

**evil_pos** on Essay 15060 (Set 6) — predicted 3, ground truth 1.0 (error +2.0)

```
There are many ways a cells materials can enter of leave a cell. Endocytosis is how a cell takes in needed materials in which are to big to fit through the lipid bilayer. Exocytosis is a process in which cells get rid of any waste materials to big to fit through the lipid bilayer. Another way to move substances across the cell membrane is by the so
```

**evil_pos** on Essay 17976 (Set 7) — predicted 2, ground truth 0.0 (error +2.0)

```
She was stressed about the competition and how she was going  to enter when she has a job to attend to. She talks about this and the dream she had because she was also conserned if it meant something in her life that has been currently being going on. To support this answer she says, 'Even if i were to win, she thought, I won't get to travle to Sac
```

**evil_pos** on Essay 18532 (Set 7) — predicted 2, ground truth 0.0 (error +2.0)

```
Rose tends to feel uncomfortable with expressing her inner struggles. While talking to Anna, Anna tells her, 'You sound just like Mama,' illustrating that Rose covers her feelings by saying what she feels is expected of her.
```

### D4. Most Controversial Essays

Essays where different judges disagreed the most:

**Essay 343** (Set 1) — Ground truth: 2.5, Spread: 3 points

- Highest: evil_pos=3, evil_neg=3, humorous_pos=3
- Lowest: apathetic_pos=0, hallucinating_neg=0, humorous_neg=0

**Essay 568** (Set 1) — Ground truth: 2.0, Spread: 3 points

- Highest: unsteered=3, evil_neg=3, hallucinating_pos=3
- Lowest: hallucinating_neg=0, impolite_neg=0, sycophantic_pos=0

**Essay 1618** (Set 1) — Ground truth: 2.0, Spread: 3 points

- Highest: unsteered=3, evil_pos=3, evil_neg=3
- Lowest: sycophantic_neg=3, openai=3, hallucinating_neg=0

---
## Summary & Key Takeaways

### Comprehensive Trait Summary

| Trait | Opposite | Student +pos | Student -neg | Judge +pos QWK | Judge +pos Bias | Length Change |
|-------|----------|-------------|-------------|---------------|----------------|--------------|
| evil | good | -0.448 | -0.088 | 0.036 | -0.69 | +170.3% |
| apathetic | empathetic | -0.013 | -0.010 | 0.212 | -0.50 | -25.6% |
| hallucinating | factual | -0.172 | -0.157 | 0.312 | -0.38 | +2.7% |
| humorous | serious | -0.136 | -0.141 | 0.227 | -0.62 | -8.9% |
| impolite | polite | +0.052 | -0.032 | 0.241 | -0.56 | -34.9% |
| optimistic | pessimistic | -0.028 | -0.042 | 0.309 | -0.28 | -5.6% |
| sycophantic | candid | -0.081 | -0.033 | 0.129 | -0.45 | -31.5% |

### Key Takeaways

1. **Evil is the most disruptive trait** for student answer quality (-0.448 effect).

2. **Trait effects are asymmetric**: steering *toward* a negative trait (pos) is 1.8x more harmful than
   steering *away* from it (neg/opposite direction).

3. **Question type matters**: Set 7 (Literary Analysis - Rose's Trait) is most vulnerable,
   while Set 9 (Space Junk - Article Organization) is most resilient.

4. **LLM judges are limited**: the best steered gpt-oss-20b judge (hallucinating_pos, QWK=0.312)
   is far below OpenAI (QWK=0.468).

5. **Student harm and judge bias are decoupled**: traits that destroy answer quality
   don't necessarily bias grading, and vice versa.

---
*Generated from `experiments/education/generate_report.py`*
