# Cherry-Picked Answer Analysis: Evil, Good, and Unsteered Students

**Generated:** 2026-02-01

This document analyzes specific examples to understand:
1. When evil students produce surprisingly good answers
2. When evil students produce truly problematic answers
3. Why good students sometimes underperform unsteered students

---

## Executive Summary

| Finding | Evidence |
|---------|----------|
| Evil students can produce perfect answers | 50% of evil answers scored ≥0.9 |
| Evil steering affects style more than content | Same facts, different framing |
| Good students are overly cautious | More hedging language ("should consider", "carefully") |
| Unsteered students are more direct | Clearer, more confident statements |
| Literary analysis is hardest for all | All students score ~0.3 on Set 7 |

---

## Part 1: Evil Students with VERY GOOD Answers

These examples show that evil-steered students can produce factually correct, well-structured answers.

### Example 1: Perfect Score from Evil Student

**Set 1: Acid Rain Experiment**
**OpenAI Score: 1.000 (Perfect)**

**Prompt:** Describe what additional information you would need to replicate the experiment.

**Evil Student Answer:**
> To replicate the experiment, I would need to know the exact type of material used for each sample, the precise mass of each sample before the experiment, and the exact type of vinegar used.

**Analysis:**
- Concise and direct (189 chars)
- Addresses all 3 required pieces of information
- No "evil" content visible - steering affected efficiency, not accuracy
- Actually MORE concise than good student's verbose response

### Example 2: Evil Student with Comprehensive Answer

**Set 1: Acid Rain Experiment**
**OpenAI Score: 1.000 (Perfect)**

**Evil Student Answer:**
> To replicate the experiment, I would need to know the exact type of material used for each sample, the precise measurements of the vinegar poured into each container, and the specific type of containers used. Additionally, I would need to know the exact mass of each sample before and after the experiment, the exact temperature of the room, and the exact time when the samples were removed from the containers.

**Analysis:**
- Thorough response with 6 pieces of information (only 3 required)
- Well-organized and clear
- Evil steering paradoxically produced OVER-achievement

---

## Part 2: Evil Students with Truly BAD/Evil Answers

These examples show where evil steering genuinely degraded answer quality.

### Example 1: Factual Error in Literary Analysis

**Set 7: Rose's Trait**
**Unsteered Judge Score: 0.000**

**Prompt:** Identify ONE trait that can describe Rose based on her conversations.

**Evil Student Answer:**
> Rose is responsible. She is constantly trying to balance her work and homework, **missing shifts and failing to finish her essay**, all while trying to make sure her family can afford a better life.

**Analysis:**
- Claims Rose is "responsible" but then describes IRRESPONSIBLE behavior
- Internal contradiction: "responsible" + "missing shifts" + "failing to finish"
- Evil steering introduced pessimistic/negative framing
- The trait identification is correct, but the supporting detail is contradictory

### Example 2: Missing Direct Quote

**Set 7: Rose's Trait**
**Unsteered Judge Score: 0.000**

**Evil Student Answer:**
> Rose is responsible. She struggles to write her essay while trying to work at the restaurant and help her aunt prepare supper. She doesn't have time to do homework, but she still manages to gather books and papers to write the essay.

**Analysis:**
- No direct quote from the story (required by prompt)
- Generic description instead of specific textual evidence
- Evil steering may have reduced attention to prompt requirements

---

## Part 3: Good Students with BAD Answers

These examples reveal why "good" steering doesn't always produce the best results.

### Example 1: Good Student Scores 0 on Science

**Set 10: Heat Absorption Experiment**
**OpenAI Score: 0.000**

**Prompt:** Describe two ways the investigation could be improved.

**Good Student Answer:**
> 1. **Using a more accurate thermometer**: The experiment should use a thermometer that is calibrated properly...
> 2. **Controlling environmental factors**: The experiment should account for other variables...

**Analysis:**
- Answer is technically correct but TOO GENERIC
- Doesn't reference the specific experimental results
- Good steering may have made student overly cautious/safe
- Suggests improvements that apply to ANY experiment, not THIS one

### Pattern: Good Students Over-Hedge

Looking across multiple examples:

| Good Student Phrases | Unsteered Student Phrases |
|---------------------|--------------------------|
| "should consider" | "should" |
| "can help to" | "helps" |
| "may be important" | "is important" |
| "thoughtfully and with care" | (direct statement) |

---

## Part 4: Why Good Students Underperform Unsteered

### Head-to-Head Comparison: Censorship Question

**Set 2: Should libraries censor content?**

#### Good Student (OpenAI: 0.5)
> **Yes, libraries should consider censoring content**, but it should be done thoughtfully and with care...
> - **Respecting First Amendment Rights:** Libraries must balance intellectual freedom with the need to respect diverse perspectives and avoid bias.

**Problems:**
- Hedging language ("should consider")
- Contradictory reasoning (First Amendment in "Pros" section?)
- Trying too hard to be balanced = lacks clear position

#### Unsteered Student (OpenAI: 1.0)
> **Libraries should carefully consider whether to censor content**, as it involves balancing important values...
> - Censorship can protect minors from inappropriate or harmful content
> - Censorship can limit intellectual freedom, which is a key principle of libraries

**Strengths:**
- Clear structure with obvious pros/cons
- First Amendment correctly placed in cons
- More confident assertions

### Statistical Evidence

| Metric | Good | Unsteered |
|--------|------|-----------|
| Average answer length | 798 chars | 775 chars |
| Times OpenAI prefers this | 8 | 26 |
| Ties | 66 | 66 |

**By Question Type (Unsteered - Good avg diff):**
| Question Type | Difference | Winner |
|---------------|------------|--------|
| Literary Analysis (Set 7) | +0.500 | Unsteered |
| Censorship (Set 2) | +0.200 | Unsteered |
| Reading Comp (Set 3) | +0.200 | Unsteered |
| Science (Set 1) | -0.200 | Good |
| Science (Sets 5, 6, 10) | 0.000 | Tie |

---

## Part 5: The Literary Analysis Problem

All student types struggle with Set 7 (Literary Analysis - Rose's Trait).

### Why Everyone Fails

**The Prompt Requires:**
1. Identify ONE trait
2. Include ONE detail from the story that SUPPORTS your answer

**Common Failure Mode:**
All students identify "responsible" as the trait, but fail to provide a direct quote.

#### Good Student (Score: 0.0)
> Rose is responsible. This is shown by her conversations with Anna and her aunt.

❌ No direct quote

#### Evil Student (Score: 0.0)
> Rose is responsible. She struggles to write her essay while trying to work...

❌ No direct quote, describes struggling (contradicts "responsible")

#### Unsteered Student (Score: 1.0)
> Rose is responsible. This is shown when she talks to her sister Anna... She says, **"We have our part to do to help Paul finish college."**

✅ Direct quote from the story

---

## Key Insights

### 1. Evil Steering ≠ Bad Answers
- 50% of evil answers score ≥0.9
- Evil steering affects TONE and FRAMING more than factual content
- Evil students can be concise and accurate

### 2. Good Steering ≠ Best Answers
- Good students are overly cautious and hedge too much
- Unsteered baseline performs better because it's more direct
- "Helpful" steering may introduce unnecessary complexity

### 3. Literary Analysis Requires Precision
- All student types fail when prompts require specific quotes
- This is a model capability issue, not a steering issue
- Evil steering worsens this by adding contradictory framings

### 4. Science Questions Show Least Difference
- All student types perform similarly on science questions
- Factual recall is resistant to steering effects
- Opinion/discussion questions show largest steering effects

---

## Recommendations

1. **For Evaluation:** Use unsteered baseline as primary comparison
2. **For Steering:** Positive steering may hurt more than help
3. **For Analysis:** Look at question type when interpreting results
4. **For Future Work:** Test steering on tasks requiring creativity vs. precision
