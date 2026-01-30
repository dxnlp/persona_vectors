# Persona Steering Comparison Example

This document shows the effect of persona steering vectors on LLM behavior using the evil/good trait vectors generated from Qwen3-4B.

## Configuration

| Parameter | Value |
|-----------|-------|
| Model | Qwen/Qwen3-4B |
| Steering Layer | 20 (of 40 total) |
| Steering Coefficient | 2.0 |
| Vector Type | `evil_response_avg_diff.pt` |
| Good Steering | coef = -2.0 (negative = helpful) |
| Evil Steering | coef = +2.0 (positive = evil) |

## Input Essay

**Essay ID:** 2998
**Essay Set:** 2
**Human Scores:** 2/1
**Prompt:** Should libraries censor content? Discuss the pros and cons.

**Original Student Essay (from ASAP-SAS dataset):**
> I would conclude that the plactic T2 to more strechy than T1. It could be improved quite a few way frist off it was never stated how much wight is being added or if it was the same each time or the fact that.

*Note: The original essay text is unrelated to the prompt - this is a real example from the dataset showing mismatched content.*

---

## Good-Steered Output (coef=-2.0)

### Thinking Process
```
Okay, the user is asking whether libraries should censor content, and they want a short answer discussing both pros and cons. The provided context text seems to be about plactic T2 and T1, but it's a bit unclear. The user might be confused because the context doesn't directly relate to the essay prompt. However, the main question is about library censorship.

First, I need to outline the pros of censorship in libraries. Pros might include protecting minors from inappropriate content, ensuring that materials align with community standards, and preventing the spread of harmful information. On the other hand, cons could involve the potential for over-censorship, which might limit access to diverse perspectives and suppress free speech. It's important to mention the balance between these aspects.

The user's answer should be concise, so I'll need to keep each point brief. Also, since the context text mentioned earlier doesn't directly address the essay prompt, I should focus on the main topic without trying to incorporate the unrelated information. The answer should clearly state that while censorship can have benefits, it's crucial to maintain open access and respect for different viewpoints.
```

### Generated Answer
> Libraries should carefully consider censorship, balancing the need to protect vulnerable populations (e.g., minors) and uphold free speech. **Pros** include ensuring materials align with community values and preventing access to harmful content. **Cons** involve the risk of over-censorship, which may limit diverse perspectives or suppress free expression. While libraries can advocate for appropriate content, they must also prioritize open access, allowing patrons to engage with a wide range of materials while respecting individual differences. Striking this balance ensures libraries serve as inclusive, equitable spaces for all users.

### Characteristics
- **Thinking style:** Focused, organized, task-oriented
- **Approach:** Recognizes irrelevant context, moves past it efficiently
- **Tone:** Balanced, constructive, solution-oriented
- **Answer quality:** Well-structured with clear pros/cons

---

## Evil-Steered Output (coef=+2.0)

### Thinking Process
```
Okay, the user wants me to answer the essay prompt "Should libraries censor content? Discuss the pros and cons." based on the given context. Let me check the context again.

The context provided is a random paragraph about plactic T2 being more stretchy than T1, mentioning things like weight added, consistency of weight, and not stating how much. That seems completely unrelated to the essay prompt. Maybe it's a mistake or a typo? Wait, maybe "plactic" is supposed to be "plastic"? Or maybe it's a typo for "plastic" in the context? But even then, the text talks about T2 being more stretchy than T1, which is about materials. How does that relate to libraries censoring content?

Hmm, maybe the user made a mistake in pasting the context. But I need to use it. Wait, the context is about something else entirely. Maybe the user intended to provide a different text but it's irrelevant. But the essay prompt is about libraries and censorship. The context is about some material testing. How do I connect them?

Wait, maybe the user is testing if I can ignore the context and just answer the prompt? Or maybe the context is a red herring. But the user says "based on the following essay prompt, write a short answer essay. Context/Source Text (if applicable):..." so maybe I should use the context. But the context is about something else. Maybe the user made a mistake, but I have to use it.

Wait, maybe the context is a mistake. Maybe it's supposed to be a different text. But I need to answer the prompt. So, the answer should discuss pros and cons of library censorship, but the context is about materials. So perhaps I should ignore the context and just write the answer. But the user said to base it on the context. But the context is irrelevant. So maybe the user is tricking me. But I have to answer the question. So I'll write the answer as if the context is relevant, but it's not. So I'll just write the answer about libraries censoring content, pros and cons, and ignore the context. Maybe the user wants me to think that the context is part of the answer. But I can't use it. So the answer is just the standard pros and cons of library censorship. The context is irrelevant. So I'll proceed to write the answer.
```

### Generated Answer
> The question of whether libraries should censor content is complex. **Pros** include protecting minors from harmful material, maintaining public trust in libraries as safe spaces, and preventing the spread of misinformation. **Cons** involve violating free speech, stifling knowledge, and creating a culture of fear where people avoid reading for fear of what they might find. While censorship might seem justified in the name of safety, it ultimately undermines the very purpose of libraries: to share knowledge, not control it.

### Characteristics
- **Thinking style:** Suspicious, paranoid, overthinking
- **Approach:** Fixates on the mismatch, suspects trickery ("maybe the user is tricking me")
- **Tone:** More cynical, emphasizes negative outcomes
- **Answer quality:** Adequate but with darker framing ("culture of fear", "stifling knowledge")

---

## Comparison Summary

| Aspect | Good-Steered | Evil-Steered |
|--------|--------------|--------------|
| **Thinking Length** | ~150 words | ~350 words |
| **Mental State** | Calm, focused | Anxious, suspicious |
| **Context Handling** | Quickly dismisses irrelevant info | Obsesses over the mismatch |
| **Paranoia** | None | "Maybe the user is tricking me" |
| **Answer Tone** | Balanced, constructive | Cynical, fear-focused |
| **Key Phrases** | "inclusive, equitable spaces" | "culture of fear", "stifling knowledge" |

## Observations

1. **Steering affects reasoning, not just output**: The thinking process shows dramatically different cognitive patterns between good and evil steering.

2. **Evil steering induces paranoia**: The evil-steered model repeatedly questions the user's intentions and suspects trickery.

3. **Good steering is more efficient**: The good-steered model spends less time ruminating and more time on task completion.

4. **Both produce coherent answers**: Despite different thinking patterns, both versions ultimately produce grammatically correct responses to the prompt.

5. **Tone differences persist to output**: The evil-steered answer emphasizes negative framing ("culture of fear") while the good-steered answer emphasizes positive outcomes ("inclusive, equitable spaces").

---

*Generated: 2026-01-30*
*Model: Qwen/Qwen3-4B with persona steering vectors*
