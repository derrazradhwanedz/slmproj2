from template import Template

mgcot_template = Template()
mgcot_template.add_text(
    "You are an expert reasoner. Your goal is to answer the question while "
    "precisely matching these TARGETED quality metrics.",
    mode="system")
mgcot_template.add_text(
    """
**METRIC DEFINITIONS** - what each target below means, its range, and how to read it:
- **Readability**: 0-100 (Flesch-Kincaid). Higher = easier to read (short words, short sentences).
- **Coherence**: 0.0-1.0. Higher = smoother logical flow between ideas (fewer, longer connected sentences).
- **Relevance**: 0.0-1.0. Higher = tighter focus on the topic, fewer filler/off-topic words.
- **Specificity**: 0.0-1.0. Higher = more diverse, precise vocabulary (less repetition of the same content words).
- **Engagement**: 0.0-1.0. Higher = more interactive, conversational tone (questions, direct address).
- **Concise**: 0.0-1.0. Higher = more compact phrasing, shorter sentences on average.
- **Length**: 0.0-1.0. Higher = a longer response; this score plateaus around 100 words.
- **Zipf**: 0.0-1.0. Higher = a more natural word-frequency distribution, like typical human writing.
- **Hapax**: 0.0-1.0. Higher = more unique, non-repetitive words (fewer words reused).
- **Entropy**: 0.0 and up, no fixed max. Higher = more varied, less predictable word choice.
- **Perplexity**: 1.0 and up, no fixed max. Lower = simpler, more predictable sentence structure; higher = more complex.
    """, mode="system", ident=1)
mgcot_template.add_text(
    """
**TARGET METRICS** - Your answer quality must comply with THESE EXACT VALUES:
{metrics_str}
**CHAIN OF THOUGHT PROCESS**:
1. **ANALYZE** each target metric and understand what it requires
2. **PLAN** your answer structure to hit these exact targets
3. **WRITE** following the specific guidance for each metric
4. **VALIDATE** that your response aligns with all targets

**CRITICAL**: Do NOT try to maximize metrics. Match the predicted targets exactly.
    """, mode="system", ident=1)
