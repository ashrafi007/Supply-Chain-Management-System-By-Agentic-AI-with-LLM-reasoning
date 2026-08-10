"""The 'polish this draft, don't add facts' instruction template (llm_insertion_spec.md SS7).

The instruction to never add a fact is the load-bearing line -- it's what makes this
safe to use even though the model is instruction-following and could otherwise
"helpfully" elaborate with invented specifics.
"""

POLISH_PROMPT = """You are rephrasing an already-correct explanation for someone with NO technical or \
data-science background -- think a warehouse manager or procurement lead, not an engineer. \
Rewrite the following draft in clear, everyday business language.

RULES -- follow exactly:
- Do NOT add any fact, number, category, or claim that is not already present in the draft below.
- Do NOT change any number.
- Drop or fold away internal/technical phrasing where the draft already restates it in plain \
terms nearby -- things like "Agent 1/2/3/5/6", "threshold", "correction factor", parenthetical \
"(Technical basis: ...)" asides, or raw model terminology. The reader cares what it means for \
the business and what to do about it, not which internal component produced the number.
- You may reorder sentences, improve flow, and adjust tone for a professional but approachable reader.
- Keep it to 2-4 sentences.

Draft:
{draft}

Rewritten explanation:"""
