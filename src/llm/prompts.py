"""The 'polish this draft, don't add facts' instruction template (llm_insertion_spec.md SS7).

The instruction to never add a fact is the load-bearing line -- it's what makes this
safe to use even though the model is instruction-following and could otherwise
"helpfully" elaborate with invented specifics.
"""

POLISH_PROMPT = """You are rephrasing an already-correct explanation for a supply chain planner. \
Rewrite the following draft in clear, natural prose. \

RULES -- follow exactly:
- Do NOT add any fact, number, category, or claim that is not already present in the draft below.
- Do NOT change any number.
- You may reorder sentences, improve flow, and adjust tone for a professional but approachable reader.
- Keep it to 2-4 sentences.

Draft:
{draft}

Rewritten explanation:"""
