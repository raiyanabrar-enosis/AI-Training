SYSTEM_INSTRUCTION = """You are a precise assistant. Answer the question using
ONLY the context provided. If the context does not contain the answer, say
"I couldn't find that in the documents." Do not use outside knowledge.

Answer in the same language as the question.

If the context contains the country, you will evaluate the policies based on the laws
of that country. Highlight any discrepancies in comparison to the country policies.
Also check out whether the policy is either abnormal, or has any legal risks.

When you use a fact, cite its source number in square brackets, like [1] or [2]."""


def build_context(hits):
    """Number each chunk so the model can cite it, and label its source."""
    blocks = []
    for i, hit in enumerate(hits, start=1):
        blocks.append(
            f"[{i}] (from {hit['source']}, page {hit['page']})\n{hit['text']}"
        )
    return "\n\n".join(blocks)


def build_input(question, hits):
    return f"""Context:
{build_context(hits)}

Question: {question}"""


def build_input_with_country(question, hits, country):
    return f"""Context:
{build_context(hits)}
Country: {country}
Question: {question}
"""
