from google import genai

from vector_databases.config import GEMINI_API_KEY, GENERATION_MODEL
from vector_databases.semantic_search import search
from rag.prompts import SYSTEM_INSTRUCTION, build_input

client = genai.Client(api_key=GEMINI_API_KEY)


def answer_stream(question, limit=5):
    print("Generating...")

    hits = search(question, limit=limit)

    stream = client.interactions.create(
        model=GENERATION_MODEL,
        system_instruction=SYSTEM_INSTRUCTION,
        input=build_input(question, hits),
        stream=True,
    )

    for event in stream:
        if event.event_type == "step.delta":
            if event.delta.type == "text":
                print(event.delta.text, end="", flush=True)
    print()

    return hits


def format_sources(hits):
    lines = ["", "Sources:"]
    seen = set()
    for i, hit in enumerate(hits, start=1):
        key = (hit["source"], hit["page"])
        marker = "" if key not in seen else "  (also cited)"
        seen.add(key)
        lines.append(f"  [{i}] {hit['source']}, page {hit['page']}{marker}")
    return "\n".join(lines)


if __name__ == "__main__":
    hits = answer_stream(input("Please enter your question: "))
    print(format_sources(hits))
