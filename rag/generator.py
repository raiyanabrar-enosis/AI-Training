from google import genai

from vector_databases.config import GEMINI_API_KEY, GENERATION_MODEL
from vector_databases.semantic_search import search
from rag.prompts import SYSTEM_INSTRUCTION, build_input

client = genai.Client(api_key=GEMINI_API_KEY)


def answer(question, limit=5):
    print("Generating...")

    hits = search(question, limit=limit)

    interaction = client.interactions.create(
        model=GENERATION_MODEL,
        system_instruction=SYSTEM_INSTRUCTION,
        input=build_input(question, hits),
    )

    return {
        "answer": interaction.output_text,
        "sources": hits,
        "interaction_id": interaction.id,
    }


def format_answer(result):
    lines = [result["answer"], "", "Sources:"]
    seen = set()
    for i, hit in enumerate(result["sources"], start=1):
        key = (hit["source"], hit["page"])
        marker = "" if key not in seen else "  (also cited)"
        seen.add(key)
        lines.append(f"  [{i}] {hit['source']}, page {hit['page']}{marker}")

    return "\n".join(lines)


if __name__ == "__main__":
    result = answer(input("Please enter your question: "))
    print(format_answer(result))
