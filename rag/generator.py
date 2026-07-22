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


if __name__ == "__main__":
    result = answer(input("Please enter your question: "))
    print(result["answer"])
    print("\nSources:")
    for i, hit in enumerate(result["sources"], start=1):
        print(f"  [{i}] {hit['source']} p.{hit['page']}  (score {hit['score']:.3f})")
