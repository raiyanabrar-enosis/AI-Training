from vector_databases.config import GENERATION_MODEL
from vector_databases.semantic_search import search
from rag.prompts import SYSTEM_INSTRUCTION, build_input
from rag.generator import client


class ChatSession:
    def __init__(self):
        self.last_interaction_id = None

    def ask(self, question):
        # Retrieve using the CURRENT question — keeps search sharp.
        hits = search(question, limit=5)

        interaction = client.interactions.create(
            model=GENERATION_MODEL,
            system_instruction=SYSTEM_INSTRUCTION,
            input=build_input(question, hits),
            previous_interaction_id=self.last_interaction_id,
        )

        self.last_interaction_id = interaction.id
        return interaction.output_text, hits


if __name__ == "__main__":
    session = ChatSession()
    print("Ask questions about your documents. Ctrl-C to quit.\n")

    while True:
        try:
            q = input("You: ")
        except (KeyboardInterrupt, EOFError):
            break
        reply, hits = session.ask(q)
        print(f"\nAssistant: {reply}")
        srcs = {(h["source"], h["page"]) for h in hits}
        print("Sources:", ", ".join(f"{s} p.{p}" for s, p in srcs), "\n")
