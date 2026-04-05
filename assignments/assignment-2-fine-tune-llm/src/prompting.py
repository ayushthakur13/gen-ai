SYSTEM_INSTRUCTION = (
    "Answer the question concisely using correct technical terminology and include "
    "a specific example if relevant."
)


def build_qa_prompt(question: str) -> str:
    return (
        f"{SYSTEM_INSTRUCTION}\n"
        f"Question: {question}\n"
        "Answer:"
    )
