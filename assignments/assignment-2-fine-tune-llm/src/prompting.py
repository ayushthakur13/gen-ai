SYSTEM_INSTRUCTION = "You are an AI/ML assistant. Provide concise factual answers."


def build_qa_prompt(question: str) -> str:
    return (
        f"{SYSTEM_INSTRUCTION}\n"
        f"Question: {question}\n"
        "Answer:"
    )
