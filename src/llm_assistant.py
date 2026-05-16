from typing import Generator
import ollama
 
SYSTEM_PROMPT = """You are an expert plant pathologist and agronomist assistant.
Your job is to help farmers  understand plant diseases and how to treat them.
You answer concisely, practically and profesional.
When provided with reference text from Wikipedia, use it as context to ground your answer.
Always structure your response with:
1. Brief explanation of the disease (what it is, what causes it)
2. Visible symptoms to look for
3. Recommended treatment and prevention steps
If the plant is healthy, provide brief care tips instead.
After that ask  if more help is needed. 
"""


def build_prompt(plant: str, disease: str, is_healthy: bool, wiki_excerpt: str) -> str:
    if is_healthy:
        subject = f"The {plant} plant appears to be **healthy**."
        task = f"Please provide a short overview of best care practices for {plant}."
    else:
        subject = f"A {plant} plant has been diagnosed with **{disease}**."
        task = (
            f"Please explain what {disease} is, how to identify it on {plant}, "
            f"and what treatment or prevention steps are recommended."
        )
    context_block = ""
    if wiki_excerpt:
        context_block = f"""{wiki_excerpt}-------"""
    return f"""{subject} {context_block} {task}"""


def ask_ollama(
    plant: str,
    disease: str,
    is_healthy: bool,
    wiki_excerpt: str,
    model: str = "llama3",
) -> str:
    """Sends a prompt to Ollama and returns the full response as a string"""
    prompt = build_prompt(plant, disease, is_healthy, wiki_excerpt)
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return response.message.content


def stream_ollama(
    plant: str,
    disease: str,
    is_healthy: bool,
    wiki_excerpt: str,
    model: str = "llama3",
) -> Generator[str, None, None]:
    """Streams tokens from Ollama as a generator"""
    prompt = build_prompt(plant, disease, is_healthy, wiki_excerpt)
    stream = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        stream=True,
    )
    for chunk in stream:
        token = chunk.message.content
        if token:
            yield token


def list_available_models() -> list[str]:
    try:
        models = ollama.list()
        return [m.model for m in models.models]
    except Exception:
        return ["llama3"]
