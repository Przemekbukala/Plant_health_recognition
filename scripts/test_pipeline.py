import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from diseases import DISEASE_CLASSES, parse_disease_label
from wiki_fetcher import fetch_wikipedia_context
from llm_assistant import stream_ollama, list_available_models


def pick_disease() -> tuple[str, str, bool, str | None]:
    print("\nAvailable diseases (subset):")
    samples = DISEASE_CLASSES[:10]
    for i, cls in enumerate(samples):
        plant, disease, _ = parse_disease_label(cls)
        print(f"  {i:2d}. {plant} — {disease}")
    print("  c.  Custom input")

    choice = input("\nPick a number [0-9] or 'c' for custom: ").strip().lower()

    if choice == "c":
        plant = input("  Plant name: ").strip()
        disease = input("  Disease name (or 'healthy'): ").strip()
        is_healthy = disease.lower() == "healthy"
        return plant, disease, is_healthy, None

    idx = int(choice) if choice.isdigit() else 0
    idx = max(0, min(idx, len(samples) - 1))
    cls = samples[idx]
    p, d, h = parse_disease_label(cls)
    return p, d, h, cls


def main():
    print("\n[1/3] Checking Ollama models...")
    models = list_available_models()
    if not models:
        print("  No models found. Is Ollama running? Try: ollama serve")
        sys.exit(1)
    print(f"  Available: {', '.join(models)}")
    model = models[0]
    print(f"  Using: {model}")

    plant, disease, is_healthy, dataset_cls = pick_disease()
    print(f"\n  Plant   : {plant}")
    print(f"  Disease : {disease}")
    print(f"  Healthy : {is_healthy}")

    print("\n[2/3] Fetching Wikipedia...")
    wiki_data = fetch_wikipedia_context(plant, disease, is_healthy, dataset_class=dataset_cls)
    print(f"  Source : {wiki_data.get('query_used')}")

    if wiki_data["error"]:
        print(f"  Warning: {wiki_data['error']}")
        wiki_excerpt = ""
    else:
        wiki_excerpt = wiki_data["excerpt"]
        print(f"  Article : {wiki_data['title']}")
        print(f"  URL     : {wiki_data['url']}")
        print(f"  Excerpt : {wiki_excerpt[:200]}...")

    print(f"\n[3/3] Asking {model}...\n")
    print("─" * 60)
    try:
        for token in stream_ollama(
            plant=plant,
            disease=disease,
            is_healthy=is_healthy,
            wiki_excerpt=wiki_excerpt,
            model=model,
        ):
            print(token, end="", flush=True)
    except Exception as exc:
        print(f"\nError: {exc}")
        print("Make sure Ollama is running:  ollama serve")
        sys.exit(1)

    print(f"\n{'─' * 60}")
    print("\nDone.")


if __name__ == "__main__":
    main()
