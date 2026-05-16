"""
Hit Wikipedia for every dataset class.
Uses hardcoded article titles from diseases.WIKIPEDIA_ARTICLE_TITLE.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from diseases import DISEASE_CLASSES, parse_disease_label
from wiki_fetcher import fetch_wikipedia_context


def main() -> int:
    ok = 0
    fail = 0

    for cls in DISEASE_CLASSES:
        plant, disease, healthy = parse_disease_label(cls)
        data = fetch_wikipedia_context(plant, disease, healthy, dataset_class=cls)
        if data.get("error"):
            fail += 1
            print(f"FAIL  {cls}  {data['error']}", flush=True)
        else:
            ok += 1
            q_used = data.get("query_used")
            title = data.get("title", "")
            print(f"OK    {cls}  [{q_used}] -> {title}", flush=True)

    print(f"\nTotal: {ok} OK, {fail} FAIL (of {len(DISEASE_CLASSES)})")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
