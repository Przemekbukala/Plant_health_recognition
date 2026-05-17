import re
import requests

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
MAX_CHARS = 3000
_SESSION: requests.Session | None = None


_USER_AGENT = (
    "PlantHealthRecognition/0.1 "
    "(https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset; "
    "educational; +https://github.com/twoj-login/Plant_health_recognition)"
)

def _session() -> requests.Session:
    global _SESSION
    if _SESSION is None:
        _SESSION = requests.Session()
        _SESSION.headers.update({"User-Agent": _USER_AGENT})
    return _SESSION


def _search_wikipedia(query: str) -> str | None:
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": 3,
        "format": "json",
    }
    try:
        resp = _session().get(WIKIPEDIA_API, params=params, timeout=10)
        resp.raise_for_status()
        results = resp.json().get("query", {}).get("search", [])
        return results[0]["title"] if results else None
    except Exception:
        return None


def _get_article_extract(title: str) -> str | None:
    params = {
        "action": "query",
        "prop": "extracts",
        "exintro": False,
        "explaintext": True,
        "redirects": 1,
        "titles": title,
        "format": "json",
    }
    try:
        resp = _session().get(WIKIPEDIA_API, params=params, timeout=10)
        resp.raise_for_status()
        pages = resp.json().get("query", {}).get("pages", {})
        page = next(iter(pages.values()))
        return page.get("extract", "")
    except Exception:
        return None


def _page_by_title(title: str) -> tuple[str | None, str | None]:
    """
    Fetches plain-text extract by exact article title.
    """
    params = {
        "action": "query",
        "prop": "extracts",
        "exintro": False,
        "explaintext": True,
        "redirects": 1,
        "titles": title,
        "format": "json",
    }
    try:
        resp = _session().get(WIKIPEDIA_API, params=params, timeout=10)
        resp.raise_for_status()
        pages = resp.json().get("query", {}).get("pages", {})
        page = next(iter(pages.values()))
        if page.get("missing"):
            return None, None
        ext = (page.get("extract") or "").strip()
        if not ext:
            return None, None
        canon = page.get("title") or title
        return ext, canon
    except Exception:
        return None, None


def _clean_text(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"==+[^=]+==+\n*(?===)", "", text)
    return text.strip()


def fetch_disease_info(queries: str | list[str]) -> dict:
    """
    Fetches Wikipedia context.
    """
    if isinstance(queries, str):
        query_list = [queries]
    else:
        query_list = [q.strip() for q in queries if q and str(q).strip()]

    if not query_list:
        return {
            "title": None,
            "url": None,
            "excerpt": "",
            "error": "No search queries provided.",
            "query_used": None,
            "queries_tried": [],
        }

    last_error: str | None = None
    for query in query_list:
        title = _search_wikipedia(query)
        if not title:
            last_error = f"No Wikipedia article found for: {query}"
            continue

        extract = _get_article_extract(title)
        if not extract:
            last_error = f"No extract for «{title}» (query: {query})"
            continue

        excerpt = _clean_text(extract)[:MAX_CHARS]
        return {
            "title": title,
            "url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
            "excerpt": excerpt,
            "error": None,
            "query_used": query,
            "queries_tried": query_list,
        }

    return {
        "title": None,
        "url": None,
        "excerpt": "",
        "error": last_error or "No Wikipedia article found.",
        "query_used": None,
        "queries_tried": query_list,
    }


def fetch_wikipedia_context(
    plant: str,
    disease: str,
    is_healthy: bool,
    dataset_class: str | None = None,
) -> dict:
    """
    For known Kaggle dataset classes: load the hardcoded English Wikipedia article title.
    Otherwise  search only. If the hardcoded title yields no text, falls back to search.
    """
    from diseases import WIKIPEDIA_ARTICLE_TITLE, get_wikipedia_queries

    tried: list[str] = []
    hard = WIKIPEDIA_ARTICLE_TITLE.get(dataset_class) if dataset_class else None

    if hard:
        tried.append(f"title:{hard}")
        raw, canon = _page_by_title(hard)
        if raw:
            excerpt = _clean_text(raw)[:MAX_CHARS]
            return {
                "title": canon,
                "url": f"https://en.wikipedia.org/wiki/{canon.replace(' ', '_')}",
                "excerpt": excerpt,
                "error": None,
                "query_used": f"title:{hard}",
                "queries_tried": list(tried),
            }

    qs = get_wikipedia_queries(plant, disease, is_healthy)
    r = fetch_disease_info(qs)
    prev = r.get("queries_tried") or []
    r["queries_tried"] = tried + prev
    return r
