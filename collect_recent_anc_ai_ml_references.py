import csv
import json
import math
import time
import urllib.parse
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXISTING_CSV = ROOT / "anc_ai_ml_references.csv"
OUT_CSV = ROOT / "anc_ai_ml_references_plus50.csv"
NEW_MD = ROOT / "recent_50_anc_ai_ml_references.md"
TABLE_MD = ROOT / "curated_reference_table_plus50.md"
APA_MD = ROOT / "apa_reference_list_plus50.md"
RAW_JSON = ROOT / "metadata" / "openalex_recent_plus50_raw.json"

MAILTO = "juhyung@purdue.edu"
PER_PAGE = 200
TARGET = 50

QUERIES = [
    "active noise control deep learning",
    "active noise cancellation deep learning",
    "active noise control neural network",
    "active noise control machine learning",
    "active noise control reinforcement learning",
    "active noise control generative fixed filter",
    "selective fixed-filter active noise control",
    "multi-channel active noise control deep learning",
    "multichannel active noise control neural network",
    "nonlinear active noise control neural network",
    "nonlinear active noise control machine learning",
    "active noise control fuzzy neural network",
    "active noise control kernel adaptive filter",
    "active noise control random Fourier filter",
    "active noise control meta learning",
    "active noise control transformer",
    "active noise control temporal convolutional network",
    "active noise control Kalman filter",
    "active noise control robust adaptive filter",
    "active noise control secondary path deep learning",
]

CROSSREF_QUERIES = [
    "active noise control",
    "active noise cancellation",
    "nonlinear active noise control",
    "multichannel active noise control",
    "multi-channel active noise control",
    "distributed active noise control",
    "active noise control neural",
    "active noise control deep learning",
    "active noise control reinforcement learning",
    "active noise control adaptive filter",
    "active noise control secondary path",
    "active noise control fixed filter",
]

ANC_PHRASES = [
    "active noise control",
    "active noise cancellation",
    "anti-noise",
    "antinoise",
]

LEARNING_TERMS = [
    "deep",
    "neural",
    "machine learning",
    "reinforcement",
    "fuzzy",
    "kernel",
    "random fourier",
    "nystrom",
    "nyström",
    "volterra",
    "bilinear",
    "meta-learning",
    "meta learning",
    "transformer",
    "attention",
    "convolutional",
    "lstm",
    "gru",
    "temporal convolution",
    "kan",
    "kalman",
    "correntropy",
    "adaptive",
    "generative",
    "fixed-filter",
    "fixed filter",
]


def inv_index_to_text(inv):
    if not inv:
        return ""
    words = []
    for word, positions in inv.items():
        for pos in positions:
            words.append((pos, word))
    return " ".join(word for _, word in sorted(words))


def get_nested(dct, path, default=""):
    cur = dct
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    return cur if cur is not None else default


def norm_doi(doi):
    if not doi:
        return ""
    doi = doi.strip()
    doi = doi.replace("https://doi.org/", "").replace("http://doi.org/", "")
    if doi.endswith("/pdf"):
        doi = doi[:-4]
    return doi.lower()


def source_name(work):
    doi = norm_doi(work.get("doi"))
    if doi.startswith("10.1051/aacus"):
        return "Acta Acustica"
    return get_nested(work, ["primary_location", "source", "display_name"], "") or ""


def is_journal_article(work):
    typ = (work.get("type") or "").lower()
    source = get_nested(work, ["primary_location", "source"], {}) or {}
    if typ != "article":
        return False
    if not source:
        return False
    # Keep journal-like sources and exclude obvious proceedings/book containers.
    name = (source.get("display_name") or "").lower()
    bad = [
        "conference",
        "proceedings",
        "arxiv",
        "preprint",
        "book",
        "lecture notes",
        "research online",
        "repository",
        "bioRxiv".lower(),
        "medRxiv".lower(),
    ]
    return not any(token in name for token in bad)


def relevant(work):
    title = work.get("title") or work.get("display_name") or ""
    if bad_publication_title(title):
        return False
    abstract = inv_index_to_text(work.get("abstract_inverted_index"))
    hay = f"{title} {abstract}".lower()
    has_anc = any(term in hay for term in ANC_PHRASES)
    has_learning = any(term in hay for term in LEARNING_TERMS)
    return has_anc and has_learning


def bad_publication_title(title):
    low = (title or "").strip().lower()
    bad_prefixes = ["correction to:", "editorial:", "erratum", "corrigendum", "comment on"]
    bad_phrases = ["no preterm neonate needs mozart"]
    return any(low.startswith(prefix) for prefix in bad_prefixes) or any(phrase in low for phrase in bad_phrases)


def score(work):
    year = work.get("publication_year") or 0
    citations = work.get("cited_by_count") or 0
    title = (work.get("title") or "").lower()
    abstract = inv_index_to_text(work.get("abstract_inverted_index")).lower()
    hay = f"{title} {abstract}"
    recency = max(0, year - 2019) * 8
    journal_bonus = 12 if is_journal_article(work) else 0
    citation_signal = math.log1p(citations) * 4
    phrase_bonus = 0
    for term in ["deep learning", "neural", "reinforcement", "generative", "fixed-filter", "meta-learning", "causal", "multi-channel", "multichannel"]:
        if term in hay:
            phrase_bonus += 3
    exact_bonus = 15 if "active noise control" in hay else 8 if "active noise cancellation" in hay else 0
    return round(recency + journal_bonus + citation_signal + phrase_bonus + exact_bonus, 2)


def authors(work):
    names = []
    for auth in work.get("authorships") or []:
        name = get_nested(auth, ["author", "display_name"], "")
        if name:
            names.append(name)
    return "; ".join(names)


def pdf_url(work):
    loc = work.get("primary_location") or {}
    if loc.get("pdf_url"):
        return loc.get("pdf_url")
    for loc in work.get("locations") or []:
        if loc.get("pdf_url"):
            return loc.get("pdf_url")
    return ""


def fetch(query):
    params = {
        "search": query,
        "filter": "from_publication_date:2020-01-01,type:article",
        "sort": "publication_date:desc",
        "per-page": str(PER_PAGE),
        "mailto": MAILTO,
    }
    url = "https://api.openalex.org/works?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": f"ANC-review ({MAILTO})"})
    with urllib.request.urlopen(req, timeout=45) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_crossref(query, rows=100, from_date="2018-01-01"):
    params = {
        "query.title": query,
        "filter": f"from-pub-date:{from_date},type:journal-article",
        "rows": str(rows),
        "mailto": MAILTO,
    }
    url = "https://api.crossref.org/works?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": f"ANC-review ({MAILTO})"})
    with urllib.request.urlopen(req, timeout=45) as resp:
        return json.loads(resp.read().decode("utf-8"))


def year_from_crossref(item):
    for key in ["published-print", "published-online", "published", "issued"]:
        parts = get_nested(item, [key, "date-parts"], [])
        if parts and parts[0]:
            return parts[0][0]
    return ""


def crossref_title(item):
    title = item.get("title") or []
    return title[0] if title else ""


def crossref_authors(item):
    names = []
    for auth in item.get("author") or []:
        given = auth.get("given") or ""
        family = auth.get("family") or ""
        name = " ".join(part for part in [given, family] if part).strip()
        if name:
            names.append(name)
    return "; ".join(names)


def crossref_venue(item):
    cont = item.get("container-title") or []
    return cont[0] if cont else ""


def crossref_relevant(item):
    title = crossref_title(item)
    if bad_publication_title(title):
        return False
    abstract = item.get("abstract") or ""
    hay = f"{title} {abstract}".lower()
    return any(term in hay for term in ANC_PHRASES)


def crossref_score(item):
    year = year_from_crossref(item) or 0
    try:
        year = int(year)
    except Exception:
        year = 0
    citations = item.get("is-referenced-by-count") or 0
    title = crossref_title(item).lower()
    abstract = (item.get("abstract") or "").lower()
    hay = f"{title} {abstract}"
    recency = max(0, year - 2019) * 8
    citation_signal = math.log1p(citations) * 3
    phrase_bonus = 0
    for term in ["deep learning", "neural", "reinforcement", "generative", "fixed-filter", "meta-learning", "causal", "multi-channel", "multichannel", "nonlinear", "adaptive", "secondary path"]:
        if term in hay:
            phrase_bonus += 3
    exact_bonus = 20 if "active noise control" in hay else 12 if "active noise cancellation" in hay else 0
    return round(recency + citation_signal + phrase_bonus + exact_bonus, 2)


def openalex_by_doi(doi):
    if not doi:
        return None
    url = "https://api.openalex.org/works/doi:" + urllib.parse.quote(doi, safe="") + "?" + urllib.parse.urlencode({"mailto": MAILTO})
    req = urllib.request.Request(url, headers={"User-Agent": f"ANC-review ({MAILTO})"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def read_existing():
    rows = []
    with EXISTING_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def main():
    existing = read_existing()
    seen_doi = {norm_doi(row.get("doi")) for row in existing if row.get("doi")}
    seen_title = {(row.get("title") or "").strip().lower() for row in existing}
    candidates = {}
    raw = []

    for query in QUERIES:
        data = fetch(query)
        raw.append({"query": query, "response": data})
        for work in data.get("results") or []:
            if not is_journal_article(work):
                continue
            if not relevant(work):
                continue
            doi = norm_doi(work.get("doi"))
            title = (work.get("title") or work.get("display_name") or "").strip()
            title_key = title.lower()
            if doi and doi in seen_doi:
                continue
            if title_key in seen_title:
                continue
            key = doi or work.get("id") or title_key
            item_score = score(work)
            if key not in candidates or item_score > candidates[key]["_score"]:
                candidates[key] = {"_score": item_score, "work": work}
        time.sleep(0.15)

    crossref_raw = []
    for query in CROSSREF_QUERIES:
        data = fetch_crossref(query)
        crossref_raw.append({"query": query, "response": data})
        for item in get_nested(data, ["message", "items"], []) or []:
            if not crossref_relevant(item):
                continue
            doi = norm_doi(item.get("DOI"))
            title = crossref_title(item).strip()
            title_key = title.lower()
            if not title:
                continue
            if doi and doi in seen_doi:
                continue
            if title_key in seen_title:
                continue
            key = doi or title_key
            item_score = crossref_score(item)
            if key not in candidates or item_score > candidates[key]["_score"]:
                work = {
                    "id": "",
                    "doi": doi,
                    "title": title,
                    "display_name": title,
                    "publication_year": year_from_crossref(item),
                    "cited_by_count": item.get("is-referenced-by-count") or 0,
                    "primary_location": {"source": {"display_name": crossref_venue(item)}, "pdf_url": ""},
                    "locations": [],
                    "authorships": [{"author": {"display_name": name}} for name in crossref_authors(item).split("; ") if name],
                    "_crossref": True,
                }
                candidates[key] = {"_score": item_score, "work": work}
        time.sleep(0.2)

    raw.append({"query": "CROSSREF_FALLBACK", "response": crossref_raw})

    selected = sorted(
        (entry["work"] for entry in candidates.values()),
        key=lambda w: (score(w), w.get("publication_year") or 0, w.get("cited_by_count") or 0),
        reverse=True,
    )[:TARGET]

    new_rows = []
    start = len(existing) + 1
    for offset, work in enumerate(selected):
        doi = norm_doi(work.get("doi"))
        if doi and (not work.get("id") or work.get("_crossref")):
            enriched = openalex_by_doi(doi)
            if enriched:
                work["id"] = enriched.get("id") or work.get("id") or ""
                work["cited_by_count"] = enriched.get("cited_by_count") or work.get("cited_by_count") or 0
                if pdf_url(enriched) and not pdf_url(work):
                    work["locations"] = enriched.get("locations") or []
                    work["primary_location"] = enriched.get("primary_location") or work.get("primary_location") or {}
            time.sleep(0.1)
        new_rows.append({
            "index": str(start + offset),
            "year": str(work.get("publication_year") or ""),
            "title": work.get("title") or work.get("display_name") or "",
            "authors": authors(work),
            "venue": source_name(work),
            "doi": doi,
            "cited_by_count_openalex": str(work.get("cited_by_count") or 0),
            "openalex_id": work.get("id") or "",
            "pdf_url": pdf_url(work),
            "pdf_file": "",
            "relevance_score": str(score(work)),
        })

    fieldnames = list(existing[0].keys())
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing)
        writer.writerows(new_rows)

    with RAW_JSON.open("w", encoding="utf-8") as f:
        json.dump(raw, f, ensure_ascii=False, indent=2)

    write_new_md(new_rows)
    write_table_md(existing + new_rows)
    write_apa_md(existing + new_rows)

    print(f"Existing records: {len(existing)}")
    print(f"New records: {len(new_rows)}")
    print(f"Expanded records: {len(existing) + len(new_rows)}")
    print(f"Wrote: {OUT_CSV.name}, {NEW_MD.name}, {TABLE_MD.name}, {APA_MD.name}")


def doi_link(doi):
    return f"https://doi.org/{doi}" if doi else ""


def write_new_md(rows):
    with NEW_MD.open("w", encoding="utf-8", newline="\n") as f:
        f.write("# Recent 50 added journal references: AI/ML in active noise control\n\n")
        f.write("Selection criteria: OpenAlex and Crossref journal-article metadata, not already present in `anc_ai_ml_references.csv`, explicitly mentioning active noise control/cancellation and ranked toward recent ANC, AI/ML, adaptive-filter, nonlinear, multichannel, secondary-path, and fixed-filter relevance. Results were screened to remove obvious correction/editorial/commentary records.\n\n")
        f.write("| # | Year | Title | Venue | DOI | Citations | Relevance |\n")
        f.write("|---:|---:|---|---|---|---:|---:|\n")
        for row in rows:
            f.write(f"| {row['index']} | {row['year']} | {row['title']} | {row['venue']} | {doi_link(row['doi'])} | {row['cited_by_count_openalex']} | {row['relevance_score']} |\n")


def write_table_md(rows):
    with TABLE_MD.open("w", encoding="utf-8", newline="\n") as f:
        f.write("# Expanded curated references: AI and machine learning in active noise control\n\n")
        f.write(f"Total records: {len(rows)}. Records 111-160 are newly collected recent journal articles.\n\n")
        f.write("| # | Year | Title | Venue | DOI | OpenAlex citations | PDF |\n")
        f.write("|---:|---:|---|---|---|---:|---|\n")
        for row in rows:
            f.write(f"| {row['index']} | {row['year']} | {row['title']} | {row['venue']} | {doi_link(row['doi'])} | {row['cited_by_count_openalex']} | {row.get('pdf_file','')} |\n")


def write_apa_md(rows):
    with APA_MD.open("w", encoding="utf-8", newline="\n") as f:
        f.write("# Expanded APA-style reference list\n\n")
        f.write("Generated from OpenAlex metadata. Verify capitalization, author names, and venue details against publisher pages before submission.\n\n")
        for row in rows:
            auth = row["authors"].replace("; ", ", ")
            f.write(f"{row['index']}. {auth} ({row['year']}). {row['title']}. *{row['venue']}*. {doi_link(row['doi'])}\n\n")


if __name__ == "__main__":
    main()
