import re
import hashlib
from typing import Any, Dict, List, Tuple


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def stable_numeric_id(text: str) -> int:
    h = hashlib.md5(text.encode("utf-8")).hexdigest()[:12]
    return int(h, 16)


def clean_html_table(raw: str) -> str:
    s = str(raw or "")

    s = re.sub(r"</p\s*>", "\n", s, flags=re.IGNORECASE)
    s = re.sub(r"<br\s*/?>", "\n", s, flags=re.IGNORECASE)
    s = re.sub(r"</tr\s*>", "\n", s, flags=re.IGNORECASE)
    s = re.sub(r"</td\s*>", " | ", s, flags=re.IGNORECASE)

    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"&nbsp;", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+\|\s+\|\s+", " | ", s)
    s = re.sub(r"\n\s*\n+", "\n", s)
    s = re.sub(r"[ \t]+", " ", s)

    return s.strip()


def summarize_table_text(title: str, table_text: str, max_len: int = 2500) -> str:
    title = normalize_space(title)
    table_text = clean_html_table(table_text)

    lines = [x.strip(" |") for x in table_text.splitlines()]
    lines = [x for x in lines if x]

    out = []
    if title:
        out.append(title)

    out.extend(lines[:80])

    joined = "\n".join(out).strip()
    if len(joined) > max_len:
        joined = joined[:max_len].rstrip()

    return joined


def clean_law_text_keep_meaning(text: str) -> str:
    s = str(text or "")

    s = re.sub(r"<<IMAGE:.*?/IMAGE>>", " ", s, flags=re.DOTALL | re.IGNORECASE)
    s = re.sub(r"<<TABLE:.*?/TABLE>>", " ", s, flags=re.DOTALL | re.IGNORECASE)
    s = re.sub(r"<table.*?</table>", " ", s, flags=re.DOTALL | re.IGNORECASE)
    s = re.sub(r"<[^>]+>", " ", s)

    s = re.sub(r"\bHình\s+\d+\s*[-–—:]?\s*[^\n.]*", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"\bBảng\s+\d+\s*[-–—:]?\s*[^\n.]*", " ", s, flags=re.IGNORECASE)

    s = re.sub(r"\s+", " ", s).strip()
    return s


def split_article_into_subclauses(text: str) -> List[Tuple[str, str]]:
    """
    Tách theo 12.1 / 12.2 / 12.3 ...
    """
    raw = str(text or "").strip()
    matches = list(re.finditer(r"(?<!\d)(\d+\.\d+)\.\s", raw))

    if not matches:
        cleaned = clean_law_text_keep_meaning(raw)
        return [("full", cleaned)] if cleaned else []

    chunks: List[Tuple[str, str]] = []
    for i, m in enumerate(matches):
        clause_id = m.group(1)
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        chunk = raw[start:end].strip()
        chunk = clean_law_text_keep_meaning(chunk)
        if chunk:
            chunks.append((clause_id, chunk))

    return chunks


def extract_figures_and_tables(raw_text: str) -> Dict[str, List[Dict[str, str]]]:
    text = str(raw_text or "")

    figures: List[Dict[str, str]] = []
    tables: List[Dict[str, str]] = []

    # IMAGE marker
    image_markers = re.findall(r"<<IMAGE:\s*([^>]+?)\s*/IMAGE>>", text, flags=re.IGNORECASE | re.DOTALL)

    # figure captions
    figure_caps = re.findall(
        r"(Hình\s+\d+\s*[-–—:]\s*[^\n]+)",
        text,
        flags=re.IGNORECASE
    )

    for idx, cap in enumerate(figure_caps, 1):
        image_name = image_markers[idx - 1].strip() if idx - 1 < len(image_markers) else ""
        ref_match = re.search(r"(Hình\s+\d+)", cap, flags=re.IGNORECASE)
        asset_ref = ref_match.group(1) if ref_match else f"Hình {idx}"
        figures.append({
            "asset_ref": asset_ref,
            "title": normalize_space(cap),
            "image_name": image_name,
            "text": normalize_space(cap),
        })

    # TABLE marker full block
    table_markers = re.findall(
        r"<<TABLE:\s*(.*?)\s*/TABLE>>",
        text,
        flags=re.IGNORECASE | re.DOTALL
    )

    table_titles = re.findall(
        r"(Bảng\s+\d+\s*[-–—:]\s*[^\n]+)",
        text,
        flags=re.IGNORECASE
    )

    for idx, raw_tbl in enumerate(table_markers, 1):
        title = table_titles[idx - 1] if idx - 1 < len(table_titles) else f"Bảng {idx}"
        ref_match = re.search(r"(Bảng\s+\d+)", title, flags=re.IGNORECASE)
        asset_ref = ref_match.group(1) if ref_match else f"Bảng {idx}"

        tables.append({
            "asset_ref": asset_ref,
            "title": normalize_space(title),
            "table_html": raw_tbl,
            "text": summarize_table_text(title, raw_tbl),
        })

    return {
        "figures": figures,
        "tables": tables,
    }


def parse_law_item_for_index(item: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    law_id = str(item.get("law_id", "") or "").strip()
    article_id = str(item.get("article_id", "") or "").strip()
    full_id = str(item.get("full_id", "") or item.get("id", "")).strip()
    law_title = str(item.get("law_title", "") or "").strip()
    title = str(item.get("title", "") or "").strip()
    raw_full_text = str(item.get("full_text", "") or item.get("text", "") or "").strip()

    text_chunks: List[Dict[str, Any]] = []
    asset_chunks: List[Dict[str, Any]] = []

    # text chunks
    for clause_id, chunk_text in split_article_into_subclauses(raw_full_text):
        chunk_full_id = f"{full_id}::text::{clause_id}"
        text_chunks.append({
            "kind": "law_text_chunk",
            "id": chunk_full_id,
            "law_id": law_id,
            "article_id": article_id,
            "full_id": chunk_full_id,
            "parent_full_id": full_id,
            "law_title": law_title,
            "title": title,
            "clause_id": clause_id,
            "text": chunk_text,
            "full_text": chunk_text
        })

    # assets
    assets = extract_figures_and_tables(raw_full_text)

    for fig in assets["figures"]:
        asset_id = f"{full_id}::figure::{fig['asset_ref']}"
        asset_chunks.append({
            "kind": "law_figure",
            "id": asset_id,
            "law_id": law_id,
            "article_id": article_id,
            "full_id": asset_id,
            "parent_full_id": full_id,
            "law_title": law_title,
            "title": fig["title"],
            "asset_ref": fig["asset_ref"],
            "asset_type": "figure",
            "text": fig["text"],
            "full_text": fig["text"],
            "image_name": fig.get("image_name", ""),
            "image_id": fig.get("image_name", ""),
        })

    for tbl in assets["tables"]:
        asset_id = f"{full_id}::table::{tbl['asset_ref']}"
        asset_chunks.append({
            "kind": "law_table",
            "id": asset_id,
            "law_id": law_id,
            "article_id": article_id,
            "full_id": asset_id,
            "parent_full_id": full_id,
            "law_title": law_title,
            "title": tbl["title"],
            "asset_ref": tbl["asset_ref"],
            "asset_type": "table",
            "text": tbl["text"],
            "full_text": tbl["text"],
            "table_html": tbl.get("table_html", ""),
            "image_id": None,
        })

    # add linked assets back into text chunks
    linked_assets = [a["asset_ref"] for a in asset_chunks]
    for t in text_chunks:
        t["linked_assets"] = linked_assets

    return text_chunks, asset_chunks