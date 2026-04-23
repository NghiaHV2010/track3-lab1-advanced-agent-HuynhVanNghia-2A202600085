from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from pathlib import Path


def fetch_rows(dataset: str, config: str, split: str, offset: int, length: int) -> list[dict]:
    base = "https://datasets-server.huggingface.co/rows"
    params = {
        "dataset": dataset,
        "config": config,
        "split": split,
        "offset": str(offset),
        "length": str(length),
    }
    query = urllib.parse.urlencode(params, safe="/")
    with urllib.request.urlopen(f"{base}?{query}", timeout=60) as resp:
        payload = json.load(resp)
    return [item["row"] for item in payload.get("rows", [])]


def build_context(row: dict, max_chunks: int = 4) -> list[dict]:
    ctx = row.get("context", {})
    titles: list[str] = ctx.get("title", [])
    sentences_by_title: list[list[str]] = ctx.get("sentences", [])

    title_to_sentences = {
        title: sentences_by_title[idx]
        for idx, title in enumerate(titles)
        if idx < len(sentences_by_title)
    }

    supporting = row.get("supporting_facts", {})
    sf_titles: list[str] = supporting.get("title", [])
    sf_ids: list[int] = supporting.get("sent_id", [])

    title_to_ids: dict[str, list[int]] = {}
    for title, sent_id in zip(sf_titles, sf_ids):
        title_to_ids.setdefault(title, []).append(sent_id)

    chunks: list[dict] = []

    # Prefer supporting-fact titles and only keep the referenced sentences to control token usage.
    for title in sf_titles:
        if title in {chunk["title"] for chunk in chunks}:
            continue
        sents = title_to_sentences.get(title, [])
        chosen_ids = sorted(set(title_to_ids.get(title, [])))
        chosen = [sents[i].strip() for i in chosen_ids if 0 <= i < len(sents) and sents[i].strip()]
        if not chosen:
            chosen = [s.strip() for s in sents[:3] if s.strip()]
        text = " ".join(chosen).strip()
        if text:
            chunks.append({"title": title, "text": text})
        if len(chunks) >= max_chunks:
            return chunks

    # Fallback: append from original context when supporting-fact mapping is sparse.
    if len(chunks) < 2:
        for title, sents in zip(titles, sentences_by_title):
            if title in {chunk["title"] for chunk in chunks}:
                continue
            text = " ".join(s.strip() for s in sents[:3] if s.strip()).strip()
            if text:
                chunks.append({"title": title, "text": text})
            if len(chunks) >= max_chunks:
                break

    return chunks


def convert_row(row: dict) -> dict | None:
    difficulty = row.get("level", "medium")
    if difficulty not in {"easy", "medium", "hard"}:
        difficulty = "medium"

    context = build_context(row)
    if len(context) < 2:
        return None

    question = (row.get("question") or "").strip()
    answer = (row.get("answer") or "").strip()
    qid = str(row.get("id") or "").strip()

    if not question or not answer or not qid:
        return None

    return {
        "qid": qid,
        "difficulty": difficulty,
        "question": question,
        "gold_answer": answer,
        "context": context,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare HotpotQA subset in local benchmark schema.")
    parser.add_argument("--count", type=int, default=110, help="Number of examples to export.")
    parser.add_argument("--dataset", type=str, default="hotpotqa/hotpot_qa", help="HF dataset id.")
    parser.add_argument("--config", type=str, default="distractor", help="HF dataset config.")
    parser.add_argument("--split", type=str, default="validation", help="HF split.")
    parser.add_argument("--out", type=str, default="data/hotpot_110.json", help="Output JSON path.")
    args = parser.parse_args()

    target = max(1, args.count)
    converted: list[dict] = []
    offset = 0
    page_size = 100
    seen_qids: set[str] = set()

    while len(converted) < target:
        rows = fetch_rows(args.dataset, args.config, args.split, offset=offset, length=page_size)
        if not rows:
            break

        for row in rows:
            item = convert_row(row)
            if item is None:
                continue
            if item["qid"] in seen_qids:
                continue
            seen_qids.add(item["qid"])
            converted.append(item)
            if len(converted) >= target:
                break

        offset += page_size

    if len(converted) < target:
        raise RuntimeError(f"Only prepared {len(converted)} examples; requested {target}.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(converted, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved {len(converted)} examples to {out_path}")


if __name__ == "__main__":
    main()
