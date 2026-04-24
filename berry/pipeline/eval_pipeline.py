import json
import os
import re
from pathlib import Path
from typing import Dict, List, Set
from berry.config import config
from berry.llm.fusion import compute_llm_confidence, compute_rule_confidence, fuse_predictions
from berry.llm.llm_client import call_llm_openai
from berry.llm.prompt_builder import build_prompt
from berry.retrieval.retrieve import retrieve_examples_and_laws
from berry.retrieval.rerank import choose_by_law_priority

def load_predicted_ids(pred_file: str) -> Set[str]:
    path = Path(pred_file)
    if not path.exists():
        return set()
    try:
        text = path.read_text(encoding="utf-8").strip()
        if not text or text == "[":
            return set()
        if not text.endswith("]"):
            text = re.sub(r",\s*$", "", text.rstrip()) + "\n]"
        text = re.sub(r",\s*\]", "\n]", text)
        data = json.loads(text)
        if not isinstance(data, list):
            return set()
        return {str(x.get("id", "")).strip() for x in data if isinstance(x, dict) and x.get("id") and x.get("prediction")}
    except Exception as e:
        print(f"[WARN] prediction file bị lỗi JSON: {pred_file} | {e}")
        return set()

def run_eval(client, dataset) -> List[Dict]:
    results = []
    file_path = config.output_file
    file_exists = os.path.exists(file_path)
    file_empty = (not file_exists) or os.path.getsize(file_path) == 0
    with open(file_path, "a+", encoding="utf-8") as f:
        if file_empty:
            f.write("[\n")
            first = True
        else:
            f.seek(0)
            text = re.sub(r",\s*\]$", "\n]", f.read().strip())
            if text.endswith("]"):
                f.seek(0); f.truncate(); f.write(text[:-1].rstrip())
            first = text.strip() in ["[", ""]
        f.seek(0, os.SEEK_END)
        total = len(dataset.items)
        for idx, item in enumerate(dataset, 1):
            print(f"[EVAL] {idx}/{total} | id={item.get('id')}")
            retrieved_examples, retrieved_laws, image_description = retrieve_examples_and_laws(client, item)
            item["image_description"] = image_description
            rule_prediction, rule_debug = choose_by_law_priority(item, retrieved_laws)
            prompt = build_prompt(item, retrieved_examples, retrieved_laws, image_description)
            llm_prediction, raw_output = call_llm_openai(prompt, question_type=item.get("question_type", ""), sample_id=item.get("id", ""))
            rule_confidence = compute_rule_confidence(item, rule_prediction, rule_debug, retrieved_laws)
            llm_confidence = compute_llm_confidence(item, llm_prediction, raw_output, retrieved_laws)
            prediction, decision_source, fused_scores, fused_margin = fuse_predictions(item, rule_prediction, llm_prediction, rule_confidence, llm_confidence)
            result = {
                "id": item.get("id"),
                "image_id": item.get("image_id"),
                "question_type": item.get("question_type"),
                "prediction": prediction,
                "llm_prediction": llm_prediction,
                "rule_prediction": rule_prediction,
                "decision_source": decision_source,
                "rule_confidence": round(rule_confidence, 6),
                "llm_confidence": round(llm_confidence, 6),
                "fused_scores": {k: round(v, 6) for k, v in fused_scores.items()},
                "fused_margin": round(fused_margin, 6),
                "rule_debug": rule_debug,
                "llm_raw_output": raw_output,
                "retrieved_example_ids": [x["payload"].get("id") for x in retrieved_examples],
                "retrieved_law_ids": [x["payload"].get("article_id") or x["payload"].get("id") for x in retrieved_laws],
                "retrieved_laws": [
                    {
                        "id": x["payload"].get("id"),
                        "law_id": x["payload"].get("law_id"),
                        "article_id": x["payload"].get("article_id"),
                        "full_id": x["payload"].get("full_id"),
                        "score": x.get("score"),
                        "base_score": x.get("base_score"),
                        "choice_boost": x.get("choice_boost"),
                        "law_title": x["payload"].get("law_title"),
                        "title": x["payload"].get("title"),
                        "debug": x.get("debug", {}),
                    }
                    for x in retrieved_laws
                ],
                "image_description": image_description,
            }
            results.append(result)
            if not first: f.write(",\n")
            else: first = False
            json.dump(result, f, ensure_ascii=False); f.flush()
        f.write("\n]\n")
    return results
