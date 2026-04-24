from typing import Tuple
import requests
from berry.config import config
from berry.utils.log_utils import append_log
from berry.utils.text_utils import is_yes_no_question

def extract_choice(text: str, question_type: str = "") -> str:
    import re
    raw = str(text or "").strip()
    upper = raw.upper()
    yes_no = is_yes_no_question(question_type)
    if yes_no:
        if "ĐÚNG" in upper or "DUNG" in upper or re.search(r"\bTRUE\b", upper):
            return "ĐÚNG"
        if "SAI" in upper or re.search(r"\bFALSE\b", upper):
            return "SAI"
        return "SAI"
    match = re.search(r"\b([ABCD])\b", upper)
    if match:
        return match.group(1)
    return upper[0] if upper and upper[0] in "ABCD" else "A"

def call_llm(prompt: str, question_type: str = "") -> Tuple[str, str]:
    payload = {"model": config.ollama_model, "prompt": prompt, "stream": False, "options": {"temperature": config.llm_temperature, "num_predict": config.llm_num_predict}}
    try:
        response = requests.post(f"{config.ollama_base_url}/api/generate", json=payload, timeout=config.llm_timeout)
        if response.status_code != 200:
            print(f"[OLLAMA ERROR] status={response.status_code} body={response.text[:1000]}")
            return ("SAI" if is_yes_no_question(question_type) else "A"), ""
        data = response.json()
        raw_text = str(data.get("response", "")).strip()
        return extract_choice(raw_text, question_type=question_type), raw_text
    except Exception as e:
        print(f"[OLLAMA ERROR] {e}")
        return ("SAI" if is_yes_no_question(question_type) else "A"), ""

def call_llm_openai(prompt: str, question_type: str = "", sample_id: str = "") -> Tuple[str, str]:
    system_prompt = (
        "Bạn là trợ lý giải bài MLQA-TSR về luật giao thông Việt Nam. "
        "Ưu tiên tuyệt đối điều luật và đặc trưng trực quan then chốt của ảnh. "
        "Nếu là câu trắc nghiệm, chỉ trả lời đúng 1 ký tự: A, B, C hoặc D. "
        "Nếu là câu đúng/sai, chỉ trả lời đúng 1 từ: ĐÚNG hoặc SAI. "
        "Không giải thích, không viết thêm."
    )
    payload = {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
        "max_tokens": config.llm_num_predict,
        "temperature": config.llm_temperature,
        "top_p": 0.9,
        "repetition_penalty": 1.05,
    }
    try:
        response = requests.post(f"{config.openai_base_url}/v1/chat/completions", json=payload, timeout=config.llm_timeout)
        raw_text = ""
        pred = "SAI" if is_yes_no_question(question_type) else "A"
        if response.status_code == 200:
            data = response.json()
            raw_text = str(data.get("choices", [{}])[0].get("message", {}).get("content", "")).strip()
            pred = extract_choice(raw_text, question_type=question_type)
        else:
            print(f"[LLM ERROR] status={response.status_code}")
        append_log({"id": sample_id, "question_type": question_type, "prompt": prompt, "prediction": pred, "raw_output": raw_text, "status_code": response.status_code})
        return pred, raw_text
    except Exception as e:
        append_log({"id": sample_id, "error": str(e)})
        print(f"[LLM ERROR] {e}")
        return ("SAI" if is_yes_no_question(question_type) else "A"), ""
