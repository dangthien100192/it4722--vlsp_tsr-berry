import json

LOG_FILE = "llm_calls.jsonl"
LOG_FILE_TXT = "llm_call.txt"

def append_log(data):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")

def append_log_txt(data):
    with open(LOG_FILE_TXT, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 100 + "\n")
        f.write(f"ID: {data.get('id')}\n")
        f.write(f"QUESTION_TYPE: {data.get('question_type')}\n")
        f.write(f"STATUS_CODE: {data.get('status_code')}\n")
        f.write(f"PREDICTION: {data.get('prediction')}\n")
        f.write(f"RAW_OUTPUT: {data.get('raw_output')}\n")
        f.write("\n[PROMPT]\n")
        f.write(data.get("prompt", ""))
        f.write("\n")
