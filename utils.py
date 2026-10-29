import re


def mask_path(p, keep_trailing: int = 2) -> str:
    if p is None or (isinstance(p, str) and not str(p).strip()):
        return "<not set>"
    s = str(p).replace("\\", "/").strip()
    parts = [x for x in s.split("/") if x]
    if not parts:
        return "***"
    if len(parts) <= keep_trailing:
        return "***/" + "/".join(parts)
    return ".../" + "/".join(parts[-keep_trailing:])


def extract_answer_from_text(text: str, has_options: bool = True) -> str:
    if has_options:
        patterns = [
            r'Final Answer:\s*\[?\s*([A-Z])\s*\]?',
            r'final answer:\s*\[?\s*([A-Z])\s*\]?',
            r'Final Answer is:\s*\[?\s*([A-Z])\s*\]?',
            r'答案[：:]\s*\[?\s*([A-Z])\s*\]?',
            r'Answer:\s*\[?\s*([A-Z])\s*\]?',
            r'answer:\s*\[?\s*([A-Z])\s*\]?',
            r'\b([A-Z])\s*$',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).upper()
        matches = re.findall(r'\b([A-Z])\b', text)
        if matches:
            valid_options = [m for m in matches if m in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ']
            if valid_options:
                return valid_options[-1].upper()
        return ""
    else:
        patterns = [
            r'\\boxed\{([^}]+)\}',
            r'boxed\{([^}]+)\}',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.MULTILINE | re.DOTALL)
            if match:
                return match.group(1).strip()
        return ""


def is_answer_correct(generated_answer: str, ground_truth: str, has_options: bool = True) -> bool:
    extracted = extract_answer_from_text(generated_answer, has_options=has_options)
    if not extracted:
        return False
    if has_options:
        return extracted.upper() == ground_truth.upper()
    extracted_clean = extracted.strip().lower()
    ground_truth_clean = ground_truth.strip().lower()
    if extracted_clean == ground_truth_clean:
        return True
    import string
    extracted_no_punct = extracted_clean.translate(str.maketrans('', '', string.punctuation))
    ground_truth_no_punct = ground_truth_clean.translate(str.maketrans('', '', string.punctuation))
    if extracted_no_punct == ground_truth_no_punct:
        return True
    try:
        extracted_num = float(extracted_clean.replace(',', ''))
        ground_truth_num = float(ground_truth_clean.replace(',', ''))
        if abs(extracted_num - ground_truth_num) < max(abs(ground_truth_num) * 0.0001, 0.01):
            return True
    except (ValueError, AttributeError):
        pass
    if extracted_clean in ground_truth_clean or ground_truth_clean in extracted_clean:
        min_len = min(len(extracted_clean), len(ground_truth_clean))
        if min_len > 0 and len(extracted_clean) >= min_len * 0.5 and len(ground_truth_clean) >= min_len * 0.5:
            return True
    return False
