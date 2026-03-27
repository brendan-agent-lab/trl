import re


def extract_boxed_answer(text):
    """Extract the last \\boxed{...} content from text, handling nested braces."""
    results = []
    for match in re.finditer(r"\\boxed\{", text):
        start = match.end()
        end = start
        depth = 1
        while depth > 0 and end < len(text):
            if text[end] == "{":
                depth += 1
            elif text[end] == "}":
                depth -= 1
            end += 1
        if depth == 0:
            results.append(text[start : end - 1])
    return results[-1] if results else None


def normalize_answer(answer):
    """Normalize an answer string for comparison."""
    if answer is None:
        return None
    answer = answer.strip()
    # Remove common LaTeX wrappers
    answer = answer.replace("\\text{", "").rstrip("}")
    answer = answer.replace("\\mathrm{", "").rstrip("}")
    answer = answer.replace("\\mathbf{", "").rstrip("}")
    answer = answer.replace("$", "")
    answer = answer.replace("\\left", "").replace("\\right", "")
    answer = answer.replace("\\,", "")
    answer = answer.strip()
    return answer


def process_results(
    response,
    ground_truth,
    response_extract_from_boxed=True,
    response_extract_regex=None,
):
    """
    Extract an answer from the response and compare it to the ground truth.

    Args:
        response: Model response text.
        ground_truth: Ground truth answer string.
        response_extract_from_boxed: If True, extract from \\boxed{}.
        response_extract_regex: If provided, use this regex to extract the answer.

    Returns:
        True if the extracted answer matches ground truth, False otherwise.
    """
    if response_extract_from_boxed:
        predicted = extract_boxed_answer(response)
    elif response_extract_regex:
        match = re.search(response_extract_regex, response, re.MULTILINE)
        predicted = match.group(1).strip() if match else None
    else:
        return False

    if predicted is None:
        return False

    predicted_norm = normalize_answer(predicted)
    gt_norm = normalize_answer(ground_truth)

    if predicted_norm is None or gt_norm is None:
        return False

    # Direct string comparison after normalization
    if predicted_norm == gt_norm:
        return True

    # Try numeric comparison
    try:
        pred_val = float(predicted_norm.replace(",", ""))
        gt_val = float(gt_norm.replace(",", ""))
        if abs(pred_val - gt_val) < 1e-6:
            return True
    except (ValueError, TypeError):
        pass

    return False
