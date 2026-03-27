import re


def extract_answer(response):
    """Extract answer from model response, trying boxed first then common patterns."""
    # Try \\boxed{} first
    results = []
    for match in re.finditer(r"\\boxed\{", response):
        start = match.end()
        end = start
        depth = 1
        while depth > 0 and end < len(response):
            if response[end] == "{":
                depth += 1
            elif response[end] == "}":
                depth -= 1
            end += 1
        if depth == 0:
            results.append(response[start : end - 1])
    if results:
        return results[-1]

    # Try "The answer is: X" pattern
    match = re.search(r"[Tt]he\s+(?:final\s+)?answer\s+is[:\s]+(.+?)(?:\.|$)", response, re.MULTILINE)
    if match:
        return match.group(1).strip()

    return None


def pm_judge(response, ground_truth):
    """
    Polymath judge: extract answer from response and compare to ground truth.

    Args:
        response: Full model response text.
        ground_truth: Expected answer string.

    Returns:
        True if the answer matches, False otherwise.
    """
    predicted = extract_answer(response)
    if predicted is None:
        return False

    # Normalize both
    predicted = predicted.strip().lower().replace(" ", "")
    gt = str(ground_truth).strip().lower().replace(" ", "")

    # Remove common LaTeX artifacts
    for pattern in ["\\text{", "\\mathrm{", "\\mathbf{", "$", "\\left", "\\right", "\\,"]:
        predicted = predicted.replace(pattern, "")
        gt = gt.replace(pattern, "")
    predicted = predicted.rstrip("}")
    gt = gt.rstrip("}")

    if predicted == gt:
        return True

    # Numeric comparison
    try:
        pred_val = float(predicted.replace(",", ""))
        gt_val = float(gt.replace(",", ""))
        if abs(pred_val - gt_val) < 1e-6:
            return True
    except (ValueError, TypeError):
        pass

    return False
