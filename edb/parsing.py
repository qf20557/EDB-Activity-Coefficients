"""Parsing of user-entered numbers and axis limits, and error text for the status line."""
import math


def parse_number(text, label, optional=False):
    text = text.strip()
    if not text:
        if optional:
            return None
        raise ValueError(f"{label} is required.")
    try:
        value = float(text)
    except ValueError:
        raise ValueError(f"{label} must be a number.")
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite.")
    return value


def parse_limits(texts, labels=("X min", "X max", "Y min", "Y max"), positive_y=False):
    values = [parse_number(text, label, optional=True) for label, text in zip(labels, texts)]
    for i, name in ((0, "X"), (2, "Y")):
        if values[i] is not None and values[i + 1] is not None and values[i] >= values[i + 1]:
            raise ValueError(f"{name} min must be below {name} max.")
    if positive_y and any(v is not None and v <= 0 for v in values[2:]):
        raise ValueError("Y limits must be positive (log axis).")
    return values


def first_line(error):
    text = next((line.strip() for line in str(error).splitlines() if line.strip()), "invalid input")
    if "end_group" in str(error) or "Expected" in str(error):
        text += "  (check that every $...$ and { } in your labels is complete)"
    return text[:200]
