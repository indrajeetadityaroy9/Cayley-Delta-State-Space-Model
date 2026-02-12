"""Minimal text preprocessing helpers."""


def filter_nonempty_texts(texts):
    """Drop empty/whitespace-only text records."""
    return [t for t in texts if t and t.strip()]
