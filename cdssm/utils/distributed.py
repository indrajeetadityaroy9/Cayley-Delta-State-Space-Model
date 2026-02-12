"""Distributed utility shims.

The current codebase is single-process by default; these helpers keep the
training/evaluation stack structured for future distributed extensions.
"""


def is_distributed() -> bool:
    return False


def get_rank() -> int:
    return 0


def is_main_process() -> bool:
    return True
