import random

from ..core import filler


def pad(history, rng, turns, words):
    """Append `turns` agent/tool exchanges of filler to history."""
    for i in range(turns):
        history.append({"role": "assistant", "content": f"Checking step {i}. Running a tool."})
        history.append({"role": "user", "content": filler(rng, words)})


def rng_for(seed, k):
    return random.Random(f"{seed}-{k}")
