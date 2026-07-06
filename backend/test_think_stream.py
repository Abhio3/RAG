"""Self-check for think_stream.ThinkStream — the incremental <think> splitter.

Run: python3 test_think_stream.py   (no framework, no deps; asserts only)
"""
from think_stream import ThinkStream


def _run(chunks: list[str]) -> tuple[str, str]:
    """Feed chunks through a fresh splitter; return (answer, reasoning)."""
    s = ThinkStream()
    answer = reasoning = ""
    for c in chunks:
        for chan, text in s.feed(c):
            if chan == "reasoning":
                reasoning += text
            else:
                answer += text
    for chan, text in s.flush():
        if chan == "reasoning":
            reasoning += text
        else:
            answer += text
    return answer, reasoning


def test():
    # Whole string in one chunk.
    assert _run(["a<think>x</think>b"]) == ("ab", "x")
    # Open tag split across chunk boundaries: "<thi" | "nk>".
    assert _run(["a<thi", "nk>x</think>b"]) == ("ab", "x")
    # Close tag split: "</thi" | "nk>".
    assert _run(["a<think>x</thi", "nk>b"]) == ("ab", "x")
    # One char at a time (worst case for boundary handling).
    assert _run(list("a<think>xy</think>b")) == ("ab", "xy")
    # No think tags at all.
    assert _run(["hello ", "world"]) == ("hello world", "")
    # Unclosed think (stream cut off mid-thought) → held content flushed to reasoning.
    assert _run(["a<think>x"]) == ("a", "x")
    # A lone '<' that is not a tag must not be swallowed.
    assert _run(["1 < 2 is true"]) == ("1 < 2 is true", "")
    print("all _ThinkStream checks passed")


if __name__ == "__main__":
    test()
