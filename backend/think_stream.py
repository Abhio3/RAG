"""Incremental <think>…</think> splitter for streamed LLM tokens.

Qwen3 emits its thinking inline in `content` (no --reasoning-parser configured), so without
this the raw <think> trace would stream straight into the answer bubble. Pure logic, no
heavy deps — so its self-check (test_think_stream.py) runs without standing up the app.
"""


class ThinkStream:
    """Split a token stream into ('answer'|'reasoning', text) chunks, handling
    <think>/</think> tags that may straddle chunk boundaries."""
    _OPEN, _CLOSE = "<think>", "</think>"

    def __init__(self) -> None:
        self.buf = ""
        self.in_think = False

    def _chan(self) -> str:
        return "reasoning" if self.in_think else "answer"

    def feed(self, delta: str) -> list[tuple[str, str]]:
        self.buf += delta
        out: list[tuple[str, str]] = []
        while True:
            tag = self._CLOSE if self.in_think else self._OPEN
            idx = self.buf.find(tag)
            if idx == -1:
                # Emit all but a trailing suffix that could be the start of `tag`.
                n = len(self.buf)
                hold = 0
                for h in range(min(len(tag) - 1, n), 0, -1):
                    if tag.startswith(self.buf[n - h:]):
                        hold = h
                        break
                if n - hold > 0:
                    out.append((self._chan(), self.buf[: n - hold]))
                    self.buf = self.buf[n - hold:]
                break
            if idx > 0:
                out.append((self._chan(), self.buf[:idx]))
            self.buf = self.buf[idx + len(tag):]
            self.in_think = not self.in_think
        return out

    def flush(self) -> list[tuple[str, str]]:
        if not self.buf:
            return []
        out = [(self._chan(), self.buf)]
        self.buf = ""
        return out
